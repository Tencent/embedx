#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Tencent is pleased to support the open source community by making embedx
# available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Licensed under the BSD 3-Clause License and other third-party components,
# please refer to LICENSE for details.
#
# Author: Chuan Cheng (chengchuancoder@gmail.com)
#

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import random

from networkx.readwrite import json_graph
from scipy import io as scipy_io
from sklearn.preprocessing import StandardScaler
import networkx as nx
import numpy as np


class Dataset:

    def __init__(self, input_path):
        self.input_path = input_path

    def load(self):
        raise NotImplementedError("Load is not implemented in Dataset")

    def to_nx_graph(self):
        raise NotImplementedError("To_nx_graph is not implemented in Dataset")


class BlogcatalogDataset(Dataset):

    def __init__(self, input_path, train_ratio):
        Dataset.__init__(self, input_path)
        self.input_path = input_path
        self.train_ratio = train_ratio
        self.mat_data = None

    def load(self):
        self.mat_data = scipy_io.loadmat(self.input_path)

    def to_nx_graph(self):
        assert "group" in self.mat_data
        node_labels = self.mat_data["group"].toarray()
        node_ids = list(range(len(node_labels)))
        train_node_ids = set(
            random.sample(node_ids, int(len(node_ids) * self.train_ratio)))

        nx_graph = nx.Graph()
        for node_id in node_ids:
            stage = "train" if node_id in train_node_ids else "test"
            nx_graph.add_node(node_id, stage=stage, label=node_labels[node_id])

        assert "network" in self.mat_data
        edges = self.mat_data["network"].nonzero()
        for src_id, dst_id in zip(edges[0], edges[1]):
            if src_id in train_node_ids and dst_id in train_node_ids:
                stage = "train"
            else:
                stage = "test"
            nx_graph.add_edge(src_id, dst_id, stage=stage, weight=1.0)
        return nx_graph


class CoraDataset(Dataset):
    CONTENT_FILE_NAME = "cora.content"
    CITE_FILE_NAME = "cora.cites"

    def __init__(self, input_path, train_ratio):
        Dataset.__init__(self, input_path)
        self.train_ratio = train_ratio
        self.content_file = os.path.join(self.input_path,
                                         CoraDataset.CONTENT_FILE_NAME)
        self.cite_file = os.path.join(self.input_path,
                                      CoraDataset.CITE_FILE_NAME)
        self.node_label_feature = []
        self.edges = []
        self.node_id_map = {}

    def load(self):
        self.node_label_feature = self.__load_node_label_feature()
        self.edges = self.__load_edges()

    def to_nx_graph(self):
        self.node_id_map = self.__build_node_id_map()

        train_num = len(self.node_id_map) * self.train_ratio

        nx_graph = nx.Graph()
        for node, label, feature in self.node_label_feature:
            node_id = self.node_id_map[node]
            stage = "train" if node_id <= train_num else "test"
            feature = np.array(feature) / (np.sum(feature) + 1e-7)
            nx_graph.add_node(node_id,
                              stage=stage,
                              feature=feature,
                              label=label)

        for src_node, dst_node in self.edges:
            if src_node in self.node_id_map and dst_node in self.node_id_map:
                src_id = self.node_id_map[src_node]
                dst_id = self.node_id_map[dst_node]
                if src_id <= train_num and dst_id <= train_num:
                    stage = "train"
                else:
                    stage = "test"
                nx_graph.add_edge(src_id, dst_id, stage=stage, weight=1.0)
        return nx_graph

    def __load_node_label_feature(self):
        node_label_feature = []
        label_id_map = {}
        with open(self.content_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                fields = line.rstrip().split("\t")
                node, label = fields[0], fields[-1]
                feature = [int(x) for x in fields[1:-1]]
                if label not in label_id_map:
                    label_id_map[label] = len(label_id_map)
                node_label_feature.append((node, label_id_map[label], feature))
        return node_label_feature

    def __load_edges(self):
        edges = []
        with open(self.cite_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                fields = line.rstrip().split("\t")
                assert len(fields) == 2
                edges.append(fields)
        return edges

    def __build_node_id_map(self):
        node_id_map = {}
        node_id = 0
        for node, _, __ in self.node_label_feature:
            if node not in node_id_map:
                node_id_map[node] = node_id
                node_id += 1
        for src_node, dst_node in self.edges:
            if src_node not in node_id_map:
                node_id_map[src_node] = node_id
                node_id += 1
            if dst_node not in node_id_map:
                node_id_map[dst_node] = node_id
                node_id += 1
        return node_id_map


class PPIDataset(Dataset):
    G_FILE_NAME = "ppi-G.json"
    FEATURE_FILE_NAME = "ppi-feats.npy"
    ID_MAP_FILE_NAME = "ppi-id_map.json"
    CLASS_MAP_FILE_NAME = "ppi-class_map.json"

    def __init__(self, input_path):
        Dataset.__init__(self, input_path)
        self.nl_graph = None
        self.feature = None
        self.node_id_map = None
        self.node_label_map = None

    def load(self):

        def get_full_path(file_name):
            return os.path.join(self.input_path, file_name)

        with open(get_full_path(PPIDataset.G_FILE_NAME), 'r',
                  encoding='utf-8') as fin:
            self.nl_graph = json_graph.node_link_graph(json.load(fin))
        raw_feature = np.load(get_full_path(PPIDataset.FEATURE_FILE_NAME))
        self.feature = self.__standardized_feature(raw_feature)

        with open(get_full_path(PPIDataset.ID_MAP_FILE_NAME),
                  'r',
                  encoding='utf-8') as fin:
            json_id_map = json.load(fin)
            self.node_id_map = {int(k): int(v) for k, v in json_id_map.items()}

        with open(get_full_path(PPIDataset.CLASS_MAP_FILE_NAME),
                  'r',
                  encoding='utf-8') as fin:
            json_class_map = json.load(fin)
            self.node_label_map = {int(k): v for k, v in json_class_map.items()}

    def to_nx_graph(self):
        nx_graph = nx.Graph()
        for node in self.nl_graph.nodes():
            node_id = self.node_id_map[node]
            stage = PPIDataset.__get_node_stage(self.nl_graph.node[node])
            label = self.node_label_map[node]
            feature = self.feature[node_id]
            nx_graph.add_node(node_id,
                              stage=stage,
                              label=label,
                              feature=feature)

            src_id = node_id
            for dst_node in self.nl_graph[node]:
                dst_id = self.node_id_map[dst_node]
                stage = self.__get_edge_stage(node, dst_node)
                nx_graph.add_edge(src_id, dst_id, stage=stage, weight=1.0)
        return nx_graph

    def __get_edge_stage(self, src_node, dst_node):
        if (self.nl_graph.node[src_node]["val"] or
                self.nl_graph.node[dst_node]["val"] or
                self.nl_graph.node[src_node]["test"] or
                self.nl_graph.node[dst_node]["test"]):
            return "test"
        return "train"

    def __standardized_feature(self, feature):
        train_nodes = [
            node for node in self.nl_graph.nodes()
            if not self.nl_graph.node[node]["val"] and
            not self.nl_graph.node[node]["test"]
        ]
        train_feature = feature[train_nodes]
        scaler = StandardScaler()
        scaler.fit(train_feature)
        feature = scaler.transform(feature)
        return feature

    @classmethod
    def __get_node_stage(cls, node):
        if node["val"]:
            return "val"
        if node["test"]:
            return "test"
        return "train"
