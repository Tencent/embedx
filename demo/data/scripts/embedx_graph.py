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

import collections
import logging
import os

from networkx.readwrite import json_graph
import numpy as np


class EmbedxNode:

    def __init__(self, json_node):
        self.json_node = json_node

    @property
    def node_id(self):
        assert "id" in self.json_node
        return self.json_node["id"]

    @property
    def label(self):
        assert "label" in self.json_node
        return self.json_node["label"]

    @property
    def stage(self):
        assert "stage" in self.json_node
        return self.json_node["stage"]

    @property
    def feature(self):
        assert "feature" in self.json_node or "feat" in self.json_node
        if "feature" in self.json_node:
            return self.json_node["feature"]
        return self.json_node["feat"]

    @property
    def valid_feature(self):
        return self.feature

    def has_feature(self):
        return "feature" in self.json_node or "feat" in self.json_node

    @classmethod
    def get_node_feature_str(cls, node):
        if not node.has_feature():
            return ""

        valid_feature_str = " ".join(
            [f"{i}:{feat}" for i, feat in enumerate(node.valid_feature) if feat != 0])
        return f"{node.node_id} {valid_feature_str}"

    @classmethod
    def get_node_label_str(cls, node):
        node_str = str(node.node_id)
        label = node.label
        if isinstance(label, int):
            node_str += " " + str(label)
        elif isinstance(label, list):
            node_str += " " + " ".join([str(int(x)) for x in label])
        elif isinstance(label, np.ndarray):
            label_nnz = np.nonzero(label)[0]
            if np.isscalar(label_nnz):
                node_str += " " + str(np.asscalar(label_nnz.a))
            else:
                node_str += " " + " ".join([str(int(x)) for x in label])
        return node_str


class Context:
    DEFAULT_GROUP_ID = 0
    DEFAULT_GROUP_DIM = 128

    def __init__(self, node_link_graph):
        self.node_link_graph = node_link_graph
        self.__build_adj_list()

    def keys(self):
        return self.adj_list.keys()

    def find_neighbor(self, node_id):
        if node_id not in self.adj_list:
            return None
        return self.adj_list[node_id]

    def total_edge_num(self):
        edge_num = 0
        for neighbors in self.adj_list.values():
            edge_num += len(neighbors)
        return edge_num

    @classmethod
    def get_neighbor_str(cls, src_id, neighbor):
        neighbor_str = " ".join(
            [f"{dst_id}:{weight}" for (dst_id, weight) in neighbor])
        return f"{src_id} {neighbor_str}"

    @classmethod
    def get_group_config_str(cls, max_feature_num):
        return f"{Context.DEFAULT_GROUP_ID} {max_feature_num} " \
               f"{Context.DEFAULT_GROUP_DIM}"

    def __build_adj_list(self):
        self.adj_list = collections.defaultdict(list)
        for (src_id, dst_id) in self.node_link_graph.edges():
            weight = self.node_link_graph[src_id][dst_id]["weight"]
            self.adj_list[src_id].append((dst_id, weight))
            self.adj_list[dst_id].append((src_id, weight))


class EmbedxGraph:

    def __init__(self, nx_graph):
        self.data = json_graph.node_link_data(nx_graph)
        self.context = Context(json_graph.node_link_graph(self.data))
        self.has_feature = self.__has_feature()

    def next_node(self):
        assert "nodes" in self.data
        for json_node in self.data["nodes"]:
            yield EmbedxNode(json_node)

    def statistics(self):
        stats = [("Total edge number", self.context.total_edge_num())]
        total_node_num, train_node_num, test_node_num = 0, 0, 0
        max_label, num_label = 0, 0
        total_feature_num, total_valid_feature_num = 0, 0
        for node in self.next_node():
            total_node_num += 1
            if node.stage == "train":
                train_node_num += 1
            if node.stage == "test":
                test_node_num += 1

            if isinstance(node.label, int):
                max_label = max(max_label, node.label)
                num_label = max(max_label, node.label) + 1
            else:
                max_label = len(node.label) - 1
                num_label = len(node.label)
            if self.has_feature:
                total_feature_num += len(node.feature)
                total_valid_feature_num += len(node.valid_feature)
        stats.append(("Total node number", total_node_num))
        stats.append(("Total train node number", train_node_num))
        stats.append(("Total test node number", test_node_num))
        stats.append(("Total feature number", total_feature_num))
        stats.append(("Total valid feature number", total_valid_feature_num))
        if total_feature_num > 0:
            ratio = 100.0 * (float(total_valid_feature_num) / total_feature_num)
            stats.append(("Valid feature ratio", ratio))
        stats.append(("Max label", max_label))
        stats.append(("Label number", num_label))

        return stats

    def write_context(self, context_path):
        with open(context_path, 'w', encoding='utf-8') as fout:
            for src_id in self.context.keys():
                neighbor = self.context.find_neighbor(src_id)
                neighbor_str = Context.get_neighbor_str(src_id, neighbor)
                fout.write(neighbor_str)
                fout.write("\n")

    def write_node_feature(self, node_feature_path, group_config_path):
        assert self.has_feature
        max_feature_num = 0
        with open(node_feature_path, 'w', encoding='utf-8') as fout:
            for node in self.next_node():
                fout.write(EmbedxNode.get_node_feature_str(node))
                fout.write("\n")

                max_feature_num = max(max_feature_num, len(node.feature))

        with open(group_config_path, 'w', encoding='utf-8') as fout:
            fout.write(Context.get_group_config_str(max_feature_num))
            fout.write("\n")

    def write_node_label(self, label_path, train_label_path, test_label_path):
        with open(label_path, 'w', encoding='utf-8') as label_f, \
            open(train_label_path, 'w', encoding='utf-8') as train_label_f, \
            open(test_label_path, 'w', encoding='utf-8') as test_label_f:
            for node in self.next_node():
                node_str = EmbedxNode.get_node_label_str(node)
                label_f.write(node_str)
                label_f.write("\n")
                if node.stage == "train":
                    train_label_f.write(node_str)
                    train_label_f.write("\n")
                if node.stage == "test":
                    test_label_f.write(node_str)
                    test_label_f.write("\n")

    def __has_feature(self):
        flag = None
        for node in self.next_node():
            if flag is None:
                flag = node.has_feature()
            else:
                tmp_flag = node.has_feature()
                assert flag == tmp_flag
        return flag


class EmbedxGraphGenerator:
    CONTEXT_FILE_NAME = "context.all"
    NODE_FEATURE_FILE_NAME = "node_feature.all"
    GROUP_CONFIG_FILE_NAME = "group_config.txt"
    LABELS_FILE_NAME = "labels.all"
    TRAIN_LABELS_FILE_NAME = "train_labels.all"
    TEST_LABELS_FILE_NAME = "test_labels.all"

    def __init__(self, nx_graph):
        self.embedx_graph = EmbedxGraph(nx_graph)

    def run(self, dst_dir):
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)

        def get_full_path(path):
            return os.path.join(dst_dir, path)

        self.embedx_graph.write_context(
            get_full_path(EmbedxGraphGenerator.CONTEXT_FILE_NAME))

        self.embedx_graph.write_node_label(
            get_full_path(EmbedxGraphGenerator.LABELS_FILE_NAME),
            get_full_path(EmbedxGraphGenerator.TRAIN_LABELS_FILE_NAME),
            get_full_path(EmbedxGraphGenerator.TEST_LABELS_FILE_NAME))

        if self.embedx_graph.has_feature:
            self.embedx_graph.write_node_feature(
                get_full_path(EmbedxGraphGenerator.NODE_FEATURE_FILE_NAME),
                get_full_path(EmbedxGraphGenerator.GROUP_CONFIG_FILE_NAME))

        self.print_graph_statistics()

    def print_graph_statistics(self):
        for stat in self.embedx_graph.statistics():
            logging.info("%s: %s.", stat[0], stat[1])
