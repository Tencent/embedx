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
# Author: Shuting Guo (shutingnjupt@gmail.com)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
import logging

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score


def _train_and_evaluate(train_embeddings,
                        train_labels,
                        test_embeddings,
                        test_labels,
                        average,
                        classifier_name="OneVsRest"):
    np.random.seed(1)

    if classifier_name == "OneVsRest":
        classifer = OneVsRestClassifier(LogisticRegression())
        classifer.fit(train_embeddings, train_labels)
        prob_list = classifer.predict_proba(test_embeddings)
        top_k_list = [sum(l) for l in test_labels]
        predict_list = []
        for i, k in enumerate(top_k_list):
            prob_list_ = prob_list[i, :]
            labels = classifer.classes_[prob_list_.argsort()[-k:]].tolist()
            prob_list_[:] = 0
            prob_list_[labels] = 1
            predict_list.append(prob_list_)
        predict_list = np.asarray(predict_list)
        score = f1_score(test_labels, predict_list, average=average)
        logging.info("{} {} = {:.4f}".format(classifier_name, average, score))

    elif classifier_name == "SGD":
        classifer = MultiOutputClassifier(SGDClassifier(loss="log", n_jobs=10))
        classifer.fit(train_embeddings, train_labels)
        predict_list = classifer.predict(test_embeddings)
        score = f1_score(test_labels, predict_list, average=average)
        logging.info("{} {} = {:.4f}".format(classifier_name, average, score))

    else:
        logging.error("Invalid classifier_name %s.", classifier_name)


def _get_file_list(path):
    if os.path.isfile(path):
        return [path]

    file_list = []
    for home, _, files in os.walk(path):
        for file_name in files:
            file_list.append(os.path.join(home, file_name))
    return file_list


def _load_data_as_dict(path, dtype=float):
    data_dict = collections.defaultdict(list)
    for file_name in _get_file_list(path):
        with open(file_name, "r") as fin:
            for line in fin:
                vec = line.strip().split(' ')
                data_dict[vec[0]] = [dtype(x) for x in vec[1:]]
    return data_dict


def _parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--embedding_file',
                        help='embedding file',
                        required=True)
    parser.add_argument('--train_label_file',
                        help='train label file',
                        required=True)
    parser.add_argument('--test_label_file',
                        help='test label dir',
                        required=True)
    parser.add_argument('--average',
                        default="micro",
                        help='optional: micro, macro')
    parser.add_argument('--classifier_name',
                        default="OneVsRest",
                        help='optional: OneVsRest, SGD')
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    node_embedding = _load_data_as_dict(args.embedding_file, dtype=float)
    train_node_label = _load_data_as_dict(args.train_label_file, dtype=int)
    test_node_label = _load_data_as_dict(args.test_label_file, dtype=int)

    train_embeddings = []
    train_labels = []
    for node in train_node_label:
        assert node in node_embedding
        train_embeddings.append(node_embedding[node])
        train_labels.append(train_node_label[node])

    test_embeddings = []
    test_labels = []
    for node in test_node_label:
        assert node in node_embedding
        test_embeddings.append(node_embedding[node])
        test_labels.append(test_node_label[node])

    logging.info("Running regression...")
    _train_and_evaluate(np.array(train_embeddings), np.array(train_labels),
                        np.array(test_embeddings), np.array(test_labels),
                        args.average, args.classifier_name)


if __name__ == '__main__':
    logging.basicConfig(filename='f1score.log',
                        level=logging.INFO,
                        format="%(asctime)s: %(message)s")
    sys.exit(main())
