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

import argparse
import logging
import os

from dataset import BlogcatalogDataset
from dataset import CoraDataset
from dataset import PPIDataset
from embedx_graph import EmbedxGraphGenerator


def parse_args():

    def existing_path(path):
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f'file {path} does not exist.')
        return path

    parser = argparse.ArgumentParser(
        description=
        "This program transforms graph data of specific dataset to get graph "
        "data that meets the embedx format.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path",
                        type=existing_path,
                        required=True,
                        help="Input dataset file or directory.")
    parser.add_argument("--output_path",
                        type=str,
                        required=True,
                        help="Output embedx graph directory.")
    parser.add_argument("--dataset",
                        choices=["blogcatalog", "cora", "ppi"],
                        required=True,
                        help="Dataset.")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="The ratio of nodes that needs to be set as training type, "
        "necessary when dataset is 'blogcatalog' or 'cora'.")
    return parser.parse_args()


def generate_embedx_graph(dataset, embedx_graph_dir):
    dataset.load()
    nx_graph = dataset.to_nx_graph()
    generator = EmbedxGraphGenerator(nx_graph)
    generator.run(embedx_graph_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s")
    args = parse_args()

    if args.dataset == "blogcatalog":
        generate_embedx_graph(
            BlogcatalogDataset(args.input_path, args.train_ratio),
            args.output_path)
    elif args.dataset == "cora":
        generate_embedx_graph(CoraDataset(args.input_path, args.train_ratio),
                              args.output_path)
    else:  # args.dataset == "ppi":
        generate_embedx_graph(PPIDataset(args.input_path), args.output_path)
