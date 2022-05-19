# !/usr/bin/env python3
# -*- coding : utf-8 -*-
#
# Tencent is pleased to support the open source community by making embedx
# available.
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Licensed under the BSD 3-Clause License and other third-party components,
# please refer to LICENSE for details.
#
# Author : Yuanhang Zou (yuanhang.nju@gmail.com)
# Author : Yong Zhou (zhouyongnju@gmail.com)
""" A hitrate computation tool.
    Example:
        python hit_rate.py --query_embed query_file --item_embed item_file --topk '100,200,500' """

import argparse
import logging
import os
import sys

import faiss
import numpy as np
from sklearn import preprocessing


def _create_option_parser():
    """ Creates an option parser instance to handle command-line options. """

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_embed",
                        type=str,
                        required=True,
                        help='query embedding file with clicked item id')
    parser.add_argument("--item_embed",
                        type=str,
                        required=True,
                        help='item embedding file with item id')
    parser.add_argument("--topk",
                        type=str,
                        required=True,
                        default="100,200,500",
                        help='topK threshold list')
    parser.add_argument("--norm_item",
                        type=bool,
                        required=False,
                        default=False,
                        help='normalize item embedding')
    parser.add_argument("--norm_query",
                        type=bool,
                        required=False,
                        default=False,
                        help='normalize query embedding')
    return parser


def _yield_file(path):
    if not os.path.exists(path):
        logging.error("Path %s not exists.", path)

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                yield os.path.join(root, filename)
    else:
        yield path


def _load_to_list(path):
    item_list = []
    embed_list = []
    for filename in _yield_file(path):
        with open(filename, 'r') as f_r:
            logging.info("Reading file %s", filename)
            for line in f_r:
                stripped_line = line.strip()
                if not stripped_line:
                    logging.error("Empty Line %s.", line)
                    continue

                item_list.append(stripped_line.split()[0])
                embed_list.append(stripped_line.split()[1:])

    logging.info("Loading to list done.")
    return item_list, embed_list


def _slice_to_array(embed_list, row_index, norm):
    """Take out 'row_index' row from embed_list and convert it to array"""

    if not row_index:
        row_index = np.arange(len(embed_list))

    array = np.asarray(embed_list)[row_index, 0:].astype(np.float32)
    logging.info('Array shape is %d, %d', array.shape[0], array.shape[1])

    if norm:
        logging.info("Normalizing embedding matrix ...")
        preprocessing.normalize(array, axis=1, copy=False)

    logging.info("Slicing to array done.")

    return array


def _extract_index(item_index_dict, query_list):
    item_index = []
    query_index = []
    for index, query in enumerate(query_list):
        if query in item_index_dict:
            item_index.append(item_index_dict[query])
            query_index.append(index)
    return item_index, query_index


def _build_faiss_index(database, nprobe=-1, pq_thold=1000000, save_path=None):
    """ Build faiss index using product quantization.

    :param database: embedding matrix for search
    :param nprobe: max probe cell number for searching, bigger is more accurate but slower
    :param pq_thold: database larger than this will be indexed using product quantization
    :param save_path: save index to this path if given
    """

    num_db, dim = database.shape
    index = faiss.IndexFlatIP(dim)  # inner product similarity

    # using product quantization
    if num_db > pq_thold:
        logging.info("Building pq index ...")
        n_subq = int(np.sqrt(dim))
        for i in range(n_subq, dim):
            if dim % i == 0:
                n_subq = i
                break

        # reasonable number of centroids to index n_db vectors
        n_cell = int(4 * np.sqrt(num_db))
        # 8 specifies that each sub-vector is encoded as 8 bits
        n_bits = 8
        index = faiss.IndexIVFPQ(index, dim, n_cell, n_subq, n_bits)

        # default is 1/3 of all centroids
        if nprobe < 0:
            nprobe = int(n_cell / 3) + 1
        index.nprobe = nprobe

    # pylint: disable = no-value-for-parameter
    index.train(database)
    index.add(database)

    if save_path:
        faiss.write_index(index, save_path)

    logging.info("Building faiss index done.")
    return index


def _extract_valid_query_item(query_list, query_embed_list, item_index_dict,
                              norm_query):
    "extrac valid query and item array"
    item_index, query_index = _extract_index(item_index_dict, query_list)
    item_array = np.asarray(item_index).reshape(-1, 1)
    query_array = _slice_to_array(query_embed_list, query_index, norm_query)
    return query_array, item_array


def pretty_print(indexed_query_num, query_num, topk_list, hit_num_list):
    """Pretty-print the hitrate information."""

    info = [
        "(indexed/all = {}/{} = {:.3f}%)".format(
            indexed_query_num, query_num, 100. * indexed_query_num / query_num)
    ]

    info += [
        "top@ {} = {}/{} = {:.3f}%".format(k, hit, indexed_query_num,
                                           100. * hit / indexed_query_num)
        for k, hit in zip(topk_list, hit_num_list)
    ]

    logging.info('\n'.join(info))


def eval_file(query_file, item_index_dict, item_faiss_index, norm_query,
              topk_list):
    """evaluate hit rate for one file"""
    query_list, query_embed_list = _load_to_list(query_file)
    query_array, item_array = _extract_valid_query_item(
        query_list, query_embed_list, item_index_dict, norm_query)

    logging.info("Searching ...")
    # pylint: disable = no-value-for-parameter
    _, index = item_faiss_index.search(query_array, max(topk_list))

    hit_num_list = [0 for k in topk_list]
    for i, k in enumerate(topk_list):
        hit_num_list[i] = np.sum(index[:, :k] == item_array)

    return len(query_list), len(query_array), hit_num_list


def eval_hit_rate(item_path, query_path, norm_item, norm_query, topk_list):
    """evaluate hit rate"""

    # build item faiss
    item_list, item_embed_list = _load_to_list(item_path)
    item_faiss_index = _build_faiss_index(
        _slice_to_array(item_embed_list, [], norm_item))
    item_index_dict = {item: index for index, item in enumerate(item_list)}

    # use 'query' to compute hitrate
    acc_query_num = 0
    acc_indexed_query_num = 0
    acc_hit_num_list = [0 for k in topk_list]
    for query_file in _yield_file(query_path):
        query_num, indexed_query_num, hit_num_list = eval_file(
            query_file, item_index_dict, item_faiss_index, norm_query,
            topk_list)

        for i in range(len(topk_list)):
            acc_hit_num_list[i] += hit_num_list[i]

        acc_query_num += query_num
        acc_indexed_query_num += indexed_query_num

    pretty_print(acc_indexed_query_num, acc_query_num, topk_list,
                 acc_hit_num_list)


def main(argv):
    """main method."""

    parser = _create_option_parser()
    args = parser.parse_args(argv)

    if not args.query_embed or not args.item_embed or not args.topk:
        parser.print_help()
        return 1

    topk_list = [int(k) for k in args.topk.split(',')]
    eval_hit_rate(args.item_embed, args.query_embed, args.norm_item,
                  args.norm_query, topk_list)

    return 0


if __name__ == '__main__':
    logging.basicConfig(filename='hit_rate.log',
                        level=logging.INFO,
                        format="%(asctime)s: %(message)s")
    sys.exit(main(sys.argv[1:]))
