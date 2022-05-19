# -*- coding: utf-8 -*-

import os
from collections import defaultdict


class CSR(object):
    def __init__(self):
        self.row_ = 0
        self.row_offset = [0]
        self.col = []
        self.val = []

    def emplace(self, key, val):
        self.col.append(key)
        self.val.append(val)

    def add_row(self):
        self.row_ += 1
        self.row_offset.append(len(self.col))


def load(dirname):
    d = defaultdict(list)
    for fname in os.listdir(dirname):
        fpath = os.path.join(dirname, fname)
        with open(fpath) as ifh:
            for line in ifh:
                line = line.strip()
                items = line.split()
                node = int(items[0])
                for item in items[1:]:
                    key, val = item.split(":")
                    key = int(key)
                    val = float(val)
                    d[node].append((key, val))
    return d


context_dir = "context"
feature_dir = "feature"

graph = load(context_dir)
feat_map = load(feature_dir)
