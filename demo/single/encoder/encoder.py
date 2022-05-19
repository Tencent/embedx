#! /usr/bin/env python
# -*- encoding: utf-8 -*-
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

GROUP_SHIFT = 48
GROUP_MASK = 0xffff000000000000
FEATURE_MASK = 0x0000ffffffffffff


def get_group_id(encoder_id):
    return (encoder_id & GROUP_MASK) >> GROUP_SHIFT


def get_sub_id(encoder_id):
    return (encoder_id & FEATURE_MASK)


def make_encoder_id(group_id, sub_id):
    return ((group_id << GROUP_SHIFT) | (sub_id & FEATURE_MASK))


if __name__ == "__main__":
    # 类型id
    group_id = 2000
    # 原节点id/特征id
    sub_id = 10
    # 编码后的新节点id/特征id
    encoder_id = make_encoder_id(group_id, sub_id)
    print(encoder_id, get_group_id(encoder_id), get_sub_id(encoder_id))
