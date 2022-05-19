#! /bin/bash
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

export BUILD_DIR_ABS=/your/compilation/output/path/
export TRAINER=$BUILD_DIR_ABS/trainer
export PREDICTOR=$BUILD_DIR_ABS/predictor
export DIST_TRAINER=$BUILD_DIR_ABS/dist_trainer
export AVERAGE_FEATURE_MAIN=$BUILD_DIR_ABS/tools/graph/average_feature_main
export GRAPH_SERVER_MAIN=$BUILD_DIR_ABS/tools/graph/graph_server_main
export GRAPH_CLIENT_MAIN=$BUILD_DIR_ABS/tools/graph/graph_client_main
export CLOSE_SERVER_MAIN=$BUILD_DIR_ABS/tools/graph/close_server_main
export RANDOM_WALKER_MAIN=$BUILD_DIR_ABS/tools/graph/random_walker_main
