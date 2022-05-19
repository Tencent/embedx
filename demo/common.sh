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
# Author: Chunchen Su (chunchen.scut@gmail.com)
#         Shuting Guo (shutingnjupt@gmail.com)
#

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')]: $*" >&2
}

get_average_feature_log() {
    echo average_feature"$1".log
}

get_random_walker_log() {
    echo random_walker"$1".log
}

get_trainer_log() {
    echo trainer"$1".log
}

get_predictor_log() {
    echo predictor"$1".log
}

get_gs_log() {
    echo gs"$1".log
}

get_ps_log() {
    echo ps"$1".log
}

get_wk_log() {
    echo wk"$1".log
}

get_wk_finished() {
    echo wk"$1".finish
}

get_ps_finished() {
    echo ps"$1".finish
}
