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
#

cd "$(dirname "$0")"

set -e

scripts=$(ls run_*.sh | grep -v run_all.sh | sort)
for script in ${scripts}; do
  echo --------------------------------
  echo Running "${script}"
  echo --------------------------------
  bash ${script}
done
