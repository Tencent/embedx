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
#         Chunchen Su (chunchen.scut@gmail.com)
#

set -e
cd "$(dirname "$0")"

usage() {
  echo "Usage: $0 [-d DIR <string>]" 1>&2
  exit 1
}

while getopts ":d:" o; do
  case "${o}" in
  d)
    dir=${OPTARG}
    ;;
  *)
    usage
    ;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "${dir}" ]]; then
  usage
fi

readonly dir
echo dir="${dir}"

err() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')]: $*" >&2
}

#######################################
# Split data evenly into multiple parts.
# Arguments:
#   Prefix of data name, which is also the output directory.
# Outputs:
#   Sliced data.
#######################################
split_data() {
  local data="$1".all
  local output="$1"

  if [[ ! -f "${data}" ]]; then
    err "${data} does not exist."
  else
    echo "${data}"
    read -r -a data_line <<<"$(wc -l "${data}")"
    local file_num=8 # magic number
    local each_file_line=$((data_line / file_num + 1))

    split -l $each_file_line "${data}" part_
    mkdir -p "${output}"
    mv part_* "${output}"

    rm -rf "${data}"
  fi
}

echo "========================="
echo "Splitting ${dir} ..."
pushd "${dir}"
split_data context
split_data node_feature
split_data train_labels
split_data test_labels
split_data labels
popd
echo "========================="
