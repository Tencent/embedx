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

set -e
cd "$(dirname "$0")"
SOURCE_DIR=$(pwd)

readonly DOWNLOAD_FILE="ppi.zip"
readonly URL="https://snap.stanford.edu/graphsage/ppi.zip"
readonly DATASET="ppi"
readonly UNCOMPRESS_DATA="ppi_data"
readonly SCRIPTS_DIR="./scripts"

if [[ ! -f ${DOWNLOAD_FILE} ]]; then
    curl ${URL} --output ${DOWNLOAD_FILE}
fi

# generate embedx format data
rm -rf ${DATASET} ${UNCOMPRESS_DATA}
unzip ${DOWNLOAD_FILE}
mv ${DATASET} ${UNCOMPRESS_DATA}
python -B ${SCRIPTS_DIR}/process_dataset.py \
    --input_path="${SOURCE_DIR}/${UNCOMPRESS_DATA}" \
    --output_path="${SOURCE_DIR}/${DATASET}" \
    --dataset=${DATASET}

bash ${SCRIPTS_DIR}/split_dataset.sh \
    -d "${SOURCE_DIR}/${DATASET}"

rm -rf ${DOWNLOAD_FILE}
rm -rf ${UNCOMPRESS_DATA}
