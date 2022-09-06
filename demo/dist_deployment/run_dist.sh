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

################################################################
# Configuration area, modify as needed
################################################################
readonly GS_MACHINES=(
    [0]=9.141.195.101
    [1]=9.141.198.7
)

readonly DIST_MACHINES=(
    [0]=9.141.200.143
    [1]=9.141.201.105
)

readonly GS_SHARD_NUM=${#GS_MACHINES[@]}
readonly DIST_SERVER_NUM=${#DIST_MACHINES[@]}
readonly DIST_WORKER_NUM=$((2 * ${#DIST_MACHINES[@]}))
# NOTE: ensure that the port is not occupied.
readonly FREE_PORT=60000

# Provide a directory that 'GS_MACHINES' and 'DIST_MACHINES' can access.
# The HDFS directory is used as an example.
export HADOOP_USER_NAME=hdpxxx
readonly HADOOP_HOME_DIR=hdfs://mmsearchhadoop-nn1.wx.com:9000/user/hdpxxx/workspace

# The following files must be stored in the 'HADOOP_HOME_DIR'.
readonly HADOOP_GRAPH_SERVER_MAIN=${HADOOP_HOME_DIR}/graph_server_main
readonly HADOOP_CLOSE_SERVER_MAIN=${HADOOP_HOME_DIR}/close_server_main
readonly HADOOP_DIST_TRAINER=${HADOOP_HOME_DIR}/dist_trainer
readonly HADOOP_GET_DIST_ADDR_MAIN=${HADOOP_HOME_DIR}/get_dist_addr_main
# 'libhdfs.so' can be obtained from the hadoop installation package.
readonly HADOOP_LIBHDFS_SO=${HADOOP_HOME_DIR}/libhdfs.so
# 'ppi' dataset cat be obtained by 'bash embedx/demo/data/prepare_ppi.sh'.
readonly HADOOP_PPI=${HADOOP_HOME_DIR}/ppi

readonly JOB_TOKEN=embedx.${GS_SHARD_NUM}.${DIST_SERVER_NUM}.${DIST_WORKER_NUM}.${FREE_PORT}

# local workspace
readonly WORKSPACE=${HOME}/${JOB_TOKEN}

readonly GRAPH_SERVER_MAIN=${WORKSPACE}/$(basename ${HADOOP_GRAPH_SERVER_MAIN})
readonly CLOSE_SERVER_MAIN=${WORKSPACE}/$(basename ${HADOOP_CLOSE_SERVER_MAIN})
readonly DIST_TRAINER=${WORKSPACE}/$(basename ${HADOOP_DIST_TRAINER})
readonly GET_DIST_ADDR_MAIN=${WORKSPACE}/$(basename ${HADOOP_GET_DIST_ADDR_MAIN})
readonly LIBHDFS_SO=${WORKSPACE}/$(basename ${HADOOP_LIBHDFS_SO})

readonly GS_ADDRS_FILE=GS_ADDRS_FILE
readonly GS_ADDRS_DIR=${HADOOP_HOME_DIR}/gs_addrs
readonly OUT_MODEL_DIR=${HADOOP_HOME_DIR}/out_model
readonly LOG_DIR=${HADOOP_HOME_DIR}/log

################################################################

GS_FLAGS="\
    --gs_shard_num=${GS_SHARD_NUM} \
    --gs_thread_num=10 \
    --node_graph=${HADOOP_HOME_DIR}/ppi/context \
    --node_feature=${HADOOP_HOME_DIR}/ppi/node_feature \
    --success_out=${GS_ADDRS_DIR}"

DIST_FLAGS="\
    --dist=1 \
    --sub_command=train \
    --ps_thread_num=10 \
    --in=${HADOOP_HOME_DIR}/ppi/train_labels \
    --in_model= \
    --model=sup_graphsage \
    --model_config=config=${HADOOP_HOME_DIR}/ppi/group_config.txt;sparse=1;depth=1;dim=128;alpha=0;max_label=1;multi_label=1;num_label=121;use_neigh_feat=0 \
    --instance_reader=sup_graphsage \
    --instance_reader_config=num_neighbors=10;max_label=1;multi_label=1;num_label=121;use_neigh_feat=0 \
    --optimizer=adam \
    --optimizer_config=rho1=0.9;rho2=0.999;alpha=0.001;beta=1e-8 \
    --epoch=1 \
    --batch=32 \
    --target_type=0 \
    --out_model=${OUT_MODEL_DIR}"

################################################################

echo ----------------
echo Preparing hadoop environment...
CDH=/data/qspace/cdh-5.5.0
export JAVA_HOME=${CDH}/jdk1.7.0_51
export JRE_HOME=${JAVA_HOME}/jre
export HADOOP_HOME=${CDH}/hadoop-2.6.0-cdh5.5.0
export HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
CLASSPATH=
HADOOP_CLASSPATH=
JARS=$(find ${HADOOP_HOME}/share -name "*.jar" -type f)
for j in ${JARS}; do
    CLASSPATH=${CLASSPATH}:${j}
    HADOOP_CLASSPATH=${HADOOP_CLASSPATH}:${j}
done
export CLASSPATH=${HADOOP_CONF_DIR}:${CLASSPATH}
export HADOOP_CLASSPATH=${HADOOP_CONF_DIR}:${HADOOP_CLASSPATH}
export PATH=${PATH}:${JAVA_HOME}/bin:${HADOOP_HOME}/bin
export LIBRARY_PATH=${JRE_HOME}/lib/amd64:${JRE_HOME}/lib/amd64/server:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${JRE_HOME}/lib/amd64:${JRE_HOME}/lib/amd64/server:${LD_LIBRARY_PATH}
echo Done
echo ----------------

################################################################

prepare_remote_env() {
    if test "x${GS_ADDRS_DIR}" != x; then
        hadoop fs -rm -r ${GS_ADDRS_DIR}
        hadoop fs -mkdir -p ${GS_ADDRS_DIR}
    fi
    if test "x${OUT_MODEL_DIR}" != x; then
        hadoop fs -rm -r ${OUT_MODEL_DIR}
        hadoop fs -mkdir -p ${OUT_MODEL_DIR}
    fi
    if test "x${LOG_DIR}" != x; then
        hadoop fs -rm -r ${LOG_DIR}
        hadoop fs -mkdir -p ${LOG_DIR}
    fi
}

prepare_bin() {
    local remote_bin=$1
    local local_bin=$2

    echo ----------------
    echo Preparing ${local_bin}...
    if test "x${remote_bin}" != x && test "x${local_bin}" != x; then
        echo Downloading ${local_bin} from ${remote_bin}...
        rm -f ${local_bin}
        hadoop fs -get ${remote_bin} ${local_bin}
        chmod 755 ${local_bin}

    fi
    echo md5sum ${local_bin}
    md5sum ${local_bin}
    echo Done
    echo ----------------
}

upload_file_to_hdfs() {
    local file=$1
    local dir=$2

    echo ----------------
    echo putting ${file} to ${dir}...
    hadoop fs -put ${file} ${dir}
    echo Done
    echo ----------------
}

check_gs() {
    while true; do
        hadoop fs -test -e ${GS_ADDRS_DIR}/${GS_ADDRS_FILE}
        if [ $? -eq 0 ]; then
            echo ${GS_ADDRS_FILE} is avaiable.
            break
        else
            echo ${GS_ADDRS_FILE} not ready, sleeping...
            sleep 5s
        fi
    done

    hadoop fs -get ${GS_ADDRS_DIR}/${GS_ADDRS_FILE} ${GS_ADDRS_FILE}
    gs_shard_num=$(awk -F ";" '{print NF}' ${GS_ADDRS_FILE})
    while true; do
        ready_num=$(hadoop fs -ls ${GS_ADDRS_DIR}/_SUCCESS* | grep SUCCESS | wc -l)
        if [ ${ready_num} -eq ${gs_shard_num} ]; then
            echo Graph server is ready.
            break
        else
            echo Graph server is not ready, sleeping...
            sleep 5s
        fi
    done
}

close_gs() {
    local gs_addrs=$1

    while true; do
        success_num=$(hadoop fs -ls ${OUT_MODEL_DIR}/SUCCESS* | grep SUCCESS | wc -l)
        if [ ${success_num} -eq ${DIST_SERVER_NUM} ]; then
            echo Graph server can be closed.
            break
        else
            echo Graph server cannot be closed, sleeping...
            sleep 5s
        fi
    done

    echo Close Graph server...
    ${CLOSE_SERVER_MAIN} --gs_addrs=${gs_addrs}
    echo Done
}

################################################################
# Don't need to modify
################################################################
MY_IP=$(ifconfig -v eth1 2>/dev/null | awk '/inet /{print $2}')
MY_GS_ID=
MY_DIST_ID=
for i in "${!GS_MACHINES[@]}"; do
    if test "x${MY_IP}" == "x${GS_MACHINES[${i}]}"; then
        MY_GS_ID=${i}
        break
    fi
done
for i in "${!DIST_MACHINES[@]}"; do
    if test "x${MY_IP}" == "x${DIST_MACHINES[${i}]}"; then
        MY_DIST_ID=${i}
        break
    fi
done
if test "x${MY_GS_ID}" == x && test "x${MY_DIST_ID}" == x; then
    # Not scheduled on the intended machine, just exit.
    echo Nothing will be run at ${MY_IP}
    exit 0
fi

################################################################

echo ----------------
echo USER=$(whoami)
echo HOSTNAME=$(hostname)
echo GS_MACHINE_NUM=${#GS_MACHINES[@]}
for i in "${!GS_MACHINES[@]}"; do
    echo [${i}] ${GS_MACHINES[${i}]}
done
echo DIST_MACHINE_NUM=${#DIST_MACHINES[@]}
for i in "${!DIST_MACHINES[@]}"; do
    echo [${i}] ${DIST_MACHINES[${i}]}
done
echo GS_SHARD_NUM=${GS_SHARD_NUM}
echo DIST_SERVER_NUM=${DIST_SERVER_NUM}
echo DIST_WORKER_NUM=${DIST_WORKER_NUM}
echo FREE_PORT=${FREE_PORT}
echo HADOOP_HOME_DIR=${HADOOP_HOME_DIR}
echo WORKSPACE=${WORKSPACE}
echo GS_ADDRS_DIR=${GS_ADDRS_DIR}
echo OUT_MODEL_DIR=${OUT_MODEL_DIR}
echo LOG_DIR=${LOG_DIR}
echo MY_IP=${MY_IP}
echo MY_GS_ID=${MY_GS_ID}
echo MY_DIST_ID=${MY_DIST_ID}
echo ----------------

################################################################

echo ----------------
echo Preparing ${WORKSPACE}...
rm -rf ${WORKSPACE}
mkdir -p ${WORKSPACE}
chmod 777 -R ${WORKSPACE}
cd ${WORKSPACE}
prepare_bin ${HADOOP_GRAPH_SERVER_MAIN} ${GRAPH_SERVER_MAIN}
prepare_bin ${HADOOP_CLOSE_SERVER_MAIN} ${CLOSE_SERVER_MAIN}
prepare_bin ${HADOOP_DIST_TRAINER} ${DIST_TRAINER}
prepare_bin ${HADOOP_GET_DIST_ADDR_MAIN} ${GET_DIST_ADDR_MAIN}
prepare_bin ${HADOOP_LIBHDFS_SO} ${LIBHDFS_SO}
echo Done
echo ----------------

################################################################

get_scheduler_addr() {
    awk -F "=" '/^scheduler_addr=/{printf("%s", $2);}' $1
}

get_server_addrs() {
    awk -F "=" '/^server_addrs=/{printf("%s", $2);}' $1
}

get_server_id() {
    awk -F "=" '/^server_id=/{printf("%s", $2);}' $1
}

run_gs_scheduler() {
    if test ${MY_IP} == ${SCHEDULER_IP}; then
        prepare_remote_env

        export ROLE=scheduler
        echo Run gs scheduler at node ${MY_GS_ID}
        local log=gs_scheduler.log
        ${GET_DIST_ADDR_MAIN} >${log} 2>&1 &&
            echo "$(get_server_addrs ${log})" >${GS_ADDRS_FILE} &&
            upload_file_to_hdfs ${GS_ADDRS_FILE} ${GS_ADDRS_DIR} &&
            upload_file_to_hdfs ${log} ${LOG_DIR} &
        sleep 1s
    fi
}

run_gs() {
    for ((i = 0; i < ${GS_SHARD_NUM}; ++i)); do
        export ROLE=server
        export SERVER_ID=${i}
        node_id=$((${SERVER_ID} % ${#GS_MACHINES[@]}))
        if test ${MY_GS_ID} == ${node_id}; then
            echo Run gs ${SERVER_ID} at node ${MY_GS_ID}
            local log=gs${SERVER_ID}.log
            ${GET_DIST_ADDR_MAIN} >${log} 2>&1 &&
                local GS_FLAGS=${GS_FLAGS}" --gs_addrs=$(get_server_addrs ${log})" &&
                local GS_FLAGS=${GS_FLAGS}" --gs_shard_id=$(get_server_id ${log})" &&
                echo ${GS_FLAGS} >>${log} &&
                ${GRAPH_SERVER_MAIN} ${GS_FLAGS} >>${log} 2>&1 &&
                upload_file_to_hdfs ${log} ${LOG_DIR} &
        fi
    done
}

run_cs() {
    local gs_addrs=$(<${GS_ADDRS_FILE})
    if test ${MY_IP} == ${SCHEDULER_IP}; then
        export ROLE=scheduler
        echo Run cs at node ${MY_DIST_ID}
        local log=cs.log
        ${GET_DIST_ADDR_MAIN} >${log} 2>&1 &&
            upload_file_to_hdfs ${log} ${LOG_DIR} &&
            close_gs ${gs_addrs} &
        sleep 1s
    fi
}

run_ps() {
    local gs_addrs=$(<${GS_ADDRS_FILE})
    for ((i = 0; i < ${DIST_SERVER_NUM}; ++i)); do
        export ROLE=server
        export SERVER_ID=${i}
        node_id=$((${SERVER_ID} % ${#DIST_MACHINES[@]}))
        if test ${MY_DIST_ID} == ${node_id}; then
            echo Run ps ${SERVER_ID} at node ${MY_DIST_ID}
            local log=ps${SERVER_ID}.log
            ${GET_DIST_ADDR_MAIN} >${log} 2>&1 &&
                local DIST_FLAGS=${DIST_FLAGS}" --role=ps" &&
                local DIST_FLAGS=${DIST_FLAGS}" --gs_addrs=${gs_addrs}" &&
                local DIST_FLAGS=${DIST_FLAGS}" --cs_addr=$(get_scheduler_addr ${log})" &&
                local DIST_FLAGS=${DIST_FLAGS}" --ps_addrs=$(get_server_addrs ${log})" &&
                local DIST_FLAGS=${DIST_FLAGS}" --ps_id=$(get_server_id ${log})" &&
                echo ${DIST_FLAGS} >>${log} &&
                ${DIST_TRAINER} ${DIST_FLAGS} >>${log} 2>&1 &&
                upload_file_to_hdfs ${log} ${LOG_DIR} &
        fi
    done
}

run_wk() {
    local gs_addrs=$(<${GS_ADDRS_FILE})
    for ((i = 0; i < ${DIST_WORKER_NUM}; ++i)); do
        export ROLE=worker
        export WORKER_ID=${i}
        node_id=$((${WORKER_ID} % ${#DIST_MACHINES[@]}))
        if test ${MY_DIST_ID} == ${node_id}; then
            echo Run wk ${WORKER_ID} at node ${MY_DIST_ID}
            local log=wk${WORKER_ID}.log
            ${GET_DIST_ADDR_MAIN} >${log} 2>&1 &&
                local DIST_FLAGS=${DIST_FLAGS}" --role=wk" &&
                local DIST_FLAGS=${DIST_FLAGS}" --gs_addrs=${gs_addrs}" &&
                local DIST_FLAGS=${DIST_FLAGS}" --cs_addr=$(get_scheduler_addr ${log})" &&
                local DIST_FLAGS=${DIST_FLAGS}" --ps_addrs=$(get_server_addrs ${log})" &&
                echo ${DIST_FLAGS} >>${log} &&
                ${DIST_TRAINER} ${DIST_FLAGS} >>${log} 2>&1 &&
                upload_file_to_hdfs ${log} ${LOG_DIR} &
        fi
    done
}

################################################################

export SERVER_NUM=${GS_SHARD_NUM}
export WORKER_NUM=0
export SCHEDULER_IP=${GS_MACHINES[0]}
export SCHEDULER_PORT=${FREE_PORT}
run_gs_scheduler
run_gs

check_gs
export SERVER_NUM=${DIST_SERVER_NUM}
export WORKER_NUM=${DIST_WORKER_NUM}
export SCHEDULER_IP=${DIST_MACHINES[0]}
export SCHEDULER_PORT=${FREE_PORT}
run_cs
run_ps
run_wk
