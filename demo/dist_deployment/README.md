# 分布式任务部署

[TOC]

本文档介绍如何在多台机器上部署分布式任务.

本文档给出典型模版, 用户请按需修改.

## 配置["run\_dist.sh"](./run_dist.sh)

### 配置机器IP

`GS_MACHINES`: 运行 graph server 的机器 IP.

`DIST_MACHINES`: 分布式训练或预测任务的机器 IP.

例子.

```shell
readonly GS_MACHINES=(
    [0]=9.141.195.101
    [1]=9.141.198.7
)

readonly DIST_MACHINES=(
    [0]=9.141.200.143
    [1]=9.141.201.105
)
```

### 上传下面几个文件到所有机器可以访问的远端目录

- graph\_server\_main, 编译得到.
- close\_server\_main, 编译得到.
- dist\_trainer, 编译得到.
- get\_dist\_addr\_main, 编译得到.
- libhdfs.so, 数据存放 hdfs 时需上传, 可从 hadoop 安装包中获取.

### 配置模型flags

例子.

```shell
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
```

### 配置机器与远端目录的通信环境

## 任务部署

运行["distribute.sh"](./distribute.sh)来分发和运行 "run\_dist.sh".

注意: 若机器间无法 `ssh` 通信, 则需手动分发和运行 "run\_dist.sh".
