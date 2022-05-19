# 邻居特征平均

[TOC]

部分图模型使用 **邻居特征** 加速模型训练，我们提供工具生成 **邻居特征数据**。

`邻居特征平均` 指的是对于 **node_graph** 中的一条样本 `node adj_node1:weight1 adj_node2:weight2 ...`。
使用 `adj_node1`、 `adj_node2` 等邻接节点的特征求平均，来表示节点 `node` 的邻居特征。

接下来将依次介绍 `使用邻居特征平均功能的模型`、`邻居特征平均工具的参数介绍` 和 `运行邻居特征平均`。

## 使用邻居特征平均功能的模型

以下图模型需要使用 `邻居特征平均` 功能生成数据，任务的输出对应[图模型数据参数介绍](param.md#图模型数据参数介绍)中的参数 `--neighbor_feature`。

| 模型                      |
| ------------------------- |
| unsup_graphsage           |
| sup_graphsage             |
| unsup_bipartite_graphsage |

## 邻居特征平均工具的参数介绍

| 名称          | 含义                                   | 注                                                                  |
| ------------- | -------------------------------------- | ------------------------------------------------------------------- |
| node_graph    | `string`, 节点关系数据目录             | 参考[节点关系数据格式](data_format.md#节点关系数据格式)             |
| node_feature  | `string`, 节点特征的目录               | 参考[节点特征数据格式](data_format.md#节点特征数据格式)             |
| sample_num    | `int`,  采样邻居进行特征平均           | 默认使用全量邻居，如果采样 10 个邻居，可设置 sample_num =10         |
| dist          | `int`, 单机或者分布式随机游走          | `1（分布式）`、 `0（单机）`                                         |
| gs_thread_num | `int`, 加载节点关系数据的线程数量      | 需要满足，`gs_thread_num <= node_graph 文件数量`                    |
| gs_addrs      | `string`, graph server 的 ip port 地址 | 分布式运行，worker 通过 gs_addrs 连接 graph server 进行邻居特征平均 |
| gs_worker_num | `int`, worker 数量                     | 分布式运行，使用的 worker 数量，越多越快                            |
| gs_worker_id  | `int`, worker id                       | 分布式运行时，每个 worker 对应的 index，从 0 开始连续编码           |
| out           | `string`, 存储邻居特征平均结果的目录   | 示例：out="output_neighbor_feature"                                 |

- 注意，程序中使用的线程数取决于 `node_graph` 对应的文件数和 `gs_thread_num` 中的 ***最小值***

> - 不要 ***只使用一个文件*** 存储 `node_graph` 数据，否则无论 gs_thread_num 设置多大，系统中始终都只有一个 cpu 运行
>
> - gs_thread_num 设置为 `10`，效率就很高了，当然在满足 `node_graph 文件数量 < gs_thread_num` 时，越多越快
>
> - 预处理时将 `node_graph` 数据 ***划分成多个文件，越多越好***, 一般可设置文件数为 ***100~500*** 个

## 运行邻居特征平均

### 编译 embedx

- 参考[编译](compile.md)文档编译

### 单机运行

- 进入到 `embedx/demo/single` 目录
- 参考[单机使用必读](intro_to_using_single.md)文档，运行 `run_average_feature.sh` 脚本，生成数据 `average_feature`

### 分布式运行

- 进入到 `embedx/demo/dist` 目录
- 参考[分布式使用必读](intro_to_using_dist.md)文档，运行 `run_average_feature.sh` 脚本，生成数据 `average_feature`
