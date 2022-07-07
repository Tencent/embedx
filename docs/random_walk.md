# 随机游走

[TOC]

部分图模型使用 **随机游走** 生成模型的训练数据，我们提供工具生成 **随机游走数据**。

我们实现了 `uniform`、`frequency(alias、partial_sum)` 等多种随机游走，支持单机和分布式的实现。

接下来将依次介绍 `使用随机游走生成训练数据的模型`、`随机游走工具的参数介绍` 和 `随机游走`。

## 使用随机游走生成训练数据的模型

以下模型需要使用 `随机游走` 生成训练数据，有些需要生成 `序列格式数据`，有些需要生成 `边格式数据`。

| 模型                      | 随机游走之序列格式 |  随机游走之边格式  |
| ------------------------- | ------------------ | ------------------ |
| deepwalk                  | :heavy_check_mark: |                    |
| node2vec                  | :heavy_check_mark: |                    |
| struc2vec                 | :heavy_check_mark: |                    |
| metapath2vec              | :heavy_check_mark: |                    |
| eges                      | :heavy_check_mark: |                    |
| unsup_graphsage           |                    | :heavy_check_mark: |
| pinsage                   |                    | :heavy_check_mark: |
| unsup_bipartite_graphsage |                    | :heavy_check_mark: |

## 随机游走工具的参数介绍

| 参数名称      | 含义                                   | 注                                                                  |
| ------------- | -------------------------------------  | ------------------------------------------------------------------- |
| node_graph    | `string`, 节点关系数据目录             | 参考[节点关系数据](data_format.md#节点关系数据格式)                 |
| node_config   | `string`, 异构图节点信息配置文件       | 参考[节点如何编码](encode.md#节点如何编码)                          |
| gs_thread_num | `int`, 加载节点关系数据的线程数量      | 需要满足，`gs_thread_num <= node_graph 文件数量`                    |
| dist          | `int`, 单机或者分布式随机游走          | `1, 分布式`、`0, 单机`                                              |
| gs_addrs      | `string`, graph server 的 ip port 地址 | 分布式运行，worker 通过 gs_addrs 连接 graph server 进行随机游走     |
| gs_worker_num | `int`, worker 数量                     | 分布式运行，使用的 worker 数量，越多越快                            |
| gs_worker_id  | `int`, worker id                       | 分布式运行时，每个 worker 对应的 index，从 0 开始连续编码           |
| walker_type   | `int`, 随机游走类型                    | 0(uniform)、 1(alias)、2(word2vec)、 3(partial_sum)                 |
| walk_length   | `int`, 随机游走的步长                  | 示例：walk_length=10                                                |
| epoch         | `int`, 跑多少轮随机游走                | 示例：epoch=5                                                       |
| dump_type     | `int`, 输出结果格式                    | 0, 序列格式；1, 边格式                                              |
| meta_path_config | `string`, 异构图节点元路径配置      | 可以是文件名，也可以是包含元路径的字符串                            |
| out           | `string`, 存储邻居特征平均结果的目录   | 示例：out="output_random_walk"                                      |

- 注意，程序中使用的线程数取决于 `node_graph` 对应的文件数和 `gs_thread_num` 中的 ***最小值***

> - 不要 ***只使用一个文件*** 存储 `node_graph` 数据，否则无论 gs_thread_num 设置多大，系统中始终都只有一个 cpu 运行
>
> - gs_thread_num 设置为 `10`，效率就很高了，当然在满足 `node_graph 文件数量 < gs_thread_num` 时，越多越快
>
> - 预处理时将 `node_graph` 数据 ***划分成多个文件，越多越好***, 一般可设置文件数为 ***100~500*** 个
>

- 注意，`meta_path_config` 可以是文件名，也可以是包含元路径的字符串

  如果是文件名，对应的文件内容是

  ```
  节点类型1 节点类型2 节点类型3
  节点类型4 节点类型5
  ...
  ```

  举例

  ```
  1 2 3
  1 4
  ```

  如果不是文件名，其内容是

  ```
  节点类型1:节点类型2:节点类型3,节点类型4:节点类型5,...
  ```

  举例

  ```
  1:2:3,1:4
  ```

## 运行随机游走

### 编译 embedx

- 参考[编译](compile.md)文档编译

### 单机运行

- 进入到 `embedx/demo/single` 目录
- 参考[单机使用必读](intro_to_using_single.md)文档，运行 `run_random_walk.sh` 脚本，生成数据 `sequence`

### 分布式运行

- 进入到 `embedx/demo/dist` 目录
- 参考[分布式使用必读](intro_to_using_dist.md)文档，运行 `run_random_walk.sh` 脚本，生成数据 `sequence`
