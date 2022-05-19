# 数据格式

[TOC]

embedx 支持图模型、深度排序和深度召回模型，不同类型的模型 在 `训练` 或者 `预测` 时，需要准备不同的数据。

以下依次介绍图模型、深度排序和深度召回模型在 ***训练阶段使用的数据***、 ***预测阶段使用的数据*** 和 两个阶段使用到的数据的 ***数据格式***。

## 训练阶段使用的数据

### 图模型

图模型训练时需要准备 `图数据` 和 `训练数据`。

- 图数据

  - 图数据包括 `节点关系数据`、`节点特征数据` 和 `邻居节点特征数据`

    - [节点关系数据格式](data_format.md#节点关系数据格式)
    - [节点特征数据格式](data_format.md#节点特征数据格式)
    - [邻居节点特征数据格式](data_format.md#邻居节点特征数据格式)

  - 不同的图模型使用不同的图数据

    - 有些图模型如 `deepwalk` 使用部分图数据如 `节点关系数据`
    - 有些图模型如 `graphsage` 使用全部图数据包括 `节点关系数据`、`节点特征数据` 和 `邻居节点特征数据`

  - 图数据使用总结

    | 模型                    | 节点关系数据       | 节点特征数据       |  邻居节点特征数据  |
    | ----------------------- | ------------------ | ------------------ | ------------------ |
    | deepwalk                | :heavy_check_mark: |                    |                    |
    | node2vec                | :heavy_check_mark: |                    |                    |
    | struc2vec               | :heavy_check_mark: |                    |                    |
    | metapath2vec            | :heavy_check_mark: |                    |                    |
    | eges                    | :heavy_check_mark: | :heavy_check_mark: |                    |
    | unsup_graphsage         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
    | pinsage                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
    | sup_graphsage（多标签） | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
    | sup_graphsage（多分类） | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

- 训练数据

  - 训练数据包括 `随机游走生成的序列数据`、`随机游走生成的边数据`、`多分类数据` 和 `多标签数据`

    - [随机游走生成的序列数据](data_format.md#随机游走生成的序列数据格式)
    - [随机游走生成的边数据](data_format.md#随机游走生成的边数据格式)
    - [多分类数据](data_format.md#多分类数据格式)
    - [多标签数据](data_format.md#多标签数据格式)

  - 不同的图模型使用不同的训练数据

    - 有些图模型如 `deepwalk` 使用 `随机游走生成的序列数据`
    - 有些图模型如 `graphsage` 使用 `随机游走生成的边数据`

  - 训练数据使用总结

    | 模型                     | 随机游走之序列数据 |  随机游走之边数据  | 多分类数据         | 多标签数据         |
    | ------------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
    | deepwalk                 | :heavy_check_mark: |                    |                    |                    |
    | node2vec                 | :heavy_check_mark: |                    |                    |                    |
    | struc2vec                | :heavy_check_mark: |                    |                    |                    |
    | metapath2vec             | :heavy_check_mark: |                    |                    |                    |
    | eges                     | :heavy_check_mark: |                    |                    |                    |
    | unsup_graphsage          |                    | :heavy_check_mark: |                    |                    |
    | pinsage                  |                    | :heavy_check_mark: |                    |                    |
    | sup_graphsage（多分类）  |                    |                    | :heavy_check_mark: |                    |
    | sup_graphsage（多标签）  |                    |                    |                    | :heavy_check_mark: |

### 深度排序模型

深度排序模型训练时需要准备 `训练数据`。

- 训练数据

  - 训练数据包括 `排序样本数据`
    - [排序样本数据格式](data_format.md#排序样本数据格式)

### 深度召回模型

深度召回模型训练时需要准备 `物品频次数据` 和 `训练数据`。

- 物品频次数据

  - 物品频次数据包括 `物品频次数据` 和 `物品编码类型配置`
    - [物品频次数据格式](data_format.md#物品频次数据格式)
    - [物品编码类型配置格式](data_format.md#物品编码类型配置)

  - 物品频次数据使用总结

    | 模型        |  物品频次数据      | 物品编码类型配置   |
    | ----------- | ------------------ | ------------------ |
    | youtube_dnn | :heavy_check_mark: | :heavy_check_mark: |
    | dssm        | :heavy_check_mark: | :heavy_check_mark: |

- 训练数据

  - 训练数据包括 `用户特征数据` 和 `物品特征数据`

    - [用户特征数据格式](data_format.md#用户特征数据格式)
    - [物品特征数据格式](data_format.md#物品特征数据格式)

  - 不同的深度召回模型使用不同的训练数据

    - 有些深度召回模型如 `youtube_dnn` 只使用 `用户特征数据`
    - 有些深度召回模型如 `dssm` 同时使用 `用户特征数据` 和 `物品特征数据`

  - 训练数据使用总结

    | 模型        |  用户特征数据      |  物品特征数据      |
    | ----------- | ------------------ | ------------------ |
    | youtube_dnn | :heavy_check_mark: |                    |
    | dssm        | :heavy_check_mark: | :heavy_check_mark: |

---

## 预测阶段使用的数据

### 图模型

- 图模型预测时使用 `图数据` 和 `图预测数据`

  - 图数据

    - 图模型在 `预测阶段` 与 `训练阶段` 使用的 ***图数据完全相同***

  - 图预测数据

    - 图模型在 `预测阶段` 使用到的 `图预测数据` 格式是相同的。[图预测数据格式](data_format.md#图预测数据格式)

### 深度排序模型

- 深度排序模型预测时使用 `排序样本数据`。

  - `预测阶段` 与 `训练阶段` 使用的 ***排序样本数据完全相同***

### 深度召回模型

- 深度召回模型预测时使用 `用户特征数据` 和 `物品特征数据`

  - 用户特征数据

    - 模型预测 `user embedding` 使用。[用户特征数据格式](data_format.md#用户特征数据格式)

  - 物品特征数据

    - 模型预测 `item embedding` 使用。[物品特征数据格式](data_format.md#物品特征数据格式)

---

## 数据格式介绍

总结以上`图模型`、`深度排序模型` 与 `深度召回模型` 在 `训练` 与 `预测` 阶段使用到的数据，如下表格，接下来依次介绍。

- 数据总结

| 数据               | 功能
| ------------------ | ------------------------------------------------ |
| 节点关系数据       | `图模型`，提供图查询功能（采样节点，采样邻居等） |
| 节点特征数据       | `图模型`，部分图模型使用节点特征                 |
| 邻居节点特征数据   | `图模型`，部分图模型使用的邻居节点特征           |
| 随机游走之序列数据 | `图模型`，部分无监督图模型的训练数据             |
| 随机游走之边数据   | `图模型`，部分无监督图模型的训练数据             |
| 多分类数据         | `图模型`，有监督图模型的训练数据                 |
| 多标签数据         | `图模型`，有监督图模型的训练数据                 |
| 图预测数据         | `图模型`，预测使用的数据                         |
| 排序样本数据       | `深度排序模型`，训练或者预测使用                 |
| 物品频次数据       | `深度召回模型`，负采样的物品集合                 |
| 物品编码类型配置   | `深度召回模型`，指定物品编码类型                 |
| 用户特征数据       | `深度召回模型`，训练或者预测使用                 |
| 物品特征数据       | `深度召回模型`，训练或者预测使用                 |

### libsvm 格式

`节点关系数据`、`节点特征数据`、`邻居节点特征数据`、`用户特征数据` 和 `物品特征特征数据` 都是 `类 libsvm 格式` 数据。

首先介绍`libsvm 格式`，参考[libsvm](https://github.com/Tencent/deepx_core/blob/master/doc/instance.md)，以空格作为分隔符，格式是：

```shell
label id1:value1 id2:value2 ...
```

### 节点关系数据格式

`部分图模型` 训练或预测时参数 `--node_graph` 指的是就是节点关系数据，参考[节点关系数据](../demo/data/cora/partition_context)。

- 格式

```shell
node adj_node1:vaule1 adj_node2:value2 ...
```

- 概述

  - `类 libsvm 格式`，区别是 `node` 取代了 `label`、`adj_node1` 取代了 `id1`，`adj_node2` 取代了 `id2`
  - 其中 `node adj_node1:value1`, 指的是图中的边(node adj_node1), `value1` 是边的权重
  - 其中 `node adj_node2:value2`, 指的是图中的边(node adj_node2), `value2` 是边的权重
  - `node`、`adj_node1` 和 `adj_node2` 等是`uint64 类型`, `value1`、`value2` 是 `浮点类型`

- 示例

```shell
41 224:1.0 302:1.0 112:1.0 542:1.0
1 202:1.0
1000 50:0.3 16:0.2 27:0.5
```

---

### 节点特征数据格式

`部分图模型` 训练或预测时参数 `--node_feature` 指的就是节点特征数据，参考[节点特征数据](../demo/data/cora/partition_node_feature)。

- 格式

```shell
node id1:value1 id2:value2 ...
```

- 介绍

  - `类 libsvm 格式`，区别是 `node` 取代了 `label`
  - 其中 `id1`、`id2` 等是 `node` 的 `特征 id`, `value1` 和 `value2` 是 `特征 id` 对应的特征权重
  - `node`、`id1` 和 `id2` 等是 `uint64 类型`, `value1` 和 `value2` 是 `浮点类型`
  - 特征 id 的编码与生成方式参考[特征如何编码](encode.md#特征如何编码)

- 示例

```shell
41 123:1.5 456:1.0 567:0.1
1 1:1.0 40000:1.5
1000 3333:1.0 4444:1.2
```

---

### 邻居节点特征数据格式

`部分图模型` 训练或者预测时参数 `--neighbor_feature` 指的是邻居特征数据，参考[邻居节点特征数据](../demo/data/cora/partition_neigh_feature)。
它用于图卷积类模型的加速，参考[邻居特征平均](average_feature.md)。

- 格式

```shell
node id1:value1 id2:value2 ...
```

- 概述

  - `类 libsvm 格式`，区别是 `node` 取代了 `label`
  - 其中 `id1`、`id2` 等是 `node` 的 `特征 id`, `value1` 和 `value2` 是 `特征 id` 对应的特征权重
  - `node`、`id1` 和 `id2` 等是 `uint64 类型`, `value1` 和 `value2` 是 `浮点类型`
  - 特征 id 的编码与生成方式参考[特征如何编码](encode.md#特征如何编码)

- 示例

```shell
41 123:1.5 456:1.0 567:0.1
1 1:1.0 40000:1.5
1000 3333:1.0 4444:1.2
```

---

### 随机游走之序列数据格式

`部分图模型` 训练时参数 `--in` 指的是随机游走之序列数据，参考[随机游走之序列数据](../demo/data/cora/sequence)。
**embedx** 提供了随机游走工具，辅助用户生成序列数据，参考[随机游走](random_walk.md)。

- 格式

```shell
node1 node2 node3
```

- 概述

  - 行间以空格分隔，以 `node1` 为源节点随机游走得到的序列
  - `node1`、`node2` 和 `node3` 等是 `uint64 类型`

- 示例

```shell
1 100 234 567
57 89 100 123
90 100 190 290
```

---

### 随机游走之边数据格式

`部分图模型` 训练时参数 `--in` 指的是随机游走之边数据，参考[随机游走之边数据](../demo/data/cora/walk)。
**embedx**提供了随机游走工具，可以辅助用户生成游走边数据，参考[随机游走](random_walk.md)。

- 格式

```shell
node1 node2
```

- 概述

  - 行间以空格分隔，`node1` 指的是 `源顶点`、`node2` 指的是 `目标顶点`
  - `node1` 和 `node2` 均是 `uint64 类型`

- 示例

```shell
1 100
1 57
23 56
```

---

### 多分类数据格式

`部分图模型` 训练时参数 `--in` 指的是多分类数据，参考[多分类数据](../demo/data/cora/cora_labels_multi_classification.train)。

- 格式

```shell
node label
```

- 概述

  - 行间以空格分隔，其中 `node` 指的是 `节点`、`label` 指的是 `此节点的类别`
  - `node` 是 `uint64 类型`，`label` 是 `int 类型`

- 示例

```shell
1 0
234 2
23 1
```

---

### 多标签数据格式

`部分图模型` 训练时参数 `--in` 指的是多标签数据，参考[多标签数据](../demo/data/cora/cora_labels_multi_label_classification.train)。

- 格式

```shell
node label1 label2 label3
```

- 概述

  - 行间以空格分隔，其中 `node` 指的是 `节点`、`label1`、`label2` 和 `label3` 指的此节点的标签
  - `node` 是 `uint64 类型`，`label1`、`label2` 和 `label3` 是 `int 类型`

- 示例

```shell
1 0 1 0
234 1 0 1
90 0 1 1
```

---

### 图预测数据格式

`图模型` 预测时参数 `--in` 指的是图预测数据，参考[图预测数据](../demo/data/cora/partition_context)。

- 格式

```shell
node
```

- 概述

  - 每行一个节点，`node` 是 `uint64 类型`，指的是用户待预测的节点

  - 用户在 demo 中会发现我们使用了 `节点关系数据` 作为 `图预测数据`。
    这是因为在预测时，embedx 代码只会解析 `节点关系数据` 的第一个节点使用

  - 用户可以选择复用 `节点关系数据` 作为图预测数据，也可以按照 `图预测数据格式` 准备数据

- 示例

```shell
1
234
90
```

### 排序样本数据格式

`深度排序模型` 训练和预测时, 输入样本数据支持 `libsvm`、`libsvm_ex` 和 `uch` 多种格式, 参考[样本格式](https://github.com/Tencent/deepx_core/blob/master/doc/instance.md).

### 用户特征数据格式

`深度召回模型` 训练或者预测时参数 `--in` 指的是就是用户特征数据，参考[用户特征数据](../demo/data/dssm/training_data)

- 格式

```shell
node id1:value1 id2:value2 ...
```

- 概述

  - `类 libsvm 格式`，区别是 `node` 取代了 `label`
  - `id1` 和 `id2`是描述当前 `node` 的 `特征 id`, `value1` 和 `value2` 是 `特征 id` 对应的特征权重
  - `item_node`、`id1` 和 `id2` 是 `uint64 类型`, `value1` 和 `value2` 是 `浮点类型`
  - 特征 id 的编码与生成方式参考[特征如何编码](encode.md#特征如何编码)

- 示例

```shell
41 224:1.0 302:1.0 112:1.0 542:1.0
1 202:1.0
1000 50:0.3 16:0.2 27:0.5
```

---

### 物品特征数据格式

`深度召回模型` 训练或者预测时候参数 `--node_feauture` 指的是就是物品特征数据，参考[物品特征数据](../demo/data/dssm/item_feature)

- 格式

```shell
node id1:value1 id2:value2 ...
```

- 概述

  - `类 libsvm 格式`，区别是 `node` 取代了 `label`、它指的是用户输入的 `物品频次数据` 中的节点
  - `id1` 和 `id2` 是描述当前 `node` 的 `特征 id`, `value1` 和 `value2` 是 `特征 id` 对应的特征权重
  - `node`、`id1` 和 `id2` 是 `uint64 类型`, `value1` 和 `value2` 是 `浮点类型`
  - 特征 id 的编码与生成方式参考[特征如何编码](encode.md#特征如何编码)

- 示例

```shell
41 224:1.0 302:1.0 112:1.0 542:1.0
1 202:1.0
1000 50:0.3 16:0.2 27:0.5
```

---

### 物品频次数据格式

`深度召回模型` 训练时参数 `--freq_file` 指的就是物品频次数据，参考[物品频次数据](../demo/data/dssm/freq_file)

- 格式

```shell
node frequncy
```

- 概述

  - 每行格式为 `node frequency`, 以空格分隔，包含 node 和对应频次信息
  - `node` 和 `frequncy` 都是 uint64 类型
  - 深度召回模型 使用物品频次数据进行负采样
  - 负例采样的时候可以选择 `均匀采样` 或 `带权采样`，带权采样时会根据 `item 的频次` 进行采样

- 示例

```shell
41 20
2 15
3 10
```

---

### 物品编码类型配置

`深度模型` 训练时参数 `--node_config` 指的是 freq_file 中物品编码类型，参考[物品编码类型文件](../demo/data/dssm/freq_file_ns_config)

- 格式

```shell
group_name group_id
```

- 概述

  - group_name 是 string 类型用来记录 id 的意义，给一个有意义的名字即可
  - group_id 是 int 类型，表明 item 编码类型
  - 生成方式参考[编码](encode.md#生成 node_config 文件)

- 示例

```shell
Item 9999
```
