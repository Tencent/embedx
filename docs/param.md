# 参数介绍

[TOC]

参数分为 `模型参数` 和 `数据参数`。首先介绍模型参数，它对于所有的模型都是适用的。
其次介绍数据参数，仅图模型和深度召回模型需要详细介绍。

## 模型参数

模型参数，介绍了模型训练和预测时候使用到的模型、数据、epoch 和 batch 等参数，用户作为参数手册查询。

模型参数中的`instance_reader_config`、`model_config` 和 `optimizer_config` 涉及到的参数比较多，在下面详细介绍。

- 参数介绍

  | 参数名称               | 含义                                         | 示例
  | ---------------------- | -------------------------------------------- | ------------------------------------------------------------- |
  | instance_reader        | `string`, 生成 `某种模型` 样本               | 无监督 graphsage 模型，`instance_reader=unsup_graphsage`      |
  | instance_reader_config | `string`, 模型样本的参数配置                 | 参考[instance_reader_config](param.md#instance_reader_config) |
  | model                  | `string`, 选择某种模型训练，                 | 无监督 graphsage 模型，`model=unsup_graphsage`                |
  | model_config           | `string`, 模型网络结构的配置参数             | 参考[model_config](param.md#model_config)                     |
  | optimizer              | `string`, 优化方法                           | 示例：optimizer="adam"                                        |
  | optimizer_config       | `string`, 优化方法参数配置                   | 参考[optimizer_config](param.md#optimizer_config)             |
  | in                     | `int`, 训练或者预测时的文件或目录            | 见表格下面 `注意`                                             |
  | epoch                  | `int`, 训练时的运行轮数                      | 示例：epoch=10                                                |
  | batch                  | `int`, 训练或预测时的 batch 大小             | 示例：batch=128                                               |
  | thread_num             | `int`, 单机训练或预测时使用的线程数          | 示例：thread=10                                               |
  | model_shard            | `int`, 训练或预测时使用的 shard 数量         | `model_shard=thread_num`                                      |
  | target_type            | `int`, 训练或者预测时候的目标                | 训练，`0 表示 loss`; 预测，`1 输出 prob`、`2 输出 embedding`  |
  | in_model               | `string`, 输入模型的目录                     | 示例：in_model="model"                                        |
  | out_predict            | `string`, 模型预测时，结果输出的目录         | 示例：out_predict="out_predict"                               |
  | num_ps_thread          | `int`, 分布式训练或者预测时，ps 使用的线程数 | 示例：num_ps_thread=10                                        |
  | out_model              | `string`, 输出模型的目录                     | 示例：out_model="model"                                       |

- 补充 1：不用模型的参数 `--in` 对应的训练数据是不同，参考[数据格式](data_format.md)文档，搜索关键字 `--in` 查看。
- 补充 2：程序中使用的线程数取决于 `--in` 对应的文件数和 `thread_num` 中的 ***最小值***

> - 不要 ***只使用一个文件*** 存储 `--in` 数据，否则无论 thread_num 设置多大，系统中始终都只有一个 cpu 运行
>
> - thread_num 设置为 `10`，效率就很高了，当然在满足 `--in 文件数量 < thread_num` 时，越多越快
>
> - 预处理时将 `--in` 数据 ***划分成多个文件，越多越好***, 一般可设置文件数为 ***100~500*** 个

---

### instance_reader_config

- 参数介绍

  | 参数名称      | 含义                                | 注                                       |
  | ------------- | ----------------------------------- | ---------------------------------------- |
  | num_neg       | `int`, 负采样数量                   | 常用值 5, 10                             |
  | window_size   | `int`, 上下文窗口大小               | 常用值 5                                 |
  | depth         | `int`, 图卷积的层数                 | 常用值 1，2                              |
  | num_neighbors | `int`, 每层采样的邻居数             | 两层图卷积可设置为 num_neighbors="10,10" |
  | is_train      | `int`, 用来区分训练和测试           | 1, 生成训练数据；0, 生成预测数据         |
  | multi_label   | `int`, 区分多标签还是多分类         | 1, 多标签分类；0, 多分类                 |
  | num_label     | `int`, 多标签分类任务中标签总数     | 如多标签为`0 0 0 1 0`，num_label=5       |
  | max_label     | `int`, 多分类任务中表示最大的 label | 如多分类的标签为`0 1 2 3` 则 max_label=3 |

- 示例

  | 模型            | 训练                                                           | 预测                                               |
  | --------------- | -------------------------------------------------------------- | -------------------------------------------------- |
  | deepwalk        | instance\_reader\_config="num_neg=5;window_size=5;is_train=1"  | instance\_reader\_config="is_train=0"              |
  | node2vec        | instance\_reader\_config="num_neg=5;window_size=5;is_train=1"  | instance\_reader\_config="is_train=0"              |
  | struc2vec       | instance\_reader\_config="num_neg=5;window_size=5;is_train=1"  | instance\_reader\_config="is_train=0"              |
  | metapath2vec    | instance\_reader\_config="num_neg=5;window_size=5;is_train=1"  | instance\_reader\_config="is_train=0"              |
  | eges            | instance\_reader\_config="num_neg=5;window_size=5;is_train=1"  | instance\_reader\_config="is_train=0"              |
  | unsup_graphsage | instance\_reader\_config="num_neg=10;depth=2;num_neighbors=10" | instance\_reader\_config=depth=2;num_neighbors=10  |
  | sup_graphsage   | instance\_reader\_config="num_neighbors=10;max_label=6;multi_label=0" | instance\_reader\_config="num_neighbors=10" |

---

### model_config

- 参数介绍

  | 参数名称         | 含义                                   | 注                                                             |
  | ---------------- | -------------------------------------- | -------------------------------------------------------------- |
  | config           | 特征组配置                             | 举例 `config=1:10000:128,12:30000:64`                          |
  | sparse           | 更新方式                               | 0, 有冲突的更新，速度快；1, 无冲突更新，速度慢                 |
  | depth(int)       | 图卷积的层数                           | 仅图卷积模型需要                                               |
  | dim(int)         | 图卷积模型输出                         | 仅图卷积模型需要                                               |
  | alpha            | relu 激活函数参数                      | 举例：alpha=0.1                                                |
  | multi_label(int) | 1, 多标签分类；0, 多分类               | 仅节点分类模型需要                                             |
  | num_label(int)   | 多标签分类任务中一个节点拥有标签的个数 | 仅节点分类模型需要                                             |
  | max_label(int)   | 多分类任务中表示最大的 label           | 仅节点分类模型需要，如多分类的标签为 `0 1 2 3`, 则 max_label=3 |

`config` 通常与 `sparse` 放在一起使用，首先介绍下 config 参数，再介绍 sparse 参数。

- config

  - config 是特征组配置[特征如何编码](encode.md#特征如何编码)，
    格式是 `group_id1:row1:col1,group_id2:row2:col2,...`，中间使用 `,` 隔开

  - group_id 指的是 `特征组 id`，row 指的是 `特征行`，col 指的是 `特征 embedding 长度`组成

  - 举例有 config= `1:10000:128,12:30000:64`

    - 第一个特征组，`特征组 id=1、特征行=10000、特征 embedding 长度=128`
    - 第二个特征组，`特征组 id=12、特征行=30000、特征 embedding 长度=64`

- sparse

  - `sparse=0`

    - embedding 矩阵是 TSR, 其形状是（embedding 矩阵行，embedding 矩阵列）
    - 不同特征 id 可能对应相同 embedding, 即冲突。为了减少冲突，通常"embedding 矩阵行"和"特征组的子特征空间"成正比

  - `sparse=1`

    - 如果 s 是 1, embedding 矩阵是 SRM, 其形状是（0, embedding 矩阵列）, "embedding 矩阵行"被忽略
    - 所有特征 id 独享自己的 embedding

- 示例

  | 模型            | 配置                                                                                                        |
  | --------------- | ----------------------------------------------------------------------------------------------------------- |
  | deepwalk        | model\_config="0:10000:128;sparse=1"                                                                        |
  | node2vec        | model\_config="0:10000:128;sparse=1"                                                                        |
  | struc2vec       | model\_config="0:10000:128;sparse=1"                                                                        |
  | metapath2vec    | model\_config="0:10000:128;sparse=1"                                                                        |
  | eges            | model\_config="config=1:10000:128,2:10000:128;sparse=1"                                                     |
  | unsup_graphsage | model\_config="config=1:10000:128,2:10000:128;depth=2;dim=128;alpha=0.1;sparse=1"                           |
  | sup_graphsage   | model\_config="config=1:10000:128,2:10000:128;depth=2;dim=128;alpha=0.1;sparse=1;max_label=6;multi_label=0" |

---

### optimizer_config

我们提供了多种优化算法，其中 `adagrad` 和 `adam` 使用较多，以下给出示例，更多优化算法待补充...

- 示例

  | 优化方法 | 例子                                                        |
  | -------- | ----------------------------------------------------------- |
  | adam     | optimizer_config="rho1=0.9;rho2=0.999;alpha=1e-4;beta=1e-8" |
  | adagrad  | optimizer_config="alpha=0.1;beta=1e-6"                      |

---

## 图模型数据参数

图模型训练或者预测时要先加载图数据，再进行模型训练或者预测。

- 参数介绍

  | 参数名称              | 含义                         | 注                                                          |
  | --------------------- | ---------------------------- | ----------------------------------------------------------- |
  | node_graph            | `string`，节点关系数据的目录 | 参考[节点关系数据](data_format.md#节点关系数据格式)         |
  | node_feature          | `string`, 节点特征的目录     | 参考[节点特征数据](data_format.md#节点特征数据格式)         |
  | neighbor_feature      | `string`, 邻居节点特征的目录 | 参考[邻居节点特征数据](data_format.md#邻居节点特征数据格式) |
  | node_config           | `string`, 节点类型配置文件   | 参考[编码](encode.md#节点如何编码)                          |
  | negative_sampler_type | `int`, 采样节点的方法        | 0(uniform)、1 (alias)、2 (word2vec)、 3 (partial_sum)       |
  | neighbor_sampler_type | `int`, 采样邻居的方法        | 0(uniform)、1 (alias)、2 (word2vec)、 3 (partial_sum)       |
  | gs_thread_num         | `int`, 加载数据的线程数量    | 越多越快，最大不要超过文件数量                              |
  | gs_addrs              | `string`, ip port 地址       | 分布式运行，worker 通过 `gs_addrs` 连接 graph server        |
  | gs_shard_num          | `int`, graph server 的数量   | 分布式参数，单机不需要提供                                  |
  | gs_shard_id           | `int`, graph server 在 gs_addrs 中的 index | 分布式参数，取值从 0 开始递增到 n             |

- 补充 1：如果数据存储在 hdfs, embedx 依赖 **libhdfs** 读写 hdfs

> - 可以从 hadoop 安装包中获取 libhdfs.so, 并和程序放到同一目录下

- 补充 2：程序中使用的线程数取决于 `--node_graph` 对应的文件数和 `gs_thread_num` 中的 ***最小值***

> - 不要 ***只使用一个文件*** 存储 `node_graph` 数据，否则无论 gs_thread_num 设置多大，系统中始终都只有一个 cpu 运行
>
> - gs_thread_num 设置为 `10`，效率就很高了，当然在满足 `node_graph 文件数量 < gs_thread_num` 时，越多越快
>
> - 预处理时将 `node_graph` 数据 ***划分成多个文件，越多越好***, 一般可设置文件数为 ***100~500*** 个

---

## 深度召回模型数据参数

深度召回模型训练需要先加载 `item 频次文件` 供负采样使用，再进行模型训练，以下介绍深度召回数据参数。

- 参数介绍

  | 参数名称              | 含义                                        | 注                                                      |
  | --------------------- | ------------------------------------------- | ------------------------------------------------------- |
  | freq_file             | `string`, item 频次文件或目录，供负采样使用 | [物品频次数据格式](data_format.md#物品频次数据格式)     |
  | node_config           | `string`, freq_file 中的 item 类型配置文件  | [编码](encode.md#节点如何编码)                          |
  | negative_sampler_type | `int`, 采样节点的方法                       | 0(uniform)、1 (alias)、2 (word2vec)、 3 (partial_sum)   |
  | item_feature          | `string`, item 特征文件或目录               | 参考[物品特征数据格式](data_format.md#物品特征数据格式) |
