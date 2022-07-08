# 模型性能与效果

[TOC]

## 简介

### embedx 支持的模型

- 已支持模型

  | 模型                      |  模型类型  |
  | ------------------------- | ---------- |
  | deepwalk                  | 图模型     |
  | eges                      | 图模型     |
  | unsup_graphsage           | 图模型     |
  | sup_graphsage             | 图模型     |
  | semisup_graphsage         | 图模型     |
  | unsup_bipartite_graphsage | 图模型     |
  | cmv                       | 图模型     |
  | deep_graph_infomax        | 图模型     |
  | deep_graph_contrastive    | 图模型     |
  | node_infograph            | 图模型     |
  | deepfm                    | 排序模型   |
  | deepfm2                   | 排序模型   |
  | din                       | 排序模型   |
  | youtubednn                | 召回模型   |
  | dssm                      | 召回模型   |
  | self-training dssm        | 召回模型   |
  | graphdeepfm               | 图排序模型 |
  | graphdssm                 | 图召回模型 |
  | ...                       |  ...       |

### 评测环境

- 机器配置

  - Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
  - 内存 256GB

- 软件版本

  - Tencent tlinux release 2.2
  - gcc (GCC) 4.8.5
  - Python 3.7.10
  - scikit-learn 1.0

- 参数设置与论文基本保持一致

  - 参考[demo/single](../demo/single)目录下对应模型脚本
  - 所有评测均使用 ***8 个线程***

### 评测流程

- 对于 **业界已有模型**

  - 有公开数据集，给出公开数据集上的评测效果和效率
  - 没有公开数据集，给出业务数据集上相对提升

- 对于 **自研模型**

  - 有公开数据集，给出公开数据集上的评测效果和效率
  - 没有公开数据集，阐述建模思路并给出业务数据集上相对提升

以下依次介绍 `图模型`、`深度模型` 和 `图与深度学习的联合建模模型` 效果与效率的评测。

## 图模型

### 评测数据集

- blogcatalog 数据集

  - 链接：<http://leitang.net/code/social-dimension/data/blogcatalog.mat>
  - 包含 10,312 个节点，667,966 条边，39 个标签

- ppi 数据集

  - 链接：<http://snap.stanford.edu/graphsage/ppi.zip>
  - 包含 56,944 个节点，1,612,348 条边，121 个标签

- cora 数据集

  - 链接：<https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz>
  - 包含 2,708 个节点，10,556 条边，7 个标签

### 评测结果

- 业界已有模型

  | 模型                      |  数据集     | 耗时（秒）| micro-F1 |
  | ------------------------- | ----------- | --------- | -------- |
  | deepwalk                  | blogcatalog |    175    |   0.408  |
  | eges                      |    ppi      |    154    |   0.431  |
  | unsup_graphsage           |    ppi      |    332    |   0.510  |
  | sup_graphsage             |    ppi      |    19     |   0.612  |
  | cmv                       |    cora     |    83     |   0.763  |
  | deep_graph_infomax        |    cora     |    34     |   0.721  |
  | deep_graph_contrastive    |    cora     |    115    |   0.757  |
  | node_infograph            |    cora     |    123    |   0.856  |

- 自研模型

  - unsup_bipartite_graphsage

    - 在 unsup_graphsage 基础上从同构图扩展到二部图
    - 节点根据类型使用不同编码器编码，增强异构表达能力

  - semisup_graphsage

    - 联合训练有监督与无监督目标
    - 有监督，利用少量的标签数据，学习业务的目标
    - 无监督，利用大量无标签数据，学习节点相似性

---

## 深度排序模型

### 评测数据集

- avazu 数据集

  - 链接：<https://www.kaggle.com/c/avazu-ctr-prediction/data>

- criteo 数据集

  - 链接：<https://www.kaggle.com/c/criteo-display-ad-challenge/data>

### 评测结果

- 业界已有模型

  | 模型   |  数据集   | 耗时（分钟） | auc   |
  | ------ | --------  | -----------  | ----- |
  | deepfm |   avazu   |    8         | 0.749 |
  | deepfm |   criteo  |    16        | 0.801 |

---

## 深度召回模型

### 评测数据集

- 没有公开数据集（若有，以后补充），使用业务数据集

### 评测结果

- 业界已有模型

  | 模型       |  数据集  |  耗时（分钟） | hitrate |
  | ---------- | -------- | ------------- | ------- |
  | youtubednn |   内部   |   60 分钟     |  0.13   |
  | dssm       |   内部   |   60 分钟     |  0.19   |

- 自研模型

  - self-training dssm

    - 通过构造补充正例，增强对长尾 item 的召回能力
    - 相比于 dssm，hitrate + 1.3%

---

## 图与深度模型的联合建模

### 背景

- 常用排序或者召回模型没法直接使用用户行为图数据，需要使用两阶段方案

  - 图模型中，首先训练 embedding，然后加入排序或者召回模型作为特征使用
  - 两阶段方案达不到预期的业务效果

我们通过联合建模的方案来解决这个问题，以 graph\_deepfm 和 graph\_dssm 为例讲述。

### graphdeepfm,  图与深度排序模型联合建模

- 图模型学习 user embedding 加入 deepfm 模型侧，联合训练
- 对比 deepfm 与 graphdeepfm 的效果

  | 模型          | auc       |
  | ------------- | --------- |
  | deepfm        |    -      |
  | graph_deepfm  |  + 0.346% |

### graphdssm, 图与深度召回模型联合建模

- user 侧引入 graph 信息，graph 与 dssm 模型
- 对比 dssm 与 graphdssm 的效果

  | 模型        | hitrate   |
  | ----------- | --------- |
  | dssm        |    -      |
  | graph_dssm  | + 2.63%   |
