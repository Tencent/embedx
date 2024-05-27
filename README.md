![logo](docs/images/logo.png)

## 简介

**embedx** 是基于 c++ 开发的大规模 embedding 训练和推理系统，累计支持公司 `12 个业务`、 `30 多个团队使用`、`上线百余次`。

我们在以下推荐、搜索、支付 和 风控等产品落地使用了 embedx: `微信看一看`、`微信视频号`、`微信搜一搜`、`微信支付`、`微信安全`、
`腾讯新闻`、`应用宝`、`QQ 音乐`、`JOOX 音乐`、`腾讯课堂`、`领航平台` 和 `腾讯黑产打击` 等 ，并取得了性能和效果双丰收。

更多介绍请参考[详细介绍](docs/intro_embedx.md)。

EmbedX系统的论文发表在PVLDB'2023, 引用 cite：
```
@article{10.14778/3611540.3611546,
author = {Zou, Yuanhang and Ding, Zhihao and Shi, Jieming and Guo, Shuting and Su, Chunchen and Zhang, Yafei},
title = {EmbedX: A Versatile, Efficient and Scalable Platform to Embed Both Graphs and High-Dimensional Sparse Data},
year = {2023},
volume = {16},
number = {12},
url = {https://doi.org/10.14778/3611540.3611546},
journal = {Proc. VLDB Endow.},
pages = {3543–3556}
}
```

## **embedx** 已经实现的模型和评测

- 已经实现的模型

  - 十亿级节点、千亿级边的 **图模型**
  - 百亿级样本、百亿特征的 **深度排序、召回模型**
  - 十亿级节点、千亿级边与百亿级样本、百亿特征的 **图与深度排序、图与深度召回的联合建模模型**

- [模型以及评测](docs/model.md)

## 快速上手

- [编译](docs/compile.md)
- [数据格式](docs/data_format.md)
- [参数介绍](docs/param.md)
- 自有集群使用

  - [单机使用必读](docs/intro_to_using_single.md)
  - [分布式使用必读](docs/intro_to_using_dist.md)

- 在线推理

  - 使用方法参考[在线推理](docs/inference.md)

- 辅助工具

  - [随机游走](docs/random_walk.md)
  - [邻居特征平均](docs/average_feature.md)
  - [数据编码](docs/encode.md)

## Contributing

- [Contributing](CONTRIBUTING.md)

## 常见问题

- [常见问题](docs/faq.md)

## 更多问题可以联系开发者

- [Authors](docs/AUTHORS.md)
