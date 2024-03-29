# 编码

[TOC]

embedx 使用 `uint64` 表示 ***节点*** 和 ***特征***，它使用 `前 16 位` 表示 `类型`，`后 48 位` 表示 `id`。

生成以上类型的数据需要用户指定 `类型` 信息，通过 ***编码*** 操作将类型和 id 结合起来生成，程序运行时 ***自动解码*** 出类型和 id 信息。

以下依次介绍 `编码方案`、`节点如何编码` 和 `特征如何编码`。

## 编码方案

- 编码方案的函数形式

  ```shell
  编码函数(group_id，id) = 编码后 id
  ```

  - 输入是 group_id 和 id，输出是 `编码后 id`
  - 其中 `group_id` 是 `类型` 信息，`int` 类型，由用户指定，比如 `11`、`12`
  - `id` 是 `节点` 或者 `特征` 的 `ID 化` 表示，`uint64` 类型
  - `编码后 id` 是编码函数生成的数据，`uint64` 类型

- 编码方案的函数实现，参考[make_encoder_id](../demo/single/encoder/encoder.py)

## 节点如何编码

### 节点类型

节点类型可划分为 `同构节点` 和 `异构节点`。

- 同构节点，数据中只有同一种类型的节点，一般 ***不需要进行编码***
- 异构节点，数据中包括多种类型的节点

  - 表示 `异构节点` 需要将 `节点类型` 编码进节点
  - 将使用的 `节点类型` 放入到 `node_config 文件`，它用于程序运行时的解码操作

以下分别介绍 `如何编码`、`编码示例` 和 `生成 node_config 文件`。

### 如何编码

- 用户指定节点类型，也就是编码函数中的 `group_id`
- 调用[make_encoder_id](../demo/single/encoder/encoder.py) 函数

### 编码示例

- 示例 1，"节点类型：user，节点 id：10000"

  - 指定节点类型 `user` 为 `11` 即 `group_id = 11`
  - 调用`make_encoder_id(group_id=11, id=10000)` 生成新的节点 id
  - 新的节点 id 为 `make_encoder_id(11，10000) = 3096224743827216`

- 示例 2，"节点类型：video，节点 id：10000"

  - 指定节点类型 `video` 为 `12` 即 `group_id = 12`
  - 调用`make_encoder_id(group_id=12, id=10000)` 生成新的节点 id
  - 新的节点 id 为 `make_encoder_id(12，10000) = 2814749767106560012`

### 生成 node_config 文件

- 格式

```shell
group_name group_id
```

- 格式介绍

  - 每行以空格分隔
  - `group_name` 是 `string` 类型，用来记录 `group_id` 的意义，给一个有意义的名字即可
  - `group_id` 是 `int` 类型，由用户指定

- 如何生成

  - 编码示例 1 中的 "节点类型：user，节点 id：10000"
    `group_id` 为 `11`，`group_name` 可以是 `user` 或 `user_node` 或其他

  - 编码示例 2 中的 "节点类型：video，节点 id：10000"
    `group_id` 为 `12`，`group_name` 可以是 `video` 或 `video_node` 或其他

  - 以上生成一个两行的 `node_config` 文件，分别是 `user 11`、`video 12`

## 特征如何编码

### 节点类型

特征类型可划分为 `一个特征组特征` 和 `多个特征组特征`。

- 一个特征组特征，数据中只有一个特征组类型的特征，一般 ***不需要进行编码***
- 多个特征组特征，数据中包括多个特征组类型的特征

  - 表示 `多个特征组特征` 需要将 `特征组类型` 编码进特征
  - 并且将使用的 `特征组类型` 放入到 `config 文件`，它用于程序运行时的解码操作

以下分别介绍 `如何编码`、`编码示例` 和 `生成 config 文件`。

### 如何编码

- 用户指定特征组值，也就是编码函数中的 `group_id`
- 调用[make_encoder_id](../demo/single/encoder/encoder.py)函数

### 编码示例

- 示例 1，"特征组类型：年龄，特征组 id：10"

  - 指定特征组类型 `年龄` 为 `1` 即 `group_id = 1`
  - 调用`make_encoder_id(group_id=1, id=10)` 生成新的节点 id
  - 新的节点 id 为 `make_encoder_id(1，10) = 281474976710666`

- 示例 2，"特征组类型：性别，特征组 id：男"

  - 指定特征组类型 `性别` 为 `12` 即 `group_id = 12`
  - 性别特征离散化，假设 `男对应 0`、`女对应 1` 和 `未知对应 2`
  - 调用`make_encoder_id(group_id=12, id=0)` 生成新的节点 id
  - 新的节点 id 为`make_encoder_id(12，0) = 3377699720527872`

### 生成 config 文件

- 格式

```shell
group_id row col
```

- 格式介绍

  - 每行以空格分隔
  - `group_id` 是 `int` 类型，由用户指定，是此特征组对应的 id 信息
  - `row` 是 `int` 类型，由用户指定，是此特征组的特征空间的大小
  - `col` 是 `int` 类型，由用户指定，是此特征组的中特征 embedding 维度

- 如何生成

  - 上述示例 1 中的 "特征组类型：年龄，特征组 id：10"
    - `group_id` 为 `1`
    - 假设特征空间为 100，那么 `row=100`
    - 假设每个特征 embeding 维度是 128，那么 `col=128`

  - 上述示例 2 中的 "特征组类型：性别，特征组 id：男"
    - `group_id` 为 `12`
    - 假设特征空间为 3，那么 `row=3`
    - 假设每个特征 embeding 维度是 64，那么 `col=64`

  - 以上生成一个两行的 `config` 文件，分别是 `1 100 128`、`12 3 64`
