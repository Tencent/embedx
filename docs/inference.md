# 在线推理

本例子提供静态库的在线推理方案.

## 头文件和核心类

头文件是 ["model\_server.h"](../include/embedx/model_server.h).

核心类是 ModelServer.

```c++
using feature_t = std::pair<uint64_t, float>;
using features_t = std::vector<feature_t>;
using embedding_t = std::vector<float>;

class ModelServer {
 private:
  std::unique_ptr<deepx_core::Graph> graph_;
  std::string target_name_;
  std::unique_ptr<deepx_core::Model> model_;
  // 1 for classification prob
  // 2 for user embedding
  // 3 for item embedding
  int target_type_ = 2;

 public:
  ModelServer();
  ~ModelServer();
  ModelServer(const ModelServer&) = delete;
  ModelServer& operator=(const ModelServer&) = delete;

 public:
  void set_target_type(int target_type) noexcept { target_type_ = target_type; }

 public:
  // 加载模型文件, 返回是否成功.
  bool Load(const std::string& file);
  // 加载计算图文件, 返回是否成功.
  bool LoadGraph(const std::string& file);
  // 加载模型参数文件, 返回是否成功.
  bool LoadModel(const std::string& file);

 public:
  // 预测1条样本, 返回是否成功.
  // 输出1个预测值.
  bool Predict(const features_t& features, float* prob) const;
  // 预测1条样本, 返回是否成功.
  // 输出n个预测值.
  bool Predict(const features_t& features, std::vector<float>* probs) const;
  // 预测1批样本, 返回是否成功.
  // 输出batch * 1个预测值.
  //
  // 'batch_features'不能为空.
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  // 预测1批样本, 返回是否成功.
  // 输出batch * n个预测值.
  //
  // 'batch_features'不能为空.
  bool BatchPredict(const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  // 预测1条样本, 返回是否成功.
  // 输出1个embedding.
  bool PredictUserEmbedding(const features_t& user_features,
                            embedding_t* embedding) const;
  // 预测1批样本, 返回是否成功.
  // 输出n个embedding.
  bool BatchPredictUserEmbedding(
      const std::vector<features_t>& batch_user_features,
      std::vector<embedding_t>* embeddings) const;
  // 为GraphDeepFM模型预测1批样本, 返回是否成功.
  // 输出batch * 1个预测值.
  //
  // 'batch_features'不能为空.
  bool BatchGraphDeepFMPredict(const std::vector<features_t>& batch_features,
                               const std::vector<features_t>& batch_users,
                               std::vector<float>* batch_prob) const;

 public:
  std::unique_ptr<OpContext, void (*)(OpContext*)> NewOpContext() const;
  // 下面的几个Predict函数和上面的对应.
  // 它们接受'NewOpContext'返回的'OpContext'对象, 它们通常用来复用'OpContext'对象.
  bool Predict(OpContext* op_context, const features_t& features,
               float* prob) const;
  bool Predict(OpContext* op_context, const features_t& features,
               std::vector<float>* probs) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<float>* batch_prob) const;
  bool BatchPredict(OpContext* op_context,
                    const std::vector<features_t>& batch_features,
                    std::vector<std::vector<float>>* batch_probs) const;
  bool PredictUserEmbedding(OpContext* op_context,
                            const features_t& user_features,
                            embedding_t* embedding) const;
  bool BatchPredictUserEmbedding(
      OpContext* op_context, const std::vector<features_t>& batch_user_features,
      std::vector<embedding_t>* embeddings) const;
  bool BatchGraphDeepFMPredict(OpContext* op_context,
                               const std::vector<features_t>& batch_features,
                               const std::vector<features_t>& batch_users,
                               std::vector<float>* batch_prob) const;
};
```

多线程安全性.

- 多线程调用 LoadXXX, 不多线程安全.
- 多线程调用 LoadXXX, XXXPredict, 不多线程安全.

模型更新时, 涉及多线程调用 LoadXXX, XXXPredict.
通常采用"双词表"或"加锁"的方式保证多线程安全.

ModelServer 的使用参考 ["model\_server\_demo\_main.cc"](../src/tools/model_server_demo_main.cc).

## 模型文件, 计算图文件, 模型参数文件和库文件

- 参考[在线推理](https://github.com/Tencent/deepx_core/blob/master/example/rank/README.md#在线推理)
