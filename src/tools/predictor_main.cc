// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include <deepx_core/common/any_map.h>
#include <deepx_core/common/misc.h>
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/shard.h>
#include <deepx_core/tensor/data_type.h>
#include <gflags/gflags.h>

#include "src/graph/client/graph_client.h"
#include "src/model/embed_instance_reader.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/shard_func_name.h"
#include "src/tools/trainer_context.h"

// used in both `graph server` and `predict tasks`
DEFINE_int32(thread_num, 1, "Number of threads.");

// predict
DEFINE_bool(gnn_model, true, "true for GNN models, false for NonGNN models.");
DEFINE_string(instance_reader, "unsup_graphsage", "Instance reader name.");
DEFINE_string(instance_reader_config, "", "Instance reader config.");
DEFINE_int32(batch, 32, "Batch size.");
DEFINE_string(in, "", "Input dir/file of testing data.");
DEFINE_string(in_model, "", "Input model dir.");
DEFINE_int32(target_type, 2, "0 for loss, 1 for prob, 2 for emb.");
DEFINE_int32(verbose, 1, "Verbose level: 0-10.");
DEFINE_string(out_predict, "", "Output predict dir.");

namespace embedx {
namespace {

deepx_core::Shard FLAGS_shard;

/************************************************************************/
/* Predictor */
/************************************************************************/
class Predictor : public deepx_core::DataType {
 protected:
  std::unique_ptr<GraphClient> graph_client_;
  deepx_core::Graph graph_;

  std::vector<std::string> files_;
  std::vector<std::string> remaining_files_;
  std::mutex file_mutex_;

  std::vector<std::unique_ptr<TrainerContext>> contexts_tls_;

 public:
  virtual ~Predictor() = default;
  virtual bool Init();
  virtual void Predict();
  virtual void PredictEntry(int thread_id);
  virtual void PredictFile(int thread_id, const std::string& in_file,
                           const std::string& out_file);
  bool InitTrainerContext(TrainerContext* context) const;
};

bool Predictor::Init() {
  if (FLAGS_gnn_model) {
    GraphConfig graph_config;
    graph_config.set_node_graph(FLAGS_node_graph);
    graph_config.set_node_config(FLAGS_node_config);
    if (!FLAGS_node_feature.empty()) {
      graph_config.set_node_feature(FLAGS_node_feature);
    }
    if (!FLAGS_neighbor_feature.empty()) {
      graph_config.set_neighbor_feature(FLAGS_neighbor_feature);
    }
    graph_config.set_negative_sampler_type(FLAGS_negative_sampler_type);
    graph_config.set_neighbor_sampler_type(FLAGS_neighbor_sampler_type);
    graph_config.set_thread_num(FLAGS_thread_num);

    graph_client_ = NewGraphClient(graph_config, GraphClientEnum::LOCAL);
    if (!graph_client_) {
      return false;
    }
  }

  DXCHECK(deepx_core::LoadGraph(FLAGS_in_model, &graph_));

  DXCHECK(deepx_core::AutoFileSystem::ListRecursive(FLAGS_in, true, &files_));
  FLAGS_thread_num = std::min(FLAGS_thread_num, (int)files_.size());

  if (!deepx_core::AutoFileSystem::Exists(FLAGS_out_predict)) {
    DXCHECK(deepx_core::AutoFileSystem::MakeDir(FLAGS_out_predict));
  }
  return true;
}

void Predictor::Predict() {
  DXCHECK(remaining_files_.empty());
  remaining_files_ = files_;
  std::vector<std::thread> threads;
  for (int j = 0; j < FLAGS_thread_num; ++j) {
    threads.emplace_back(&Predictor::PredictEntry, this, j);
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

void Predictor::PredictEntry(int thread_id) {
  for (;;) {
    size_t file_size = 0;
    std::string file;
    {
      std::lock_guard<std::mutex> guard(file_mutex_);
      if (!remaining_files_.empty()) {
        file_size = remaining_files_.size();
        file = remaining_files_.back();
        remaining_files_.pop_back();
      }
    }

    if (file.empty()) {
      DXINFO("[%d] [%3.1f%%] Predicting completed. ", thread_id, 100.0);
      break;
    }

    DXINFO("[%d] [%3.1f%%] Predicting %s...", thread_id,
           (100.0 - 100.0 * file_size / files_.size()), file.c_str());
    PredictFile(thread_id, file,
                deepx_core::GetOutputPredictFile(FLAGS_out_predict, file));
  }
}

void Predictor::PredictFile(int thread_id, const std::string& in_file,
                            const std::string& out_file) {
  TrainerContext* context = contexts_tls_[thread_id].get();
  context->PredictFile(thread_id, in_file, out_file);
}

bool Predictor::InitTrainerContext(TrainerContext* context) const {
  auto instance_reader_creator = [this]() {
    std::unique_ptr<EmbedInstanceReader> instance_reader(
        NewEmbedInstanceReader(FLAGS_instance_reader));
    DXCHECK(instance_reader);
    deepx_core::StringMap config;
    DXCHECK(deepx_core::ParseConfig(FLAGS_instance_reader_config, &config));
    config["batch"] = std::to_string(FLAGS_batch);
    DXCHECK(instance_reader->InitConfig(config));

    if (FLAGS_gnn_model) {
      // GNN models
      DXCHECK(instance_reader->InitGraphClient(graph_client_.get()));
      instance_reader->PostInit(FLAGS_node_config);
    }

    return instance_reader;
  };

  std::string target_name = graph_.target(FLAGS_target_type).name();

  context->set_verbose(FLAGS_verbose);
  context->set_target_name(target_name);
  context->set_target_type(FLAGS_target_type);
  context->set_instance_reader_creator(instance_reader_creator);
  return true;
}

/************************************************************************/
/* PredictorNonShard */
/************************************************************************/
class PredictorNonShard : public Predictor {
 private:
  deepx_core::ModelShard model_shard_;

 public:
  bool Init() override;
};

bool PredictorNonShard::Init() {
  if (!Predictor::Init()) {
    return false;
  }

  model_shard_.seed(0);
  model_shard_.InitShard(&FLAGS_shard, 0);
  model_shard_.InitGraph(&graph_);
  DXCHECK(model_shard_.LoadModel(FLAGS_in_model));

  contexts_tls_.resize(FLAGS_thread_num);
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    std::unique_ptr<TrainerContextNonShard> context(new TrainerContextNonShard);
    DXCHECK(InitTrainerContext(context.get()));
    DXCHECK(context->Init(&model_shard_));
    contexts_tls_[i] = std::move(context);
  }
  return true;
}

/************************************************************************/
/* PredictorShard */
/************************************************************************/
class PredictorShard : public Predictor {
 private:
  int shard_size_ = 0;
  std::vector<deepx_core::ModelShard> model_shards_;
  std::vector<deepx_core::ModelShard> model_shards_tls_;

 public:
  bool Init() override;
  void Predict() override;
};

bool PredictorShard::Init() {
  if (!Predictor::Init()) {
    return false;
  }

  shard_size_ = FLAGS_shard.shard_size();
  model_shards_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i].seed(0);
    model_shards_[i].InitShard(&FLAGS_shard, i);
    model_shards_[i].InitGraph(&graph_);
    DXCHECK(model_shards_[i].LoadModel(FLAGS_in_model));
  }

  contexts_tls_.resize(FLAGS_thread_num);
  model_shards_tls_.resize(FLAGS_thread_num);
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    model_shards_tls_[i].seed(0);
    model_shards_tls_[i].InitShard(&FLAGS_shard, 0);
    model_shards_tls_[i].InitGraph(&graph_);
    DXCHECK(model_shards_tls_[i].InitModelPlaceholder());
    std::unique_ptr<TrainerContextShard> context(new TrainerContextShard);
    DXCHECK(InitTrainerContext(context.get()));
    DXCHECK(context->Init(&model_shards_, &model_shards_tls_[i]));
    contexts_tls_[i] = std::move(context);
  }
  return true;
}

void PredictorShard::Predict() {
  for (int i = 0; i < shard_size_; ++i) {
    DXCHECK(model_shards_[i].InitThreadPool());
    model_shards_[i].StartThreadPool();
  }

  Predictor::Predict();

  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i].StopThreadPool();
  }
}

/************************************************************************/
/* main */
/************************************************************************/
void CheckGNNFlags() {
  // for GNN models
  // 1 : dump classification prob
  // 2 : dump node embedding
  DXCHECK(FLAGS_target_type == 1 || FLAGS_target_type == 2);

  deepx_core::CanonicalizePath(&FLAGS_node_graph);
  DXCHECK(!FLAGS_node_graph.empty());
  DXCHECK(FLAGS_negative_sampler_type == 0 ||
          FLAGS_negative_sampler_type == 1 ||
          FLAGS_negative_sampler_type == 2 || FLAGS_negative_sampler_type == 3);
  DXCHECK(FLAGS_neighbor_sampler_type == 0 ||
          FLAGS_neighbor_sampler_type == 1 ||
          FLAGS_neighbor_sampler_type == 2 || FLAGS_neighbor_sampler_type == 3);
}

void CheckNonGNNFlags() {
  // for NonGNN models
  // 1 : dump classification prob
  // 2 : dump user embedding
  // 3 : dump item embedding
  DXCHECK(FLAGS_target_type == 1 || FLAGS_target_type == 2 ||
          FLAGS_target_type == 3);
}

void CheckFlags() {
  deepx_core::AutoFileSystem fs;

  if (FLAGS_gnn_model) {
    CheckGNNFlags();
  } else {
    CheckNonGNNFlags();
  }

  DXCHECK(FLAGS_thread_num > 0);
  DXCHECK(!FLAGS_instance_reader.empty());
  DXCHECK(FLAGS_batch > 0);

  DXCHECK(!FLAGS_in.empty());
  deepx_core::CanonicalizePath(&FLAGS_in_model);
  DXCHECK(!FLAGS_in_model.empty());
  DXCHECK(fs.Open(FLAGS_in_model));
  DXCHECK(!deepx_core::IsStdinStdoutPath(FLAGS_in_model));

  DXCHECK(FLAGS_verbose >= 0);

  DXCHECK(LoadShard(FLAGS_in_model, &FLAGS_shard));
  if (FLAGS_shard.shard_mode() == 1 &&
      FLAGS_shard.shard_func_name() != MOD9973_NAME) {
    FLAGS_shard.InitShard(FLAGS_shard.shard_size(), MOD9973_NAME);
  }
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  std::unique_ptr<Predictor> predictor;
  if (FLAGS_shard.shard_mode() == 0) {
    predictor.reset(new PredictorNonShard);
  } else {
    predictor.reset(new PredictorShard);
  }

  DXCHECK(predictor->Init());
  predictor->Predict();

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
