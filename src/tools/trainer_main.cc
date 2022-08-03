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
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/shard.h>
#include <deepx_core/tensor/data_type.h>
#include <gflags/gflags.h>

#include <cmath>

#include "src/deep/client/deep_client.h"
#include "src/deep/deep_config.h"
#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/model_zoo.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/model_util.h"
#include "src/tools/shard_func_name.h"
#include "src/tools/trainer_context.h"

// used in both `graph server` and `training tasks`
DEFINE_int32(thread_num, 1, "Number of threads.");

// train
DEFINE_bool(gnn_model, true, "true for GNN models, false for NonGNN models.");
DEFINE_bool(deep_model, false,
            "true for Deep models, false for NonDeep models.");
DEFINE_string(model, "unsup_graphsage", "Model name.");
DEFINE_string(model_config, "", "Model config.");
DEFINE_string(instance_reader, "unsup_graphsage", "Instance reader name.");
DEFINE_string(instance_reader_config, "", "Instance reader config.");
DEFINE_string(optimizer, "adam", "Optimizer name.");
DEFINE_string(optimizer_config, "", "Optimizer config.");
DEFINE_int32(model_shard, 0,
             "Number of model shards, zero disables the model shard mode.");
DEFINE_string(in_model, "", "Input dir of model.");
DEFINE_string(in, "", "Input dir/file of training data.");
DEFINE_string(pretrain_path, "", "Input dir/file of pretrain param.");
DEFINE_string(item_feature, "", "Input dir/file of item feature.");
DEFINE_string(inst_file, "", "Input dir/file of instance file.");
DEFINE_string(freq_file, "", "Input dir/file of item frequency.");
DEFINE_bool(shuffle, true, "Shuffle input files for each epoch.");
DEFINE_int32(epoch, 1, "Number of epochs.");
DEFINE_int32(batch, 64, "Batch size.");
DEFINE_bool(ts_enable, false, "Enable timestamp.");
DEFINE_uint64(ts_now, 0, "Timestamp of now.");
DEFINE_uint64(ts_expire_threshold, 0, "Timestamp expiration threshold.");
DEFINE_uint64(freq_filter_threshold, 0, "Frequency filter threshold.");
DEFINE_int32(verbose, 1, "Verbose level: 0-10.");
DEFINE_int32(seed, 9527, "Seed of random engine.");
DEFINE_int32(target_type, 0, "0, for loss, 1 for prob, 2 for emb.");
DEFINE_bool(out_model_remove_zeros, false, "Remove zeros from output model.");
DEFINE_string(out_model, "", "Output dir of model (optional).");
DEFINE_string(out_model_text, "", "Output text dir of model(optional).");

namespace embedx {
namespace {

deepx_core::Shard FLAGS_shard;

/************************************************************************/
/* Trainer */
/************************************************************************/
class Trainer : public deepx_core::DataType {
 protected:
  std::unique_ptr<GraphClient> graph_client_;
  std::unique_ptr<DeepClient> deep_client_;
  std::unique_ptr<ModelUtil> model_util_;
  std::unique_ptr<ModelZoo> model_zoo_;
  deepx_core::Graph graph_;

  std::default_random_engine engine_;

  std::vector<std::string> files_;
  std::vector<std::string> remaining_files_;
  std::mutex file_mutex_;

  int epoch_ = 0;
  double epoch_loss_ = 0;
  double epoch_loss_weight_ = 0;
  std::mutex epoch_loss_mutex_;

  std::vector<std::unique_ptr<TrainerContext>> contexts_tls_;

 public:
  virtual ~Trainer() = default;
  virtual bool Init();
  virtual void Train();
  void TrainEntry(int thread_id);
  virtual void TrainFile(int thread_id, const std::string& file);
  virtual void Save();
  bool InitTrainerContext(TrainerContext* context) const;
};

bool Trainer::Init() {
  model_util_.reset(new ModelUtil(&graph_));

  DXCHECK(deepx_core::AutoFileSystem::ListRecursive(FLAGS_in, true, &files_));

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

  if (FLAGS_deep_model) {
    if (!FLAGS_item_feature.empty() || !FLAGS_inst_file.empty() ||
        !FLAGS_freq_file.empty()) {
      DeepConfig deep_config;

      if (!FLAGS_item_feature.empty()) {
        deep_config.set_item_feature(FLAGS_item_feature);
      }
      if (!FLAGS_inst_file.empty()) {
        deep_config.set_inst_file(FLAGS_inst_file);
      }
      if (!FLAGS_freq_file.empty()) {
        deep_config.set_freq_file(FLAGS_freq_file);
      }

      if (!FLAGS_node_config.empty()) {
        deep_config.set_node_config(FLAGS_node_config);
      }

      deep_config.set_negative_sampler_type(FLAGS_negative_sampler_type);
      deep_config.set_thread_num(FLAGS_thread_num);

      deep_client_ = NewDeepClient(deep_config, DeepClientEnum::LOCAL);
      DXCHECK(deep_client_ != nullptr);
    }
  }

  if (FLAGS_in_model.empty()) {
    model_zoo_ = NewModelZoo(FLAGS_model);
    DXCHECK(model_zoo_);
    deepx_core::StringMap config;
    DXCHECK(deepx_core::ParseConfig(FLAGS_model_config, &config));
    DXCHECK(model_zoo_->InitConfig(config));
    DXCHECK(model_zoo_->InitGraph(&graph_));
    graph_.meta()["model"] = FLAGS_model;
    graph_.meta()["model_config"] = FLAGS_model_config;
  } else {
    DXCHECK(deepx_core::LoadGraph(FLAGS_in_model, &graph_));
    model_zoo_ = NewModelZoo(graph_.meta().at("model"));
    DXCHECK(model_zoo_);
    deepx_core::StringMap config;
    DXCHECK(deepx_core::ParseConfig(graph_.meta().at("model_config"), &config));
    DXCHECK(model_zoo_->InitConfig(config));
  }

  engine_.seed(FLAGS_seed);

  std::string new_path;
  if (deepx_core::AutoFileSystem::BackupIfExists(FLAGS_out_model, &new_path)) {
    DXINFO("Backed up %s to %s.", FLAGS_out_model.c_str(), new_path.c_str());
  }

  if (!deepx_core::AutoFileSystem::Exists(FLAGS_out_model)) {
    DXCHECK(deepx_core::AutoFileSystem::MakeDir(FLAGS_out_model));
  }

  if (!FLAGS_out_model_text.empty()) {
    if (deepx_core::AutoFileSystem::BackupIfExists(FLAGS_out_model_text,
                                                   &new_path)) {
      DXINFO("Backed up %s to %s.", FLAGS_out_model_text.c_str(),
             new_path.c_str());
    }

    if (!deepx_core::AutoFileSystem::Exists(FLAGS_out_model_text)) {
      DXCHECK(deepx_core::AutoFileSystem::MakeDir(FLAGS_out_model_text));
    }
  }
  return true;
}

void Trainer::Train() {
  for (epoch_ = 0; epoch_ < FLAGS_epoch; ++epoch_) {
    DXINFO("Epoch: %d begins.", epoch_ + 1);

    DXCHECK(remaining_files_.empty());
    remaining_files_ = files_;
    if (FLAGS_shuffle) {
      std::shuffle(remaining_files_.begin(), remaining_files_.end(), engine_);
    }

    epoch_loss_ = 0;
    epoch_loss_weight_ = 0;

    std::vector<std::thread> threads;
    for (int j = 0; j < FLAGS_thread_num; ++j) {
      threads.emplace_back(&Trainer::TrainEntry, this, j);
    }
    for (auto& thread : threads) {
      thread.join();
    }

    DXINFO("Epoch: %d completed.", epoch_ + 1);
  }
}

void Trainer::TrainEntry(int thread_id) {
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
      DXINFO("[%d] [%3.1f%%] Training completed. ", thread_id, 100.0);
      break;
    }

    DXINFO("[%d] [%3.1f%%] Training %s...", thread_id,
           (100.0 - 100.0 * file_size / files_.size()), file.c_str());
    TrainFile(thread_id, file);
  }
}

void Trainer::TrainFile(int thread_id, const std::string& file) {
  TrainerContext* context = contexts_tls_[thread_id].get();
  context->TrainFile(thread_id, file);
  if (context->file_loss_weight() > 0) {
    double out_loss;
    {
      std::lock_guard<std::mutex> guard(epoch_loss_mutex_);
      epoch_loss_ += context->file_loss();
      epoch_loss_weight_ += context->file_loss_weight();
      out_loss = epoch_loss_ / epoch_loss_weight_;
    }
    DXINFO("epoch = %d, loss = %f.", epoch_ + 1, out_loss);
  }
}

void Trainer::Save() {
  DXCHECK(deepx_core::SaveGraph(FLAGS_out_model, graph_));
  DXCHECK(graph_.SaveDot(FLAGS_out_model + ".dot"));
  DXCHECK(deepx_core::SaveShard(FLAGS_out_model, FLAGS_shard));
}

bool Trainer::InitTrainerContext(TrainerContext* context) const {
  auto instance_reader_creator = [this]() {
    std::unique_ptr<EmbedInstanceReader> instance_reader(
        NewEmbedInstanceReader(FLAGS_instance_reader));
    DXCHECK(instance_reader);
    deepx_core::StringMap config;
    DXCHECK(deepx_core::ParseConfig(FLAGS_instance_reader_config, &config));
    config["batch"] = std::to_string(FLAGS_batch);
    DXCHECK(instance_reader->InitConfig(config));

    if (graph_client_) {
      DXCHECK(instance_reader->InitGraphClient(graph_client_.get()));
    }

    if (deep_client_) {
      DXCHECK(instance_reader->InitDeepClient(deep_client_.get()));
    }

    return instance_reader;
  };

  std::string target_name = graph_.target(FLAGS_target_type).name();

  if (FLAGS_freq_filter_threshold > 0) {
    context->set_freq_filter_threshold(
        (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold);
  }
  context->set_verbose(FLAGS_verbose);
  context->set_target_name(target_name);
  context->set_instance_reader_creator(instance_reader_creator);
  return true;
}

/************************************************************************/
/* TrainerNonShard */
/************************************************************************/
class TrainerNonShard : public Trainer {
 private:
  deepx_core::ModelShard model_shard_;

 public:
  bool Init() override;
  void Save() override;
};

bool TrainerNonShard::Init() {
  if (!Trainer::Init()) {
    return false;
  }

  model_shard_.seed(FLAGS_seed);
  model_shard_.InitShard(&FLAGS_shard, 0);
  model_shard_.InitGraph(&graph_);
  if (FLAGS_in_model.empty()) {
    DXCHECK(model_shard_.InitModel());
    DXCHECK(
        model_shard_.InitOptimizer(FLAGS_optimizer, FLAGS_optimizer_config));
  } else {
    DXCHECK(model_shard_.LoadModel(FLAGS_in_model));
    DXCHECK(model_shard_.LoadOptimizer(FLAGS_in_model, FLAGS_optimizer_config));
  }

  DXCHECK(!model_shard_.model().HasSRM());

  contexts_tls_.resize(FLAGS_thread_num);
  for (int i = 0; i < FLAGS_thread_num; ++i) {
    std::unique_ptr<TrainerContextNonShard> context(new TrainerContextNonShard);
    DXCHECK(InitTrainerContext(context.get()));
    DXCHECK(context->Init(&model_shard_));
    contexts_tls_[i] = std::move(context);
  }
  return true;
}

void TrainerNonShard::Save() {
  Trainer::Save();
  if (FLAGS_out_model_remove_zeros) {
    model_shard_.mutable_model()->RemoveZerosSRM();
  }

  DXCHECK(model_shard_.SaveModel(FLAGS_out_model));
  DXCHECK(model_shard_.SaveOptimizer(FLAGS_out_model));

  if (!FLAGS_out_model_text.empty()) {
    DXCHECK(model_shard_.SaveTextModel(FLAGS_out_model_text));
  }
}

/************************************************************************/
/* TrainerShard */
/************************************************************************/
class TrainerShard : public Trainer {
 private:
  int shard_size_ = 0;
  std::vector<deepx_core::ModelShard> model_shards_;
  std::vector<deepx_core::ModelShard> model_shards_tls_;

 public:
  bool Init() override;
  void Train() override;
  void Save() override;
};

bool TrainerShard::Init() {
  if (!Trainer::Init()) {
    return false;
  }

  shard_size_ = FLAGS_shard.shard_size();
  model_shards_.resize(shard_size_);
  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i].seed(i * FLAGS_seed);
    model_shards_[i].InitShard(&FLAGS_shard, i);
    model_shards_[i].InitGraph(&graph_);
    if (FLAGS_in_model.empty()) {
      DXCHECK(model_shards_[i].InitModel());
      if (!FLAGS_pretrain_path.empty()) {
        DXCHECK(model_util_->LoadPretrainParam(FLAGS_pretrain_path, FLAGS_shard,
                                               &model_shards_[i]));
      }
      DXCHECK(model_shards_[i].InitOptimizer(FLAGS_optimizer,
                                             FLAGS_optimizer_config));

      if (FLAGS_ts_enable) {
        DXCHECK(model_shards_[i].InitTSStore(
            (deepx_core::DataType::ts_t)FLAGS_ts_now,
            (deepx_core::DataType::ts_t)FLAGS_ts_expire_threshold));
      }

      if (FLAGS_freq_filter_threshold > 0) {
        DXCHECK(model_shards_[i].InitFreqStore(
            (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold));
      }
    } else {
      DXCHECK(model_shards_[i].LoadModel(FLAGS_in_model));
      DXCHECK(model_shards_[i].LoadOptimizer(FLAGS_in_model,
                                             FLAGS_optimizer_config));

      if (FLAGS_ts_enable) {
        if (!model_shards_[i].LoadTSStore(FLAGS_in_model, FLAGS_ts_now,
                                          FLAGS_ts_expire_threshold)) {
          DXCHECK(model_shards_[i].InitTSStore(
              (deepx_core::DataType::ts_t)FLAGS_ts_now,
              (deepx_core::DataType::ts_t)FLAGS_ts_expire_threshold));
        }
      }

      if (FLAGS_freq_filter_threshold > 0) {
        if (!model_shards_[i].LoadFreqStore(
                FLAGS_in_model,
                (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold)) {
          DXCHECK(model_shards_[i].InitFreqStore(
              (deepx_core::DataType::freq_t)FLAGS_freq_filter_threshold));
        }
      }
    }
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

void TrainerShard::Train() {
  for (int i = 0; i < shard_size_; ++i) {
    DXCHECK(model_shards_[i].InitThreadPool());
    model_shards_[i].StartThreadPool();
  }

  Trainer::Train();

  for (int i = 0; i < shard_size_; ++i) {
    model_shards_[i].StopThreadPool();
  }
}

void TrainerShard::Save() {
  Trainer::Save();
  for (int i = 0; i < shard_size_; ++i) {
    if (FLAGS_out_model_remove_zeros) {
      model_shards_[i].mutable_model()->RemoveZerosSRM();
    }

    if (FLAGS_ts_enable && FLAGS_ts_expire_threshold > 0) {
      model_shards_[i].ExpireTSStore();
    }

    DXCHECK_THROW(model_shards_[i].SaveModel(FLAGS_out_model));
    DXCHECK_THROW(model_shards_[i].SaveOptimizer(FLAGS_out_model));

    if (FLAGS_ts_enable) {
      DXCHECK_THROW(model_shards_[i].SaveTSStore(FLAGS_out_model));
    }

    if (FLAGS_freq_filter_threshold > 0) {
      DXCHECK_THROW(model_shards_[i].SaveFreqStore(FLAGS_out_model));
    }

    if (!FLAGS_out_model_text.empty()) {
      DXCHECK_THROW(model_shards_[i].SaveTextModel(FLAGS_out_model_text));
    }
  }
}

/************************************************************************/
/* main */
/************************************************************************/
void CheckGNNFlags() {
  deepx_core::AutoFileSystem fs;
  DXCHECK(!FLAGS_node_graph.empty());
  deepx_core::CanonicalizePath(&FLAGS_node_graph);
  DXCHECK(fs.Open(FLAGS_node_graph));
  if (deepx_core::IsStdinStdoutPath(FLAGS_node_graph)) {
    if (FLAGS_epoch != 1) {
      DXINFO("--epoch will be set to 1.");
      FLAGS_epoch = 1;
    }
    if (FLAGS_thread_num != 1) {
      DXINFO("--thread will be set to 1.");
      FLAGS_thread_num = 1;
    }
  }
  DXCHECK(FLAGS_negative_sampler_type == 0 ||
          FLAGS_negative_sampler_type == 1 ||
          FLAGS_negative_sampler_type == 2 || FLAGS_negative_sampler_type == 3);
  DXCHECK(FLAGS_neighbor_sampler_type == 0 ||
          FLAGS_neighbor_sampler_type == 1 ||
          FLAGS_neighbor_sampler_type == 2 || FLAGS_neighbor_sampler_type == 3);
}

void CheckFlags() {
  deepx_core::AutoFileSystem fs;

  if (FLAGS_gnn_model) {
    CheckGNNFlags();
  }

  DXCHECK(FLAGS_thread_num > 0);
  DXCHECK(!FLAGS_instance_reader.empty());
  DXCHECK(FLAGS_epoch > 0);
  DXCHECK(FLAGS_batch > 0);

  deepx_core::CanonicalizePath(&FLAGS_in_model);
  if (FLAGS_in_model.empty()) {
    DXCHECK(!FLAGS_model.empty());
    DXCHECK(!FLAGS_optimizer.empty());
    DXCHECK(FLAGS_model_shard >= 0);
  } else {
    DXINFO("--model will be ignored.");
    DXINFO("--model_config will be ignored.");
    DXINFO("--model_shard will be ignored.");
    DXINFO("--optimizer will be ignored.");
    DXINFO("--optimizer_config will be ignored.");
    DXCHECK(fs.Open(FLAGS_in_model));
    DXCHECK(!deepx_core::IsStdinStdoutPath(FLAGS_in_model));
  }
  DXCHECK(!FLAGS_in.empty());

  if (FLAGS_ts_enable) {
    DXCHECK(
        FLAGS_ts_now <=
        (google::uint64)std::numeric_limits<deepx_core::DataType::ts_t>::max());
    DXCHECK(
        FLAGS_ts_expire_threshold <=
        (google::uint64)std::numeric_limits<deepx_core::DataType::ts_t>::max());
  }

  DXCHECK(
      FLAGS_freq_filter_threshold <=
      (google::uint64)std::numeric_limits<deepx_core::DataType::freq_t>::max());

  DXCHECK(FLAGS_verbose >= 0);
  DXCHECK(FLAGS_target_type == 0);

  deepx_core::CanonicalizePath(&FLAGS_out_model);
  if (FLAGS_out_model.empty()) {
    if (deepx_core::IsStdinStdoutPath(FLAGS_node_graph)) {
      FLAGS_out_model = "stdin.train";
    } else {
      FLAGS_out_model = FLAGS_node_graph + ".train";
    }
    DXINFO("Didn't specify --out_model, output to: %s.",
           FLAGS_out_model.c_str());
  }
  DXCHECK(fs.Open(FLAGS_out_model));
  DXCHECK(!deepx_core::IsStdinStdoutPath(FLAGS_out_model));

  deepx_core::CanonicalizePath(&FLAGS_out_model_text);
  if (!FLAGS_out_model_text.empty()) {
    DXCHECK(fs.Open(FLAGS_out_model_text));
    DXCHECK(!deepx_core::IsStdinStdoutPath(FLAGS_out_model_text));
  }

  if (FLAGS_model_shard == 0) {
    FLAGS_shard.InitNonShard();
  } else {
    FLAGS_shard.InitShard(FLAGS_model_shard, MOD9973_NAME);
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

  std::unique_ptr<Trainer> trainer;
  if (FLAGS_shard.shard_mode() == 0) {
    trainer.reset(new TrainerNonShard);
  } else {
    trainer.reset(new TrainerShard);
  }

  DXCHECK(trainer->Init());
  trainer->Train();
  trainer->Save();

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
