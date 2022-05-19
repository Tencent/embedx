// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include "src/graph/post_builder.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64
#include <mutex>
#include <unordered_set>

#include "src/io/io_util.h"

namespace embedx {
namespace {

class PostBuilderHelper {
 private:
  const id_name_t& id_name_map_;
  uint16_t ns_size_ = 1;
  int estimated_size_ = 0;
  const Storage* store_ = nullptr;

  std::vector<vec_int_t>* uniq_nodes_list_ = nullptr;
  std::vector<vec_float_t>* uniq_freqs_list_ = nullptr;
  vec_int_t* total_freqs_ = nullptr;

  std::mutex mtx_;
  std::unordered_set<int_t> node_set_;
  std::vector<weight_map_t> node_weight_maps_;

 public:
  PostBuilderHelper(const id_name_t& id_name_map, uint16_t ns_size,
                    int estimated_size, const Storage* storage)
      : id_name_map_(id_name_map),
        ns_size_(ns_size),
        estimated_size_(estimated_size),
        store_(storage) {}

 public:
  bool Build(std::vector<vec_int_t>* uniq_nodes_list,
             std::vector<vec_float_t>* uniq_freqs_list, vec_int_t* total_freqs,
             int thread_num);

 private:
  void Clear() noexcept;
  void Prepare(uint16_t ns_size, uint64_t estimated_size);
  bool Insert(int_t node);
  void AccumulateFreqs();
  bool Process(const vec_int_t& nodes, int thread_num);
  bool ProcessEntry(const vec_int_t& nodes, int thread_id);
};

bool PostBuilderHelper::Build(std::vector<vec_int_t>* uniq_nodes_list,
                              std::vector<vec_float_t>* uniq_freqs_list,
                              vec_int_t* total_freqs, int thread_num) {
  uniq_nodes_list_ = uniq_nodes_list;
  uniq_freqs_list_ = uniq_freqs_list;
  total_freqs_ = total_freqs;

  Clear();
  Prepare(ns_size_, estimated_size_);

  return Process(store_->Keys(), thread_num);
}

void PostBuilderHelper::Clear() noexcept {
  uniq_nodes_list_->clear();
  uniq_freqs_list_->clear();
  total_freqs_->clear();

  node_set_.clear();
  node_weight_maps_.clear();
}

void PostBuilderHelper::Prepare(uint16_t ns_size, uint64_t estimated_size) {
  uniq_nodes_list_->resize(ns_size);
  uniq_freqs_list_->resize(ns_size);
  total_freqs_->resize(ns_size, 0);

  node_set_.reserve(estimated_size);
  node_weight_maps_.resize(ns_size);
}

bool PostBuilderHelper::Insert(int_t node) {
  uint16_t ns_id = io_util::GetNodeType(node);
  if (id_name_map_.find(ns_id) == id_name_map_.end()) {
    DXERROR("Couldn't find node: %" PRIu64
            " namespace id: %d in the config file.",
            node, (int)ns_id);
    return false;
  }

  if (node_set_.find(node) == node_set_.end()) {
    node_set_.insert(node);
    (*uniq_nodes_list_)[ns_id].emplace_back(node);
  }

  auto& node_weight_map = node_weight_maps_[ns_id];
  auto it = node_weight_map.find(node);
  if (it == node_weight_map.end()) {
    node_weight_map.emplace(node, 1.0);
  } else {
    it->second += 1.0;
  }

  (*total_freqs_)[ns_id] += 1;
  return true;
}

void PostBuilderHelper::AccumulateFreqs() {
  for (size_t i = 0; i < uniq_freqs_list_->size(); ++i) {
    const auto& nodes = (*uniq_nodes_list_)[i];
    auto& freqs = (*uniq_freqs_list_)[i];
    freqs.reserve(nodes.size());

    for (auto node : nodes) {
      auto ns_id = io_util::GetNodeType(node);
      auto it = node_weight_maps_[ns_id].find(node);
      DXCHECK(it != node_weight_maps_[ns_id].end());
      freqs.emplace_back(it->second);
    }
  }
}

bool PostBuilderHelper::Process(const vec_int_t& nodes, int thread_num) {
  DXINFO("Post building ...");

  if (!io_util::ParallelProcess<int_t>(
          nodes,
          [this](const vec_int_t& nodes, int thread_id) {
            return ProcessEntry(nodes, thread_id);
          },
          thread_num)) {
    DXERROR("Failed to post building.");
    return false;
  }

  AccumulateFreqs();

  DXINFO("Done.");
  return true;
}

bool PostBuilderHelper::ProcessEntry(const vec_int_t& nodes, int thread_id) {
  DXINFO("Thread: %d is processing ...", thread_id);

  for (auto node : nodes) {
    // node
    {
      std::lock_guard<std::mutex> guard(mtx_);
      if (!Insert(node)) {
        return false;
      }
    }

    // context
    const auto* context = store_->FindNeighbor(node);
    if (context == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", node);
      return false;
    }
    for (auto& entry : *context) {
      {
        std::lock_guard<std::mutex> guard(mtx_);
        if (!Insert(entry.first)) {
          return false;
        }
      }
    }
  }

  DXINFO("Done.");
  return true;
}

}  // namespace

/************************************************************************/
/* Post build graph */
/************************************************************************/
bool PostBuilder::Build(const std::string& config, int thread_num) {
  if (!io_util::LoadConfig(config, &ns_size_, &id_name_map_)) {
    return false;
  }

  PostBuilderHelper builder_helper(id_name_map_, ns_size_, estimated_size_,
                                   store_);
  return builder_helper.Build(&uniq_nodes_list_, &uniq_freqs_list_,
                              &total_freqs_, thread_num);
}

std::unique_ptr<PostBuilder> PostBuilder::Create(const Storage* store,
                                                 const GraphConfig& config) {
  std::unique_ptr<PostBuilder> post_builder;
  post_builder.reset(new PostBuilder());

  post_builder->set_store(store);
  post_builder->set_estimated_size(config.estimated_size());

  if (!post_builder->Build(config.node_config(), config.thread_num())) {
    DXERROR("Failed to create post builder.");
    post_builder.reset();
  }

  return post_builder;
}

}  // namespace embedx
