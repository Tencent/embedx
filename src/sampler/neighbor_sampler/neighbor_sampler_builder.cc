// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/sampler/neighbor_sampler/neighbor_sampler_builder.h"

#include <deepx_core/dx_log.h>

#include <cinttypes>  // PRIu64

#include "src/common/random.h"
#include "src/io/io_util.h"

namespace embedx {
namespace {

bool NormNeighborProb(const SamplerSource& sampler_source, int_t node,
                      vec_float_t* probs) {
  const auto* context = sampler_source.FindContext(node);
  if (context == nullptr) {
    DXERROR("Couldn't find node: %" PRIu64 " context.", node);
    return false;
  }

  float_t sum = 0;
  for (const auto& entry : *context) {
    if (entry.second <= 0) {
      DXERROR("Weight %f of node: %" PRIu64 " and neighbor: %" PRIu64
              " must be greater than 0.",
              entry.second, node, entry.first);
      return false;
    }
    sum += entry.second;
  }

  probs->clear();
  for (const auto& entry : *context) {
    probs->emplace_back(entry.second / sum);
  }

  return true;
}

}  // namespace

std::unique_ptr<SamplerBuilder> NeighborSamplerBuilder::Create(
    const SamplerSource* sampler_source, int sampler_type, int thread_num) {
  std::unique_ptr<SamplerBuilder> sampler_builder;
  sampler_builder.reset(
      new NeighborSamplerBuilder(sampler_source, sampler_type, thread_num));

  if (!sampler_builder->Init()) {
    DXERROR("Failed to init neighbor sampler builder.");
    sampler_builder.reset();
  }

  return sampler_builder;
}

bool NeighborSamplerBuilder::InitUniformFuncs() {
  DXINFO("Initing uniform neighbor sampler funcs...");

  next_func_ = [this](int_t cur_node, int_t* next_node) -> bool {
    const auto* context = sampler_source_.FindContext(cur_node);
    if (context == nullptr) {
      return false;
    }
    int k = int(ThreadLocalRandom() * context->size());
    *next_node = (*context)[k].first;
    return true;
  };

  range_next_func_ = [this](int_t cur_node, int begin, int end,
                            int_t* next_node) -> bool {
    const auto* context = sampler_source_.FindContext(cur_node);
    if (context == nullptr) {
      return false;
    }
    int k = begin + int(ThreadLocalRandom() * (end - begin));
    *next_node = (*context)[k].first;
    return true;
  };

  DXINFO("Done.");
  return true;
}

bool NeighborSamplerBuilder::InitFrequencySampler() {
  DXINFO("Building transition probability...");
  auto& nodes = sampler_source_.node_keys();
  sampling_map_.clear();
  if (!io_util::ParallelProcess<int_t>(
          nodes,
          [this](const vec_int_t& nodes, int thread_id) {
            return InitEntry(nodes, thread_id);
          },
          thread_num_)) {
    DXERROR("Failed to build Transition Probs.");
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool NeighborSamplerBuilder::InitFrequencyFuncs() {
  DXINFO("Initing frequency neighbor sampler funcs, with sampler_type: %d...",
         sampling_type_);

  next_func_ = [this](int_t cur_node, int_t* next_node) -> bool {
    const auto* context = sampler_source_.FindContext(cur_node);
    if (context == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", cur_node);
      return false;
    }

    auto it = sampling_map_.find(cur_node);
    if (it == sampling_map_.end()) {
      DXERROR("Couldn't find node: %" PRIu64 " sampler.", cur_node);
      return false;
    }

    int k = int(it->second->Next());
    *next_node = (*context)[k].first;
    return true;
  };

  range_next_func_ = [this](int_t cur_node, int begin, int end,
                            int_t* next_node) -> bool {
    const auto* context = sampler_source_.FindContext(cur_node);
    if (context == nullptr) {
      DXERROR("Couldn't find node: %" PRIu64 " context.", cur_node);
      return false;
    }

    auto it = sampling_map_.find(cur_node);
    if (it == sampling_map_.end()) {
      DXERROR("Couldn't find node: %" PRIu64 " sampler.", cur_node);
      return false;
    }

    int k = int(it->second->Next(begin, end));
    *next_node = (*context)[k].first;
    return true;
  };

  DXINFO("Done.");
  return true;
}

bool NeighborSamplerBuilder::InitEntry(const vec_int_t& nodes, int thread_id) {
  DXINFO("Thread: %d is processing...", thread_id);

  vec_float_t norm_probs;
  // more namespace
  for (const auto& node : nodes) {
    if (!NormNeighborProb(sampler_source_, node, &norm_probs)) {
      return false;
    }

    auto sampling = NewSampling(&norm_probs, (SamplingEnum)sampling_type_);
    if (!sampling) {
      return false;
    }
    std::lock_guard<std::mutex> guard(mtx_);
    sampling_map_.emplace(node, std::move(sampling));
  }

  DXINFO("Done.");
  return true;
}

std::unique_ptr<SamplerBuilder> NewNeighborSamplerBuilder(
    const SamplerSource* sampler_source, int sampler_type, int thread_num) {
  return NeighborSamplerBuilder::Create(sampler_source, sampler_type,
                                        thread_num);
}

}  // namespace embedx
