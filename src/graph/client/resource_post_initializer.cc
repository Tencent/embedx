// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng(chengchuancoder@gmail.com)
//

#include "src/graph/client/resource_post_initializer.h"

#include <deepx_core/dx_log.h>

#include <vector>

#include "src/graph/cache/cache_storage.h"
#include "src/graph/data_op/cache_storage_lookuper_op/dist_cache_storage_lookuper.h"
#include "src/graph/data_op/gs_op_factory.h"
#include "src/graph/data_op/meta_lookuper_op/dist_meta_lookuper.h"
#include "src/sampler/sampling.h"

namespace embedx {
namespace {

bool BuildCacheStorage(CacheStorage* cache_storage) {
  auto* op = graph_op::DistGSOpFactory::GetInstance()->LookupOrCreate(
      "DistCacheStorageLookuper");
  DXCHECK(op != nullptr);

  DXINFO("Building cache storage...");
  if (!dynamic_cast<graph_op::DistCacheStorageLookuper*>(op)->Run(
          cache_storage)) {
    DXERROR("Failed to build cache storage.");
    return false;
  }

  DXINFO("Done.");
  return true;
}

bool FetchServerDistribution(int shard_num, int* ns_size, vec_float_t* probs) {
  std::vector<vec_int_t> node_freqs_list;
  auto* op = graph_op::DistGSOpFactory::GetInstance()->LookupOrCreate(
      "DistMetaLookuper");
  DXCHECK(op != nullptr);
  if (!dynamic_cast<graph_op::DistMetaLookuper*>(op)->Run(&node_freqs_list)) {
    DXERROR("Failed to fetch server distribution.");
    return false;
  }

  DXINFO("Fetching server distribution...");
  *ns_size = (int)node_freqs_list.size();
  float_t freq_sum = 0;
  vec_float_t freq_sums(shard_num, 0);
  for (int i = 0; i < shard_num; ++i) {
    for (int j = 0; j < *ns_size; ++j) {
      freq_sum += node_freqs_list[j][i];
      freq_sums[i] += node_freqs_list[j][i];
    }
  }

  // fill
  probs->resize(shard_num, 0.0);
  for (int i = 0; i < shard_num; ++i) {
    if (freq_sum != 0) {
      (*probs)[i] = 1.0f * freq_sums[i] / freq_sum;
      DXINFO("Server prob[%d] is: %f.", i, (*probs)[i]);
    }
  }

  DXINFO("Done.");
  return true;
}

}  // namespace

bool PostInitCacheStorage(graph_op::DistGSOpResource* resource) {
  auto cache_storage = NewCacheStorage();
  if (!BuildCacheStorage(cache_storage.get())) {
    return false;
  }
  resource->set_cache_storage(std::move(cache_storage));
  return true;
}

bool PostInitServerDistribution(int shard_num,
                                graph_op::DistGSOpResource* resource) {
  int ns_size = 0;
  vec_float_t probs;
  if (!FetchServerDistribution(shard_num, &ns_size, &probs)) {
    return false;
  }
  if (ns_size <= 0) {
    DXINFO("Number of namespace size: %d must be greater than 0.", ns_size);
    return false;
  }
  DXINFO("Number of namespace size is: %d.", ns_size);
  resource->set_ns_size(ns_size);

  auto sampling = NewSampling(&probs, SamplingEnum::ALIAS);
  if (!sampling) {
    return false;
  }
  resource->set_sampling(std::move(sampling));

  return true;
}
}  // namespace embedx
