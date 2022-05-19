// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/deep/client/deep_client.h"

#include <utility>

#include "src/deep/client/deep_client_impl.h"

namespace embedx {

DeepClient::DeepClient(std::unique_ptr<DeepClientImpl>&& impl) {
  impl_ = std::move(impl);
}

DeepClient::~DeepClient() {}

bool DeepClient::SharedSampleNegative(
    int count, const vec_int_t& nodes, const vec_int_t& excluded_nodes,
    std::vector<vec_int_t>* sampled_nodes_list) const {
  return impl_->SharedSampleNegative(count, nodes, excluded_nodes,
                                     sampled_nodes_list);
}

bool DeepClient::LookupItemFeature(const vec_int_t& items,
                                   std::vector<vec_pair_t>* item_feats) const {
  return impl_->LookupItemFeature(items, item_feats);
}

bool DeepClient::SampleInstance(int count, vec_int_t* insts,
                                std::vector<vecl_t>* vec_labels_list) const {
  return impl_->SampleInstance(count, insts, vec_labels_list);
}

std::unique_ptr<DeepClient> NewDeepClient(const DeepConfig& config,
                                          DeepClientEnum type) {
  std::unique_ptr<DeepClient> deep_client;
  switch (type) {
    case DeepClientEnum::LOCAL:
      deep_client.reset(new DeepClient(NewLocalDeepClientImpl(config)));
      break;
    default:
      DXERROR("Need type: LOCAL(0), got type: %d.", (int)type);
      break;
  }

  return deep_client;
}

}  // namespace embedx
