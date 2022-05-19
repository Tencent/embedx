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

#pragma once
#include <deepx_core/graph/tensor_map.h>  // Instance

#include <memory>  // std::unique_ptr
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client.h"
#include "src/io/indexing.h"

namespace embedx {

using ::deepx_core::Instance;

class RandomWalkFlow : public deepx_core::DataType {
 private:
  const GraphClient& graph_client_;

 public:
  explicit RandomWalkFlow(const GraphClient* graph_client)
      : graph_client_(*graph_client) {}
  virtual ~RandomWalkFlow() = default;

 public:
  void FillNodeOrIndex(Instance* inst, const std::string& id_name,
                       const csr_t& nodes, const Indexing* indexing) const;
  void FillNodeOrIndex(Instance* inst, const std::string& id_name,
                       const vec_int_t& nodes, const Indexing* indexing) const;
  void FillNodeFeature(Instance* inst, const std::string& name,
                       const vec_int_t& nodes, bool add_self) const;
  void FillEdgeAndLabel(Instance* inst, const std::string& src_name,
                        const std::string& dst_name, const std::string& y_name,
                        const vec_int_t& src_nodes,
                        const std::vector<vec_int_t>& dst_nodes_list,
                        const std::vector<vec_int_t>& neg_nodes_list) const;
};

std::unique_ptr<RandomWalkFlow> NewRandomWalkFlow(
    const GraphClient* graph_client);

}  // namespace embedx
