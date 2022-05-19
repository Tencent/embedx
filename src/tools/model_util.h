// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#pragma once
#include <deepx_core/common/stream.h>
#include <deepx_core/graph/graph.h>
#include <deepx_core/graph/model_shard.h>
#include <deepx_core/graph/shard.h>
#include <deepx_core/tensor/data_type.h>

#include <sstream>  // std::istringstream
#include <string>
#include <unordered_set>
#include <vector>

namespace embedx {

class ModelUtil : public deepx_core::DataType {
 private:
  const deepx_core::Graph& graph_;

  std::vector<std::string> files_;
  std::istringstream iss_;
  deepx_core::AutoInputFileStream ifs_;

  std::string line_;
  std::string tensor_name_;
  int tensor_type_ = 0;
  int_t feature_id_ = 0;
  std::string str_vals_;
  std::vector<float_t> float_vals_;
  std::unordered_set<std::string> valid_tensor_;

 public:
  explicit ModelUtil(const deepx_core::Graph* graph) : graph_(*graph) {}
  ~ModelUtil() = default;

 public:
  // support load param from text file, only support srm and tsr
  // line format:
  // one line, one tsr
  // tsr -> "tensor_name tensor_type v1,v2,...,vn"
  // one line, one srm row
  // srm -> "tensor_name tensor_type feature_id v1,v1,...,vm"
  bool LoadPretrainParam(const std::string& param_path,
                         const deepx_core::Shard& shard,
                         deepx_core::ModelShard* model_shard);

 private:
  bool LoadTSR(deepx_core::ModelShard* model_shard,
               const std::string& str_vals);
  bool LoadSRMRow(deepx_core::ModelShard* model_shard,
                  const std::string& str_vals);
};

}  // namespace embedx
