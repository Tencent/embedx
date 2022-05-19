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

#include "src/tools/model_util.h"

#include <deepx_core/common/str_util.h>
#include <deepx_core/tensor/tensor_type.h>

#include <cinttypes>  // PRIu64
#include <limits>

#include "src/io/io_util.h"

namespace embedx {

bool ModelUtil::LoadPretrainParam(const std::string& param_path,
                                  const deepx_core::Shard& shard,
                                  deepx_core::ModelShard* model_shard) {
  if (!io_util::ListFile(param_path, &files_)) {
    DXERROR("Failed to init files from param_path: %s.", param_path.c_str());
    return false;
  }
  DXINFO("Got %d files.", (int)files_.size());

  for (const std::string& file : files_) {
    if (!ifs_.Open(file)) {
      DXERROR("Failed to open file: %s.", file.c_str());
      return false;
    }

    while (GetLine(ifs_, line_)) {
      iss_.clear();
      iss_.str(line_);
      if (!(iss_ >> tensor_name_ >> tensor_type_)) {
        DXERROR("Invalid line: %s", line_.c_str());
        return false;
      }
      const auto* node = graph_.find_node(tensor_name_);
      if (node == nullptr) {
        DXERROR("Tensor: %s doesn't exist.", tensor_name_.c_str());
        return false;
      }
      if (node->tensor_type() != tensor_type_) {
        DXERROR("Expected tensor type: %d, got: %d.", node->tensor_type(),
                tensor_type_);
        return false;
      }

      if (node->tensor_type() == deepx_core::TENSOR_TYPE_SRM) {
        if (!(iss_ >> feature_id_ >> str_vals_)) {
          DXERROR("Invalid line: %s.", line_.c_str());
          return false;
        }
        if (shard.HasSRM(model_shard->shard_id(), feature_id_)) {
          if (LoadSRMRow(model_shard, str_vals_)) {
            valid_tensor_.emplace(tensor_name_);
          } else {
            return false;
          }
        }

      } else if (node->tensor_type() == deepx_core::TENSOR_TYPE_TSR) {
        if (!(iss_ >> str_vals_)) {
          DXERROR("Invalid line: %s.", line_.c_str());
          return false;
        }

        if (shard.HasTSR(model_shard->shard_id(), tensor_name_)) {
          if (LoadTSR(model_shard, str_vals_)) {
            valid_tensor_.emplace(tensor_name_);
          } else {
            return false;
          }
        }
      } else {
        DXERROR("Tensor name: %s, with unsupported tensor type.",
                tensor_name_.c_str());
        return false;
      }
    }
  }

  for (const auto& entry : valid_tensor_) {
    DXINFO("Tensor name: %s is successfully loaded.", entry.c_str());
  }
  return true;
}

bool ModelUtil::LoadSRMRow(deepx_core::ModelShard* model_shard,
                           const std::string& str_vals) {
  if (!deepx_core::Split<float_t>(str_vals, ",", &float_vals_)) {
    DXERROR("Wrong format of str_embed: %s.", str_vals.c_str());
    return false;
  }
  const auto& it = model_shard->mutable_param()->find(tensor_name_);
  if (it != model_shard->mutable_param()->end()) {
    auto& srm = it->second.unsafe_to_ref<srm_t>();
    if (srm.col() != (int)float_vals_.size()) {
      DXERROR("Tensor name: %s, tensor_type: %s ,feature_id: %" PRIu64
              " has wrong col size.",
              tensor_name_.c_str(), "srm", feature_id_);
      return false;
    }
    srm.assign(feature_id_, &float_vals_[0]);
  }
  return true;
}

bool ModelUtil::LoadTSR(deepx_core::ModelShard* model_shard,
                        const std::string& str_vals) {
  if (!deepx_core::Split<float_t>(str_vals, ",", &float_vals_)) {
    DXERROR("Wrong format for str_tsr_vals: %s.", str_vals.c_str());
    return false;
  }
  const auto& it = model_shard->mutable_param()->find(tensor_name_);
  if (it != model_shard->mutable_param()->end()) {
    auto& tsr = it->second.unsafe_to_ref<tsr_t>();
    if (float_vals_.size() > (size_t)std::numeric_limits<int>::max()) {
      DXERROR("Input tensor size exceeds max value of int.");
      return false;
    }
    if (tsr.total_dim() != (int)float_vals_.size()) {
      DXERROR("Total dim not equal, %d != %d.", tsr.total_dim(),
              (int)float_vals_.size());
      return false;
    }
    tsr.set_data(float_vals_);
  }
  return true;
}

}  // namespace embedx
