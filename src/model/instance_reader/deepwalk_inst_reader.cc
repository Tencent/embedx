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

#include <deepx_core/dx_log.h>

#include <vector>

#include "src/io/value.h"
#include "src/model/data_flow/random_walk_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"

namespace embedx {

class DeepwalkInstReader : public EmbedInstanceReader {
 private:
  std::unique_ptr<RandomWalkFlow> flow_;

  int num_neg_ = 5;
  int window_size_ = 10;
  bool train_ = true;

  std::vector<vec_int_t> seqs_;
  std::vector<vec_int_t> context_nodes_list_;
  std::vector<vec_int_t> neg_nodes_list_;

 public:
  DEFINE_INSTANCE_READER_LIKE(DeepwalkInstReader);

 public:
  bool InitGraphClient(const GraphClient* graph_client) override {
    if (!EmbedInstanceReader::InitGraphClient(graph_client)) {
      return false;
    }

    flow_ = NewRandomWalkFlow(graph_client);
    return true;
  }

  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      if (num_neg_ < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "window_size") {
      window_size_ = std::stoi(v);
      if (window_size_ <= 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
    } else if (k == "is_train") {
      auto val = std::stoi(v);
      if (val < 0) {
        DXERROR("Invalid %s: %s.", k.c_str(), v.c_str());
        return false;
      }
      train_ = val == 0 ? false : true;
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Instance reader argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

 public:
  bool GetBatch(Instance* inst) override {
    return train_ ? GetTrainBatch(inst) : GetPredictBatch(inst);
  }

  /************************************************************************/
  /* Read batch data from file for training */
  /************************************************************************/
  bool GetTrainBatch(Instance* inst) {
    std::vector<SeqValue> values;
    if (!NextInstanceBatch<SeqValue>(inst, 1, &values)) {
      return false;
    }
    seqs_ = Collect<SeqValue, vec_int_t>(values, &SeqValue::nodes);
    inst_util::CheckLengthValid(seqs_);

    auto& seq = seqs_[0];
    DXCHECK(graph_client_->IndepSampleNegative(num_neg_, seq, seq,
                                               &neg_nodes_list_));

    // Generate context and negative nodes based on sequence data.
    inst_util::GenerateContext(window_size_, window_size_, seq,
                               &context_nodes_list_);

    // 1. Fill instance(edge and label)
    flow_->FillEdgeAndLabel(inst, instance_name::X_SRC_NODE_NAME,
                            instance_name::X_DST_NODE_NAME, deepx_core::Y_NAME,
                            seq, context_nodes_list_, neg_nodes_list_);

    inst->set_batch(1);
    return true;
  }

  /************************************************************************/
  /* Read batch data from file for predicting */
  /************************************************************************/
  bool GetPredictBatch(Instance* inst) {
    std::vector<NodeValue> values;
    if (!NextInstanceBatch<NodeValue>(inst, batch_, &values)) {
      return false;
    }
    auto* predict_node_ptr =
        &inst->get_or_insert<vec_int_t>(instance_name::X_PREDICT_NODE_NAME);
    *predict_node_ptr = Collect<NodeValue, int_t>(values, &NodeValue::node);

    // Fill Instance
    // 1. Fill node
    flow_->FillNodeOrIndex(inst, instance_name::X_SRC_NODE_NAME,
                           *predict_node_ptr, nullptr);

    inst->set_batch(predict_node_ptr->size());
    return true;
  }
};

INSTANCE_READER_REGISTER(DeepwalkInstReader, "DeepWalkInstReader");
INSTANCE_READER_REGISTER(DeepwalkInstReader, "deepwalk");

}  // namespace embedx
