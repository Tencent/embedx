// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhenting Yu (zhenting.yu@gmail.com)
//         Yuanhang Zou (yuanhang.nju@gmail.com)
//

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/instance/libsvm.h>

#include <vector>

#include "src/io/indexing_wrapper.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {
namespace {

const std::string USER_ENCODER_NAME = "USER_ENCODER_NAME";

}  // namespace

/************************************************************************/
/* GraphDeepFM2IntanceReader */
/************************************************************************/
class GraphDeepFM2InstReader : public EmbedInstanceReader {
 private:
  bool is_train_ = true;
  int num_neg_ = 5;
  std::vector<int> num_neighbors_;
  bool use_neigh_feat_ = false;
  WalkerInfo walker_info_;
  int walk_length_ = 1;
  int window_size_ = 1;
  uint16_t user_group_ = 0;
  uint16_t item_group_ = 0;

 private:
  std::unique_ptr<NeighborAggregationFlow> flow_;

  // deepfm2 feature
  csr_t* X_ = nullptr;

  vec_int_t merged_nodes_;
  vec_set_t level_nodes_;
  vec_map_neigh_t level_neighbors_;

  uint16_t ns_id_;
  std::unique_ptr<IndexingWrapper> indexing_wrapper_;

 public:
  DEFINE_INSTANCE_READER_LIKE(GraphDeepFM2InstReader);

 public:
  bool InitGraphClient(const GraphClient* graph_client) override {
    if (!EmbedInstanceReader::InitGraphClient(graph_client)) {
      return false;
    }

    flow_ = NewNeighborAggregationFlow(graph_client);
    return true;
  }

  void PostInit(const std::string& /*node_config*/) override {
    ns_id_ = 0;
    indexing_wrapper_ = IndexingWrapper::Create("");
  }

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override {
    if (InstanceReaderImpl::InitConfigKV(k, v)) {
    } else if (k == "is_train") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      is_train_ = val;
    } else if (k == "num_neg") {
      num_neg_ = std::stoi(v);
      DXCHECK(num_neg_ > 0);
    } else if (k == "num_neighbors") {
      DXCHECK(deepx_core::Split<int>(v, ",", &num_neighbors_));
      DXCHECK(!num_neighbors_.empty());
    } else if (k == "use_neigh_feat") {
      auto val = std::stoi(v);
      DXCHECK(val == 1 || val == 0);
      use_neigh_feat_ = val;
    } else if (k == "walk_length") {
      walk_length_ = std::stoi(v);
      DXCHECK(walk_length_ > 1);
    } else if (k == "window_size") {
      window_size_ = std::stoi(v);
      DXCHECK(window_size_ > 1);
    } else if (k == "user_group_id") {
      user_group_ = std::stoi(v);
    } else if (k == "item_group_id") {
      item_group_ = std::stoi(v);
    } else {
      DXERROR("Unexpected config: %s = %s.", k.c_str(), v.c_str());
      return false;
    }

    DXINFO("Instance reader argument: %s = %s.", k.c_str(), v.c_str());
    return true;
  }

 protected:
  void InitXBatch(Instance* inst) override {
    X_ = &inst->get_or_insert<csr_t>(deepx_core::X_NAME);
    X_->clear();
    X_->reserve(batch_);
  }

  bool ParseLine() override {
    deepx_core::LibsvmInstanceReaderHelper<float_t, int_t> helper(line_);
    return helper.Parse(X_, Y_, W_, uuid_);
  }

 public:
  bool GetBatch(Instance* inst) override {
    return is_train_ ? GetTrainBatch(inst) : GetPredictBatch(inst);
  }

  bool Open(const std::string& file) override {
    return InstanceReaderImpl::Open(file);
  }

  /************************************************************************/
  /* Read batch data from file for training */
  /************************************************************************/
  bool GetTrainBatch(Instance* inst) {
    // ctr data
    bool ret_flag = true;
    if (!InstanceReaderImpl::GetBatch(inst)) {
      if (inst->batch() == 0) {
        return false;
      }
      ret_flag = false;
    }

    // Parse user and item nodes from instance
    vec_int_t user_nodes, item_nodes;
    inst_util::ParseUserAndItemFrom(inst, deepx_core::X_NAME, user_group_,
                                    item_group_, &user_nodes, &item_nodes);

    // Generate sequence from user and item nodes
    std::vector<vec_int_t> seqs;
    std::vector<int> walk_lengths(2 * batch_, walk_length_);
    inst_util::GenerateSeqFrom(graph_client_, user_nodes, item_nodes,
                               walk_lengths, walker_info_, &seqs);

    // Parse sequence to src and dst nodes
    vec_int_t src_nodes, dst_nodes;
    std::vector<vec_int_t> neg_nodes_list;
    inst_util::ParseSeqTo(seqs, window_size_, user_group_, &src_nodes,
                          &dst_nodes);
    DXCHECK(graph_client_->SharedSampleNegative(num_neg_, dst_nodes, dst_nodes,
                                                &neg_nodes_list));

    // merge nodes to avoid repeated construction of node computation graph.
    merged_nodes_.clear();
    flow_->MergeTo(user_nodes, &merged_nodes_);
    flow_->MergeTo(src_nodes, &merged_nodes_);

    flow_->SampleSubGraph(merged_nodes_, num_neighbors_, &level_nodes_,
                          &level_neighbors_);

    // Fill Instance
    // 1. Fill node feature
    flow_->FillLevelNodeFeature(inst, instance_name::X_NODE_FEATURE_NAME,
                                level_nodes_);

    // 2. Fill self And neigbor block
    indexing_wrapper_->Clear();
    indexing_wrapper_->BuildFrom(level_nodes_);
    const auto& indexings = indexing_wrapper_->subgraph_indexing(ns_id_);
    flow_->FillSelfAndNeighGraphBlock(inst, instance_name::X_SELF_BLOCK_NAME,
                                      instance_name::X_NEIGH_BLOCK_NAME,
                                      level_nodes_, level_neighbors_, indexings,
                                      false);

    // 3. Fill index
    auto user_id_name = instance_name::X_NODE_ID_NAME + USER_ENCODER_NAME;
    flow_->FillNodeOrIndex(inst, user_id_name, user_nodes, &indexings[0]);

    auto user_node_name = instance_name::X_USER_NODE_NAME;
    flow_->FillNodeOrIndex(inst, user_node_name, user_nodes, nullptr);

    // 4. Fill edge and label
    auto src_index_func = [this](int_t node) {
      return indexing_wrapper_->GlobalGet(node);
    };

    auto dst_index_func = [](int_t node) { return node; };
    flow_->FillEdgeAndLabel(
        inst, instance_name::X_SRC_ID_NAME, instance_name::X_DST_NODE_NAME,
        instance_name::Y_UNSUPVISED_NAME, src_nodes, dst_nodes, neg_nodes_list,
        src_index_func, dst_index_func);

    return ret_flag;
  }

  /************************************************************************/
  /* Read batch data from file for predicting */
  /************************************************************************/
  bool GetPredictBatch(Instance* inst) {
    // ctr data
    bool ret_flag = true;
    if (!InstanceReaderImpl::GetBatch(inst)) {
      if (inst->batch() == 0) {
        return false;
      }
      ret_flag = false;
    }

    return ret_flag;
  }
};

INSTANCE_READER_REGISTER(GraphDeepFM2InstReader, "GraphDeepFM2InstReader");
INSTANCE_READER_REGISTER(GraphDeepFM2InstReader, "graph_deepfm2");
INSTANCE_READER_REGISTER(GraphDeepFM2InstReader, "graph_dfm2");

}  // namespace embedx
