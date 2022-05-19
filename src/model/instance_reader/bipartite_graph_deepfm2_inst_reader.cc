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

#include <deepx_core/common/str_util.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/instance/libsvm.h>

#include <vector>

#include "src/io/indexing.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"
#include "src/model/instance_node_name.h"
#include "src/model/instance_reader_util.h"
#include "src/sampler/random_walker_data_types.h"

namespace embedx {
namespace {

const std::string USER_ENCODER_NAME = "USER_ENCODER_NAME";
const std::string ITEM_ENCODER_NAME = "ITEM_ENCODER_NAME";

}  // namespace

/************************************************************************/
/* BipartiteDeepFM2IntanceReader */
/************************************************************************/
class BipartiteDeepFM2InstReader : public EmbedInstanceReader {
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
  // nodes involved in graph encoder
  vec_int_t total_user_nodes_;
  vec_int_t total_item_nodes_;

  std::vector<int> walk_lengths_;
  std::vector<Indexing> user_indexings_;
  std::vector<Indexing> item_indexings_;

 public:
  DEFINE_INSTANCE_READER_LIKE(BipartiteDeepFM2InstReader);

 public:
  bool InitGraphClient(const GraphClient* graph_client) override {
    if (!EmbedInstanceReader::InitGraphClient(graph_client)) {
      return false;
    }

    flow_ = NewNeighborAggregationFlow(graph_client);
    return true;
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

    // user
    total_user_nodes_.clear();
    flow_->MergeTo(user_nodes, &total_user_nodes_);
    flow_->MergeTo(src_nodes, &total_user_nodes_);

    // user encoder
    FillInstance(inst, USER_ENCODER_NAME, total_user_nodes_, num_neighbors_,
                 &user_indexings_);
    flow_->FillNodeOrIndex(inst,
                           instance_name::X_USER_ID_NAME + USER_ENCODER_NAME,
                           user_nodes, &user_indexings_[0]);

    // item
    total_item_nodes_.clear();
    flow_->MergeTo(item_nodes, &total_item_nodes_);
    flow_->MergeTo(dst_nodes, &total_item_nodes_);
    flow_->MergeTo(neg_nodes_list[item_group_], &total_item_nodes_);

    // item encoder
    FillInstance(inst, ITEM_ENCODER_NAME, total_item_nodes_, num_neighbors_,
                 &item_indexings_);
    flow_->FillNodeOrIndex(inst,
                           instance_name::X_ITEM_ID_NAME + ITEM_ENCODER_NAME,
                           item_nodes, &item_indexings_[0]);

    // Fill edge and label
    auto src_indexing_func = [this](int_t node) {
      int index = user_indexings_[0].Get(node);
      DXCHECK(index >= 0);
      return (int_t)index;
    };
    auto dst_indexing_func = [this](int_t node) {
      int index = item_indexings_[0].Get(node);
      DXCHECK(index >= 0);
      return (int_t)index;
    };

    flow_->FillEdgeAndLabel(
        inst, instance_name::X_SRC_ID_NAME, instance_name::X_DST_ID_NAME,
        instance_name::Y_UNSUPVISED_NAME, src_nodes, dst_nodes, neg_nodes_list,
        src_indexing_func, dst_indexing_func);

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

    // Parse user and item nodes from instance
    vec_int_t user_nodes, item_nodes;
    inst_util::ParseUserAndItemFrom(inst, deepx_core::X_NAME, user_group_,
                                    item_group_, &user_nodes, &item_nodes);

    // user encoder
    FillInstance(inst, USER_ENCODER_NAME, user_nodes, num_neighbors_,
                 &user_indexings_);
    flow_->FillNodeOrIndex(inst,
                           instance_name::X_USER_ID_NAME + USER_ENCODER_NAME,
                           user_nodes, &user_indexings_[0]);

    // item encoder
    FillInstance(inst, ITEM_ENCODER_NAME, item_nodes, num_neighbors_,
                 &item_indexings_);
    flow_->FillNodeOrIndex(inst,
                           instance_name::X_ITEM_ID_NAME + ITEM_ENCODER_NAME,
                           item_nodes, &item_indexings_[0]);

    return ret_flag;
  }

  void FillInstance(Instance* inst, const std::string& encoder_name,
                    const vec_int_t& nodes,
                    const std::vector<int>& num_neighbors,
                    std::vector<Indexing>* indexings) {
    // Sample subgraph
    vec_set_t level_nodes;
    vec_map_neigh_t level_neighs;
    flow_->SampleSubGraph(nodes, num_neighbors, &level_nodes, &level_neighs);

    // Fill node feature
    flow_->FillLevelNodeFeature(
        inst, instance_name::X_NODE_FEATURE_NAME + encoder_name, level_nodes);

    // Fill neighbor feature
    if (use_neigh_feat_) {
      flow_->FillLevelNeighFeature(
          inst, instance_name::X_NEIGH_FEATURE_NAME + encoder_name,
          level_nodes);
    }

    // Fill self And neigbor block
    inst_util::CreateIndexings(level_nodes, indexings);
    flow_->FillSelfAndNeighGraphBlock(
        inst, instance_name::X_SELF_BLOCK_NAME + encoder_name,
        instance_name::X_NEIGH_BLOCK_NAME + encoder_name, level_nodes,
        level_neighs, *indexings, false);
  }
};

INSTANCE_READER_REGISTER(BipartiteDeepFM2InstReader,
                         "BipartiteDeepFM2InstReader");
INSTANCE_READER_REGISTER(BipartiteDeepFM2InstReader, "bipartite_graph_deepfm2");
INSTANCE_READER_REGISTER(BipartiteDeepFM2InstReader, "bipartite_graph_dfm2");

}  // namespace embedx
