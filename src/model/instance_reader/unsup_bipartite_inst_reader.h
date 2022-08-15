// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//         Chuan Cheng (chengchuancoder@gmail.com)
//

#include "src/io/indexing_wrapper.h"
#include "src/model/data_flow/neighbor_aggregation_flow.h"
#include "src/model/embed_instance_reader.h"

namespace embedx {

class UnsupBipartiteInstReader : public EmbedInstanceReader {
 protected:
  bool is_train_ = true;
  int num_neg_ = 5;
  std::vector<int> num_neighbors_;
  bool use_neigh_feat_ = false;
  uint16_t user_ns_id_ = 0;
  uint16_t item_ns_id_ = 1;

 protected:
  std::unique_ptr<NeighborAggregationFlow> flow_;
  vec_int_t src_nodes_;
  vec_int_t dst_nodes_;
  std::vector<vec_int_t> neg_nodes_list_;
  std::unique_ptr<IndexingWrapper> indexing_wrapper_;

 public:
  bool InitGraphClient(const GraphClient* graph_client) override;
  void PostInit(const std::string& node_config) override;
  bool GetBatch(Instance* inst) override;

 protected:
  bool InitConfigKV(const std::string& k, const std::string& v) override;

 protected:
  bool GetTrainBatch(Instance* inst);
  bool GetPredictBatch(Instance* inst);
  void FillIndex(Instance* inst, const std::string& name,
                 const vec_int_t& nodes) const;
  void FillInstance(Instance* inst, const std::string& encoder_name,
                    const vec_int_t& nodes, uint16_t ns_id,
                    const std::vector<int>& num_neighbors);
};

}  // namespace embedx
