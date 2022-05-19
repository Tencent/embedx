// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yuanhang Zou (yuanhang.nju@gmail.com)
//

#pragma once
#include <deepx_core/common/group_config.h>
#include <deepx_core/graph/graph_module_creator.h>
#include <deepx_core/graph/graph_node.h>
#include <deepx_core/graph/variable_scope.h>
#include <deepx_core/tensor/shape.h>
#include <deepx_core/tensor/tensor_type.h>

#include <string>
#include <vector>

#include "src/common/data_types.h"

namespace embedx {

using ::deepx_core::BATCH_PLACEHOLDER;
using ::deepx_core::GetVariable;
using ::deepx_core::GraphNode;
using ::deepx_core::GroupConfigItem3;
using ::deepx_core::InstanceNode;
using ::deepx_core::Shape;
using ::deepx_core::TENSOR_INITIALIZER_TYPE_RAND;
using ::deepx_core::TENSOR_INITIALIZER_TYPE_RANDN;
using ::deepx_core::TENSOR_INITIALIZER_TYPE_ZEROS;
using ::deepx_core::TENSOR_TYPE_CSR;
using ::deepx_core::TENSOR_TYPE_SRM;
using ::deepx_core::TENSOR_TYPE_TSR;

/************************************************************************/
/* instance input functions */
/************************************************************************/
GraphNode* GetXInput(const std::string& name);

std::vector<GraphNode*> GetXBlockInputs(const std::string& name, int depth);

GraphNode* GetYUnsup(const std::string& name, int label_size);

/************************************************************************/
/* embedding lookup functions */
/************************************************************************/
// rand initialization (-1 / log(col), 1 / log(col))
GraphNode* XInputEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                 const GroupConfigItem3& item, int sparse);

// randn initialization (0, 1e-3) for graph-ctr models
GraphNode* XInputEmbeddingLookup2(const std::string& prefix, GraphNode* X,
                                  const GroupConfigItem3& item, int sparse);

// rand initiialization (-1 / col, 1 / col)
GraphNode* XInputEmbeddingLookup3(const std::string& prefix, GraphNode* X,
                                  const GroupConfigItem3& item, int sparse);

GraphNode* XOutputEmbeddingLookup(const std::string& prefix, GraphNode* X,
                                  const GroupConfigItem3& item, int sparse);

/************************************************************************/
/* group embedding lookup functions */
/************************************************************************/
// get weight for each feature group
GraphNode* XNodeGroupWeightLookup(const std::string& prefix, GraphNode* X,
                                  const std::vector<GroupConfigItem3>& items,
                                  int sparse);

// rand initialization (-1 / log(col), 1 / log(col))
GraphNode* XInputGroupEmbeddingLookup(
    const std::string& prefix, GraphNode* X,
    const std::vector<GroupConfigItem3>& items, int sparse);

// rand initiialization (-1 / col, 1 / col)
GraphNode* XInputGroupEmbeddingLookup2(
    const std::string& prefix, GraphNode* X,
    const std::vector<GroupConfigItem3>& items, int sparse);

GraphNode* XOutputGroupEmbeddingLookup(
    const std::string& prefix, GraphNode* X,
    const std::vector<GroupConfigItem3>& items, int sparse);

/************************************************************************/
/* BatchLookupAndDot functions */
/************************************************************************/
GraphNode* BatchLookupAndDot(const std::string& prefix, GraphNode* Xin,
                             GraphNode* Xout, const GroupConfigItem3& item,
                             int sparse);

/************************************************************************/
/* eges encoder functions */
/************************************************************************/
GraphNode* EgesEncoder(const std::string& prefix, GraphNode* Xsrc_feat,
                       GraphNode* Xsrc_node,
                       const std::vector<GroupConfigItem3>& items, bool sparse);

/************************************************************************/
/* sage encoder functions */
/************************************************************************/
GraphNode* SparseSageEncoder(const std::string& prefix, GraphNode* self_feat,
                             GraphNode* neigh_feat,
                             const std::vector<GroupConfigItem3>& items,
                             bool sparse, bool is_act, double alpha);

GraphNode* DenseSageEncoder(const std::string& prefix, GraphNode* hidden,
                            GraphNode* self_block, GraphNode* neigh_block,
                            int dim, bool is_act, double alpha);

GraphNode* GraphSageEncoder(const std::string& encoder_name,
                            const std::vector<GroupConfigItem3>& items,
                            int depth, bool use_neigh_feat, bool sparse,
                            double relu_alpha, int dim);

GraphNode* GraphSageEncoder(const std::string& encoder_name,
                            const std::vector<GroupConfigItem3>& items,
                            GraphNode* Xnode_feat, GraphNode* Xneigh_feat,
                            const std::vector<GraphNode*> self_blocks,
                            const std::vector<GraphNode*> neigh_blocks,
                            bool sparse, double relu_alpha, int dim);

GraphNode* HeterGraphSageEncoder(const id_name_t& id_2_name,
                                 const std::string& prefix,
                                 const std::vector<GroupConfigItem3>& items,
                                 int depth, bool use_neigh_feat, bool sparse,
                                 double relu_alpha, int dim);

GraphNode* PinsageEncoder(const std::string& prefix, GraphNode* hidden,
                          GraphNode* self_block, GraphNode* neigh_block,
                          int dim, double alpha);

GraphNode* PinsageRootEncoder(const std::string& prefix, GraphNode* hidden,
                              int dim, double alpha);
GraphNode* SageEncoder(const std::string& prefix, GraphNode* hidden,
                       std::vector<GraphNode*> self_blocks,
                       std::vector<GraphNode*> neigh_blocks,
                       int sage_encoder_type, int depth, int sage_dim,
                       double relu_alpha);

/************************************************************************/
/* target funtions */
/************************************************************************/
std::vector<GraphNode*> BinaryClassificationTarget(const std::string& prefix,
                                                   GraphNode* X, GraphNode* Y,
                                                   int has_w);
std::vector<GraphNode*> BinaryClassificationTarget(GraphNode* X, GraphNode* Y,
                                                   int has_w);

std::vector<GraphNode*> MultiClassificationTarget(const std::string& prefix,
                                                  GraphNode* X, int has_w);
std::vector<GraphNode*> MultiClassificationTarget(GraphNode* X, int has_w);

std::vector<GraphNode*> MultiLabelClassificationTarget(
    const std::string& prefix, GraphNode* X, int has_w);

std::vector<GraphNode*> MultiLabelClassificationTarget(GraphNode* X, int has_w);

}  // namespace embedx
