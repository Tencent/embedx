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
#include <string>

namespace embedx {
namespace instance_name {

// GNN
const std::string X_SRC_NODE_NAME = "__instXsrc_node_";  // NOLINT
const std::string X_SRC_ID_NAME = "__instXsrc_id_";      // NOLINT
const std::string X_DST_NODE_NAME = "__instXdst_node_";  // NOLINT
const std::string X_DST_ID_NAME = "__instXdst_id_";      // NOLINT

const std::string X_NODE_FEATURE_NAME = "__instXnode_feature_";    // NOLINT
const std::string X_NEIGH_FEATURE_NAME = "__instXneigh_feature_";  // NOLINT
const std::string X_ENHANCE_NODE_FEATURE_NAME = "__instXnode_enhance_feature_";
const std::string X_ENHANCE_NODE_SHUFFLED_FEATURE_NAME =
    "__instXnode_enhance_shuffled_feature_";  // NOLINT
const std::string X_NODE_SHUFFLED_FEATURE_NAME =
    "__instXnode_shuffled_feature_";  // NOLINT
const std::string X_NODE_LEFT_MASKED_FEATURE_NAME =
    "__instXnode_left_masked_feature_";  // NOLINT
const std::string X_NODE_RIGHT_MASKED_FEATURE_NAME =
    "__instXnode_right_masked_feature_";  // NOLINT
const std::string X_NEIGH_LEFT_MASKED_FEATURE_NAME =
    "__instXneigh_left_masked_feature_";  // NOLINT
const std::string X_NEIGH_RIGHT_MASKED_FEATURE_NAME =
    "__instXneigh_right_masked_feature_";  // NOLINT

const std::string X_SELF_BLOCK_NAME = "__instXself_block_";    // NOLINT
const std::string X_NEIGH_BLOCK_NAME = "__instXneigh_block_";  // NOLINT
const std::string X_SELF_ENHANCE_BLOCK_NAME = "__instXself_enhance_block_";
const std::string X_NEIGH_ENHANCE_BLOCK_NAME = "__instXneigh_enhance_block_";
const std::string X_SELF_LEFT_DROPPED_BLOCK_NAME =
    "__instXself_left_dropped_block_";  // NOLINT
const std::string X_SELF_RIGHT_DROPPED_BLOCK_NAME =
    "__instXself_right_dropped_block_";  // NOLINT
const std::string X_NEIGH_LEFT_DROPPED_BLOCK_NAME =
    "__instXneigh_left_dropped_block_";  // NOLINT
const std::string X_NEIGH_RIGHT_DROPPED_BLOCK_NAME =
    "__instXneigh_right_dropped_block_";  // NOLINT

const std::string X_NODE_ID_NAME = "__instXnode_id_";            // NOLINT
const std::string X_PREDICT_NODE_NAME = "__instXpredict_node_";  // NOLINT
const std::string X_UNIQUE_NODE_NAME = "__instXunique_node_";    // NOLINT
const std::string Y_UNSUPVISED_NAME = "__instYunsup_";           // NOLINT

// NonGNN
const std::string X_USER_NODE_NAME = "__instXuser_node_";        // NOLINT
const std::string X_USER_ID_NAME = "__instXuser_id_";            // NOLINT
const std::string X_USER_FEATURE_NAME = "__instXuser_feature_";  // NOLINT

const std::string X_ITEM_NODE_NAME = "__instXitem_node_";        // NOLINT
const std::string X_ITEM_ID_NAME = "__instXitem_id_";            // NOLINT
const std::string X_ITEM_FEATURE_NAME = "__instXitem_feature_";  // NOLINT

}  // namespace instance_name
}  // namespace embedx
