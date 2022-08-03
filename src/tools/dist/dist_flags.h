// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#pragma once
#include <deepx_core/graph/shard.h>
#include <deepx_core/ps/tcp_connection.h>
#include <gflags/gflags.h>

#include <vector>

// train & predict
DECLARE_string(sub_command);
DECLARE_string(role);
DECLARE_string(cs_addr);
DECLARE_string(ps_addrs);
DECLARE_int32(ps_id);
DECLARE_int32(ps_thread_num);

DECLARE_bool(gnn_model);
DECLARE_bool(deep_model);
DECLARE_string(model);
DECLARE_string(model_config);
DECLARE_string(instance_reader);
DECLARE_string(instance_reader_config);
DECLARE_string(optimizer);
DECLARE_string(optimizer_config);
DECLARE_int32(epoch);
DECLARE_int32(batch);
DECLARE_string(in_model);
DECLARE_string(warmup_model);
DECLARE_string(in);
DECLARE_string(pretrain_path);
DECLARE_string(item_feature);
DECLARE_string(inst_file);
DECLARE_string(freq_file);
DECLARE_bool(shuffle);
DECLARE_bool(ts_enable);
DECLARE_uint64(ts_now);
DECLARE_uint64(ts_expire_threshold);
DECLARE_uint64(freq_filter_threshold);
DECLARE_int32(verbose);
DECLARE_int32(seed);
DECLARE_int32(target_type);
DECLARE_bool(out_model_remove_zeros);
DECLARE_string(out_model);
DECLARE_string(out_model_text);
DECLARE_string(out_model_fkv);
DECLARE_int32(out_model_fkv_pb_version);
DECLARE_string(out_predict);

namespace embedx {

extern int FLAGS_is_train;
extern deepx_core::TcpEndpoint FLAGS_cs_endpoint;
extern std::vector<deepx_core::TcpEndpoint> FLAGS_ps_endpoints;
extern int FLAGS_ps_size;
extern deepx_core::Shard FLAGS_shard;

void CheckFlags();

}  // namespace embedx
