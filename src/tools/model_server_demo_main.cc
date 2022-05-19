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

#include <deepx_core/dx_log.h>
#include <gflags/gflags.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "embedx/model_server.h"

DEFINE_int32(target_type, 2, "Target type for inference.");
DEFINE_string(in, "", "Input file.");
DEFINE_string(in_graph, "", "Input graph file.");
DEFINE_string(in_model, "", "Input model param file.");

namespace embedx {
namespace {

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  ModelServer model_server;
  std::string line;
  std::istringstream is;
  uint64_t feature_id;
  float feature_value;
  char colon;
  std::vector<features_t> batch_features;
  std::vector<std::vector<float>> batch_probs;

  // The target type must be set before loading the model.
  // 1 for classification prob
  // 2 for user embedding
  // 3 for item embedding
  DXCHECK_THROW(FLAGS_target_type == 1 || FLAGS_target_type == 2 ||
                FLAGS_target_type == 3);
  model_server.set_target_type(FLAGS_target_type);

  if (!FLAGS_in.empty()) {
    DXCHECK_THROW(model_server.Load(FLAGS_in));
  } else {
    DXCHECK_THROW(!FLAGS_in_graph.empty());
    DXCHECK_THROW(!FLAGS_in_model.empty());
    DXCHECK_THROW(model_server.LoadGraph(FLAGS_in_graph));
    DXCHECK_THROW(model_server.LoadModel(FLAGS_in_model));
  }

  auto op_context = model_server.NewOpContext();
  DXCHECK_THROW(op_context);

  while (std::getline(std::cin, line)) {
    is.clear();
    is.str(line);
    batch_features.resize(1);
    auto& features = batch_features[0];
    features.clear();
    while (is >> feature_id >> colon >> feature_value) {
      features.emplace_back(feature_id, feature_value);
    }
    DXCHECK_THROW(model_server.BatchPredictUserEmbedding(
        op_context.get(), batch_features, &batch_probs));
    auto& probs = batch_probs[0];
    for (size_t i = 0; i < probs.size(); ++i) {
      std::cout << probs[i];
      if (i != probs.size() - 1) {
        std::cout << " ";
      }
    }
    std::cout << "\n";
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
