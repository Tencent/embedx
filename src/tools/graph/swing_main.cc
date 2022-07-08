// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Chunchen Su (chunchen.scut@gmail.com)
//

#include <deepx_core/common/misc.h>
#include <deepx_core/dx_log.h>
#include <gflags/gflags.h>

#include <memory>  // std::unique_ptr

#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/io/line_parser.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/graph/main_util.h"
#include "src/tools/graph/swing/swing.h"

// swing_main
DEFINE_string(in, "", "Input dir/file of item node data.");
DEFINE_double(swing_alpha, 1.0, "Swing alpha.");
DEFINE_int32(swing_cache_thld, 10,
             "If number of user contexts(item) less than swing_cache_thld, "
             "cache item intersection.");
DEFINE_int32(swing_sample_thld, 5000,
             "If number of item contexts(user) greater than swing_sample_thld, "
             "sampling item contexts.");

namespace embedx {
namespace {

class SwingMain : public MainUtil {
 private:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig graph_config_;
  SwingConfig swing_config_;

  int batch_node_;
  std::string entry_flag_;

 public:
  ~SwingMain() override = default;

  bool Init() override {
    if (FLAGS_dist) {
      graph_config_.set_ip_ports(FLAGS_gs_addrs);
    } else {
      graph_config_.set_node_graph(FLAGS_node_graph);
      graph_config_.set_thread_num(FLAGS_gs_thread_num);
    }

    graph_client_ = NewGraphClient(graph_config_, (GraphClientEnum)FLAGS_dist);
    if (!graph_client_) {
      return false;
    }

    // used by RunEntry
    swing_config_.alpha = FLAGS_swing_alpha;
    swing_config_.cache_thld = FLAGS_swing_cache_thld;
    swing_config_.sample_thld = FLAGS_swing_sample_thld;

    batch_node_ = FLAGS_batch_node;
    entry_flag_ = FLAGS_dist ? "Worker" : "Thread";
    return true;
  }

 private:
  const char* task_name() const noexcept override { return "Swing"; }

 private:
  bool RunEntry(int entry_id, const vec_str_t& files,
                const std::string& out) override {
    DXINFO("%s id: %d is processing ...", entry_flag_.c_str(), entry_id);

    Swing swing(graph_client_.get(), swing_config_);
    LineParser line_parser;
    deepx_core::AutoOutputFileStream ofs;
    std::ostringstream oss;

    for (auto& file : files) {
      DXINFO("Processed file: %s.", file.c_str());
      if (!line_parser.Open(file)) {
        return false;
      }

      auto out_file = deepx_core::GetOutputPredictFile(out, file);
      if (!ofs.Open(out_file)) {
        return false;
      }

      // swing
      std::vector<AdjValue> values;
      vec_int_t item_nodes;
      std::vector<vec_pair_t> item_contexts;
      std::vector<vec_pair_t> item_scores;
      while (line_parser.NextBatch<>(batch_node_, &values)) {
        item_nodes = Collect<AdjValue, int_t>(values, &AdjValue::node);
        item_contexts = Collect<AdjValue, vec_pair_t>(values, &AdjValue::pairs);
        item_scores.clear();
        if (!swing.ComputeItemScore(item_nodes, item_contexts, &item_scores)) {
          return false;
        }

        if (!MainUtil::DumpText(ofs, oss, item_nodes, item_scores)) {
          return false;
        }
      }
    }

    return true;
  }
};

/************************************************************************/
/* main */
/************************************************************************/
void CheckFlags() {
  if (FLAGS_dist) {
    DXCHECK(!FLAGS_gs_addrs.empty());
    DXCHECK(FLAGS_gs_worker_num > 0);
    DXCHECK(FLAGS_gs_worker_id >= 0);
  } else {
    DXCHECK(!FLAGS_node_graph.empty());
    DXCHECK(FLAGS_gs_thread_num > 0);
  }

  DXCHECK(FLAGS_batch_node > 0);
  DXCHECK(!FLAGS_in.empty());
  DXCHECK(!FLAGS_out.empty());
  DXCHECK(FLAGS_swing_alpha > 0);
  DXCHECK(FLAGS_swing_cache_thld >= 0);
  DXCHECK(FLAGS_swing_sample_thld >= 0);
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  std::unique_ptr<MainUtil> main(new SwingMain);
  if (!main->Init()) {
    return -1;
  }

  if (FLAGS_dist) {
    if (!main->RunSingleWorker(FLAGS_in, FLAGS_out, FLAGS_gs_worker_num,
                               FLAGS_gs_worker_id)) {
      return -1;
    }
  } else {
    if (!main->RunMultiThread(FLAGS_in, FLAGS_out, FLAGS_gs_thread_num)) {
      return -1;
    }
  }

  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
