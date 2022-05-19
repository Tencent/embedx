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

#include <deepx_core/common/misc.h>
#include <deepx_core/common/str_util.h>
#include <gflags/gflags.h>

#include <memory>  // std::unique_ptr

#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/io/line_parser.h"
#include "src/sampler/random_walker_data_types.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/graph/main_util.h"

// random_walk_main
DEFINE_int32(walk_length, 10, "Length of random walk sequence.");
DEFINE_int32(dump_type, 0,
             "When dump_type is 0 output sequence, otherwise output edge.");
DEFINE_int32(epoch, 1, "Number of epochs.");

namespace embedx {
namespace {

class RandomWalkerMain : public MainUtil {
 private:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig graph_config_;

  int epoch_;
  int batch_node_;
  int walk_length_;
  int dump_type_;
  std::string entry_flag_;

 public:
  ~RandomWalkerMain() override = default;

  bool Init() override {
    if (FLAGS_dist) {
      graph_config_.set_ip_ports(FLAGS_gs_addrs);
    } else {
      graph_config_.set_node_graph(FLAGS_node_graph);
      graph_config_.set_node_config(FLAGS_node_config);
      graph_config_.set_random_walker_type(FLAGS_random_walker_type);
      graph_config_.set_thread_num(FLAGS_gs_thread_num);
    }

    graph_client_ = NewGraphClient(graph_config_, (GraphClientEnum)FLAGS_dist);
    if (!graph_client_) {
      return false;
    }

    // used by RunEntry
    epoch_ = FLAGS_epoch;
    batch_node_ = FLAGS_batch_node;
    walk_length_ = FLAGS_walk_length;
    dump_type_ = FLAGS_dump_type;
    entry_flag_ = FLAGS_dist ? "Worker" : "Thread";
    return true;
  }

 private:
  const char* task_name() const noexcept override { return "RandomWalk"; }
  void DumpText(deepx_core::AutoOutputFileStream& ofs /* NOLINT */,
                std::ostringstream& oss /* NOLINT */,
                const std::vector<vec_int_t>& seqs) noexcept {
    for (const auto& seq : seqs) {
      oss.clear();
      oss.str("");

      if (dump_type_ == 0) {
        for (const auto& node : seq) {
          oss << " " << node;
        }
        oss << "\n";
      } else {
        for (size_t i = 1; i < seq.size(); ++i) {
          oss << seq[0] << " " << seq[i] << "\n";
        }
      }

      std::string s = oss.str();
      ofs.Write(s.data(), s.size());
    }
  }

  bool RunEntry(int entry_id, const vec_str_t& files,
                const std::string& out) override {
    DXINFO("%s id: %d is processing ...", entry_flag_.c_str(), entry_id);

    LineParser line_parser;
    deepx_core::AutoOutputFileStream ofs;
    std::ostringstream oss;
    for (int i = 0; i < epoch_; ++i) {
      DXINFO("Random walker epoch: %d.", i);

      for (auto& file : files) {
        DXINFO("Processed file: %s.", file.c_str());
        if (!line_parser.Open(file)) {
          return false;
        }

        auto out_file = deepx_core::GetOutputPredictFile(
            out, file + "_" + std::to_string(i));
        if (!ofs.Open(out_file)) {
          return false;
        }

        // traverse
        std::vector<NodeValue> values;
        vec_int_t cur_nodes;
        std::vector<int> walk_lens(batch_node_, walk_length_);
        WalkerInfo walk_info;
        std::vector<vec_int_t> seqs;
        while (line_parser.NextBatch<NodeValue>(batch_node_, &values)) {
          cur_nodes = Collect<NodeValue, int_t>(values, &NodeValue::node);
          graph_client_->StaticTraverse(cur_nodes, walk_lens, walk_info, &seqs);
          DumpText(ofs, oss, seqs);
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
    DXCHECK(FLAGS_gs_thread_num > 0);
  }
  DXCHECK(!FLAGS_node_graph.empty());

  DXCHECK(FLAGS_walk_length > 0);
  DXCHECK(FLAGS_random_walker_type == 0 || FLAGS_random_walker_type == 1);
  DXCHECK(FLAGS_epoch > 0);
  DXCHECK(FLAGS_batch_node > 0);
  DXCHECK(!FLAGS_out.empty());
}

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  std::unique_ptr<MainUtil> main(new RandomWalkerMain);
  if (!main->Init()) {
    return -1;
  }

  if (FLAGS_dist) {
    if (!main->RunSingleWorker(FLAGS_node_graph, FLAGS_out, FLAGS_gs_worker_num,
                               FLAGS_gs_worker_id)) {
      return -1;
    }
  } else {
    if (!main->RunMultiThread(FLAGS_node_graph, FLAGS_out,
                              FLAGS_gs_thread_num)) {
      return -1;
    }
  }

  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
