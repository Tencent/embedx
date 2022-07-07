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
#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <gflags/gflags.h>

#include <cinttypes>  // PRIu64
#include <memory>     // std::unique_ptr
#include <sstream>    // std::ostringstream
#include <string>
#include <vector>

#include "src/common/data_types.h"
#include "src/graph/client/graph_client.h"
#include "src/graph/graph_config.h"
#include "src/io/io_util.h"
#include "src/io/line_parser.h"
#include "src/io/value.h"
#include "src/sampler/random_walker_data_types.h"
#include "src/tools/graph/graph_flags.h"
#include "src/tools/graph/main_util.h"

// random_walk_main
DEFINE_int32(walk_length, 10, "Length of random walk sequence.");
DEFINE_int32(dump_type, 0,
             "When dump_type is 0 output sequence, otherwise output edge.");
DEFINE_int32(epoch, 1, "Number of epochs.");
DEFINE_string(meta_path_config, "", "Meta path config.");

namespace embedx {
namespace {

constexpr int BATCH = 128;

bool LoadMetaPathConfig(const std::string& file,
                        std::vector<meta_path_t>* meta_paths) {
  LineParser line_parser;
  if (!line_parser.Open(file)) {
    return false;
  }

  meta_paths->clear();
  std::vector<SeqValue> seq_values;
  while (line_parser.NextBatch<SeqValue>(BATCH, &seq_values)) {
    for (const auto& seq_value : seq_values) {
      meta_paths->emplace_back(
          meta_path_t(seq_value.nodes.begin(), seq_value.nodes.end()));
    }
  }

  DXINFO("Loaded meta path size: %zu.", meta_paths->size());
  return !meta_paths->empty();
}

bool ParseMetaPathConfig(const std::string& config,
                         std::vector<meta_path_t>* meta_paths) {
  std::vector<std::string> str_meta_paths;
  deepx_core::Split(config, ",", &str_meta_paths);

  meta_paths->clear();
  meta_path_t meta_path;
  for (const auto& str_meta_path : str_meta_paths) {
    if (deepx_core::Split(str_meta_path, ":", &meta_path)) {
      meta_paths->emplace_back(meta_path);
    } else {
      DXERROR("Invalid meta path: %s.", str_meta_path.c_str());
      return false;
    }
  }

  DXINFO("Loaded meta path size: %zu.", meta_paths->size());
  return !meta_paths->empty();
}

class RandomWalkerMain : public MainUtil {
 private:
  std::unique_ptr<GraphClient> graph_client_;
  GraphConfig graph_config_;

  int epoch_;
  int batch_node_;
  int walk_length_;
  int dump_type_;
  std::string meta_path_config_;
  std::vector<WalkerInfo> walker_infos_;
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

    meta_path_config_ = FLAGS_meta_path_config;
    if (!meta_path_config_.empty()) {
      if (!InitWalkerInfo(meta_path_config_, walk_length_)) {
        return false;
      }
    }

    entry_flag_ = FLAGS_dist ? "Worker" : "Thread";
    return true;
  }

 private:
  bool InitWalkerInfo(const std::string& meta_path_config, int walk_length) {
    std::vector<meta_path_t> meta_paths;
    if (!LoadMetaPathConfig(meta_path_config, &meta_paths) &&
        !ParseMetaPathConfig(meta_path_config, &meta_paths)) {
      return false;
    }

    WalkerInfo walker_info;
    for (const auto& meta_path : meta_paths) {
      walker_info.meta_path = meta_path;
      walker_info.walker_length = walk_length;
      walker_infos_.emplace_back(walker_info);
    }
    return true;
  }

 private:
  const char* task_name() const noexcept override { return "RandomWalk"; }
  void DumpText(deepx_core::AutoOutputFileStream& ofs /* NOLINT */,
                std::ostringstream& oss /* NOLINT */,
                const vec_int_t& cur_nodes,
                const std::vector<vec_int_t>& seqs) noexcept {
    for (size_t i = 0; i < cur_nodes.size(); ++i) {
      auto cur_node = cur_nodes[i];
      const auto& seq = seqs[i];
      if (seq.empty()) {
        continue;
      }

      oss.clear();
      oss.str("");

      if (dump_type_ == 0) {
        oss << cur_node;
        for (auto node : seq) {
          oss << " " << node;
        }
        oss << "\n";
      } else {
        for (auto node : seq) {
          oss << cur_node << " " << node << "\n";
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
        std::vector<vec_int_t> seqs;

        if (!meta_path_config_.empty()) {
          while (line_parser.NextBatch<NodeValue>(batch_node_, &values)) {
            for (const auto& walker_info : walker_infos_) {
              FilterNode(values, walker_info.meta_path, &cur_nodes);
              graph_client_->StaticTraverse(cur_nodes, walk_lens, walker_info,
                                            &seqs);
              DumpText(ofs, oss, cur_nodes, seqs);
            }
          }
        } else {
          WalkerInfo walker_info;
          while (line_parser.NextBatch<NodeValue>(batch_node_, &values)) {
            cur_nodes = Collect<NodeValue, int_t>(values, &NodeValue::node);
            graph_client_->StaticTraverse(cur_nodes, walk_lens, walker_info,
                                          &seqs);
            DumpText(ofs, oss, cur_nodes, seqs);
          }
        }
      }
    }
    return true;
  }

 private:
  void FilterNode(const std::vector<NodeValue>& values,
                  const meta_path_t& meta_path, vec_int_t* cur_nodes) {
    DXCHECK(meta_path.size() > 0);
    cur_nodes->clear();
    for (const auto& value : values) {
      uint16_t node_type = io_util::GetNodeType(value.node);
      if (node_type != meta_path[0]) {
        DXERROR("The first node: %" PRIu64
                " mismatchs the first type of meta path, expected node type: "
                "%d, got "
                "node type: %d",
                value.node, meta_path[0], node_type);
      } else {
        cur_nodes->emplace_back(value.node);
      }
    }
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
