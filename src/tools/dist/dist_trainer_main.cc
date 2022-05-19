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

#include <thread>

#include "src/tools/dist/dist_flags.h"

namespace embedx {

void RunCoordServer();
void RunParamServer();
void RunWorker();

namespace {

int main(int argc, char** argv) {
  google::SetUsageMessage("Usage: [Options]");
#if HAVE_COMPILE_FLAGS_H == 1
  google::SetVersionString("\n\n"
#include "compile_flags.h"
  );
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);

  CheckFlags();

  if (FLAGS_role == "ps") {
    if (FLAGS_ps_id == 0) {
      std::thread cs_thread(RunCoordServer);
      std::thread ps_thread(RunParamServer);
      ps_thread.join();
      cs_thread.join();
    } else {
      RunParamServer();
    }
    DXINFO("Param server: %d normally exits.", FLAGS_ps_id);
  } else {
    RunWorker();
    DXINFO("Worker normally exits.");
  }

  google::ShutDownCommandLineFlags();
  return 0;
}

}  // namespace
}  // namespace embedx

int main(int argc, char** argv) { return embedx::main(argc, argv); }
