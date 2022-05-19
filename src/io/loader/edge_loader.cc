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

#include <deepx_core/dx_log.h>

#include <memory>  // std::unique_ptr
#include <vector>

#include "src/common/data_types.h"
#include "src/io/line_parser.h"
#include "src/io/loader/loader.h"
#include "src/io/storage/storage.h"
#include "src/io/value.h"

namespace embedx {
namespace {

constexpr int BATCH = 128;

}

class EdgeLoader : public Loader {
 private:
  int shard_num_;
  int shard_id_;
  std::unique_ptr<Storage> store_;

 public:
  explicit EdgeLoader(int shard_num = 1, int shard_id = 0, int store_type = 0)
      : shard_num_(shard_num), shard_id_(shard_id) {
    store_ = NewEdgeStorage(store_type);
  }

  ~EdgeLoader() override = default;

 public:
  void Clear() noexcept override { store_->Clear(); }
  void Reserve(uint64_t estimated_size) override {
    store_->Reserve(estimated_size);
  }
  const Storage* storage() const noexcept override { return store_.get(); }

 private:
  bool LoadEntry(const vec_str_t& files, int thread_id) override {
    std::vector<EdgeValue> values;
    LineParser line_parser;

    for (const auto& file : files) {
      DXINFO("Thread: %d is processing file: %s.", thread_id, file.c_str());

      if (!line_parser.Open(file)) {
        return false;
      }

      while (line_parser.NextBatch<EdgeValue>(BATCH, &values)) {
        store_->Lock();

        for (auto& value : values) {
          if (Loader::PartOfShard(value.src_node, shard_num_, shard_id_)) {
            if (!store_->InsertEdge(&value)) {
              store_->UnLock();
              return false;
            }
          }
        }

        store_->UnLock();
      }
    }

    DXINFO("Done.");
    return true;
  }
};

std::unique_ptr<Loader> NewEdgeLoader(int shard_num, int shard_id,
                                      int store_type) {
  std::unique_ptr<Loader> loader;
  loader.reset(new EdgeLoader(shard_num, shard_id, store_type));
  return loader;
}

}  // namespace embedx
