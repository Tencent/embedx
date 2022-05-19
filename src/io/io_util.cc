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

#include "src/io/io_util.h"

#include <deepx_core/common/stream.h>
#include <deepx_core/dx_log.h>
#include <deepx_core/tensor/ll_tensor.h>

#include <sstream>  // std::istringstream

namespace embedx {
namespace io_util {

bool ListFile(const std::string& dir, vec_str_t* files) {
  files->clear();
  if (!deepx_core::AutoFileSystem::ListRecursive(dir, true, files)) {
    DXERROR("Failed to list file: %s.", dir.c_str());
    return false;
  }

  if (files->empty()) {
    DXERROR("No file: %s.", dir.c_str());
    return false;
  }

  return true;
}

uint16_t GetNodeType(int_t node) {
  return deepx_core::LLSparseTensor<float_t, int_t>::get_group_id(node);
}

void ParseMaxNodeType(int max_num, const vec_int_t& node_list,
                      std::unordered_set<uint16_t>* ns_id_set) {
  ns_id_set->clear();
  for (auto node : node_list) {
    ns_id_set->emplace(GetNodeType(node));
    if (ns_id_set->size() > (size_t)max_num) {
      break;
    }
  }
}

static bool InitEmptyConfig(uint16_t* ns_size, id_name_t* id_name_map) {
  uint16_t max_ns_id = 0;
  id_name_map->clear();
  id_name_map->emplace(max_ns_id, "EMPTY_CONFIG_FILE");  // magic namespace name
  *ns_size = max_ns_id + 1;
  return !id_name_map->empty();
}

static bool LoadConfigFromFile(const std::string& file, uint16_t* ns_size,
                               id_name_t* id_name_map) {
  deepx_core::AutoInputFileStream ifs;
  if (!ifs.Open(file)) {
    DXERROR("Failed to open file: %s.", file.c_str());
    return false;
  }

  std::string line;
  std::istringstream iss;
  std::string ns_name;
  uint16_t ns_id = 0, max_ns_id = 0;

  id_name_map->clear();
  while (GetLine(ifs, line)) {
    if (line.find("#") != std::string::npos ||
        line.find("//") != std::string::npos) {
      continue;
    }

    iss.clear();
    iss.str(line);
    if (!(iss >> ns_name >> ns_id)) {
      continue;
    }

    if (id_name_map->find(ns_id) != id_name_map->end()) {
      DXERROR("Duplicate namespace id: %d.", (int)ns_id);
      return false;
    }

    id_name_map->emplace(ns_id, ns_name);
    if (ns_id > max_ns_id) {
      max_ns_id = ns_id;
    }
  }

  DXINFO("Loaded %d namespaces, max namespace id is: %d",
         (int)id_name_map->size(), max_ns_id);

  *ns_size = max_ns_id + 1;
  return !id_name_map->empty();
}

bool LoadConfig(const std::string& file, uint16_t* max_ns_id,
                id_name_t* id_name_map) {
  if (file.empty()) {
    return InitEmptyConfig(max_ns_id, id_name_map);
  } else {
    return LoadConfigFromFile(file, max_ns_id, id_name_map);
  }
}

}  // namespace io_util
}  // namespace embedx
