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

#include "src/io/line_parser.h"

#include <deepx_core/common/str_util.h>

#include "src/common/data_types.h"

namespace embedx {
namespace {

constexpr float_t MAX_WEIGHT = 10;
constexpr float_t MIN_WEIGHT = -10;

bool CheckWeightInRange(float_t weight) {
  if (weight > MAX_WEIGHT || weight < MIN_WEIGHT) {
    DXERROR("Too large or small value: %f, weight should be in [%f, %f].",
            weight, MIN_WEIGHT, MAX_WEIGHT);
    return false;
  }
  return true;
}

}  // namespace

/************************************************************************/
/* NodeValue */
/************************************************************************/
bool LineParser::ParseValue(const std::string& line, NodeValue* value) {
  iss_.clear();
  iss_.str(line);

  if (!(iss_ >> value->node)) {
    DXERROR("Failed to parse node from line: %s.", line.c_str());
    return false;
  }

  // IF a NODE is made up of [node, weight].
  float_t weight;
  if (iss_ >> weight && iss_.eof()) {
    value->weight = weight;
  }

  return true;
}

/************************************************************************/
/* EdgeValue */
/************************************************************************/
bool LineParser::ParseValue(const std::string& line, EdgeValue* value) {
  iss_.clear();
  iss_.str(line);

  if (!(iss_ >> value->src_node >> value->dst_node)) {
    DXERROR("Failed to parse edge from line: %s.", line.c_str());
    return false;
  }

  // IF an EDGE is made up of [src_node, dst_node, weight].
  float_t weight;
  if (iss_ >> weight && iss_.eof()) {
    if (!CheckWeightInRange(weight)) {
      return false;
    }
    value->weight = weight;
  }

  return true;
}

/************************************************************************/
/* SeqValue */
/************************************************************************/
bool LineParser::ParseValue(const std::string& line, SeqValue* value) {
  iss_.clear();
  iss_.str(line);

  int_t node;
  value->nodes.clear();
  while (iss_ >> node) {
    value->nodes.emplace_back(node);
  }

  return !value->nodes.empty();
}

/************************************************************************/
/* AdjValue */
/************************************************************************/
bool LineParser::ParseValue(const std::string& line, AdjValue* value) {
  // AdjValue is make up of [node, id:weight, id:weight ...]
  iss_.clear();
  iss_.str(line);
  if (!(iss_ >> value->node)) {
    DXERROR("Failed to parse node from line: %s.", line.c_str());
    return false;
  }

  // pair
  value->pairs.clear();
  std::string pair;
  vec_str_t tokens;

  while (iss_ >> pair) {
    deepx_core::Split(pair, ":", &tokens);
    if (tokens.size() != 2u) {
      DXERROR("The pair: %s format must be id:value.", pair.c_str());
      return false;
    }

    auto id = std::stoull(tokens[0]);
    auto weight = (float_t)std::stod(tokens[1]);
    if (!CheckWeightInRange(weight)) {
      return false;
    }
    value->pairs.emplace_back(id, weight);
  }

  return !value->pairs.empty();
}

/************************************************************************/
/* NodeAndLabelValue */
/************************************************************************/
bool LineParser::ParseValue(const std::string& line, NodeAndLabelValue* value) {
  iss_.clear();
  iss_.str(line);
  if (!(iss_ >> value->node)) {
    DXERROR("Failed to parse node from line: %s.", line.c_str());
    return false;
  }

  int label;
  value->labels.clear();
  while ((iss_ >> label)) {
    value->labels.emplace_back(label);
  }

  return !value->labels.empty();
}

/************************************************************************/
/* EdgeAndLabelValue */
/************************************************************************/
bool LineParser::ParseValue(const std::string& line, EdgeAndLabelValue* value) {
  iss_.clear();
  iss_.str(line);

  if (!(iss_ >> value->src_node >> value->dst_node)) {
    DXERROR("Failed to parse edge from line: %s.", line.c_str());
    return false;
  }

  if (!(iss_ >> value->node)) {
    DXERROR("Failed to parse node from line: %s.", line.c_str());
    return false;
  }

  int label;
  value->labels.clear();
  while ((iss_ >> label)) {
    value->labels.emplace_back(label);
  }

  return !value->labels.empty();
}

}  // namespace embedx
