// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Shuting Guo (shutingnjupt@gmail.com)
//

#include "src/model/instance_reader/unsup_bipartite_inst_reader.h"

namespace embedx {

/************************************************************************/
/* UnsupBipartiteInstReader */
/************************************************************************/
class UnsupBipartiteGraphsageInstReader : public UnsupBipartiteInstReader {
 public:
  DEFINE_INSTANCE_READER_LIKE(UnsupBipartiteGraphsageInstReader);
};

INSTANCE_READER_REGISTER(UnsupBipartiteGraphsageInstReader,
                         "UnsupBipartiteGraphsageInstReader");
INSTANCE_READER_REGISTER(UnsupBipartiteGraphsageInstReader,
                         "unsup_bipartite_graphsage");

}  // namespace embedx
