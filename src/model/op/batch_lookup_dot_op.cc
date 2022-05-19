// Tencent is pleased to support the open source community by making embedx
// available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
//
// Licensed under the BSD 3-Clause License and other third-party components,
// please refer to LICENSE for details.
//
// Author: Zhenting Yu (zhenting.yu@gmail.com)
//

#include <deepx_core/dx_log.h>

#include "src/model/op/gnn_graph_node.h"

namespace embedx {

void BatchLookupDotInferShape(int row, Shape* Z) noexcept { Z->resize(row, 1); }

/************************************************************************/
/* forward functions */
/************************************************************************/

// Win: tsr; Wout: tsr
template <typename T, typename I>
void BatchLookupDotForward(const CSRMatrix<T, I>& Xin,
                           const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                           const Tensor<T>& Wout, Tensor<T>* Z) {
  DXASSERT(Win.dim(1) == Wout.dim(1));
  DXASSERT(Z->same_shape(Xin.row(), 1));

  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int Wout_row = Wout.dim(0);
  int col = Win.dim(1);

  const auto* _Win = Win.data();
  const auto* _Wout = Wout.data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = _Wout + (Xout.col(i) % Wout_row) * col;
    Z->data(i) = deepx_core::LLMath<T>::dot(col, src, dst);
  }
}

// Win: tsr; Wout: srm
template <typename T, typename I>
void BatchLookupDotForward(const CSRMatrix<T, I>& Xin,
                           const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                           const SparseRowMatrix<T, I>& Wout, Tensor<T>* Z) {
  DXASSERT(Win.dim(1) == Wout.col());
  DXASSERT(Z->same_shape(Xin.row(), 1));

  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int col = Win.dim(1);

  const auto* _Win = Win.data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = Wout.get_row_no_init(Xout.col(i));
    Z->data(i) = deepx_core::LLMath<T>::dot(col, src, dst);
  }
}

// Win: srm; Wout: srm
template <typename T, typename I>
void BatchLookupDotForward(const CSRMatrix<T, I>& Xin,
                           const CSRMatrix<T, I>& Xout,
                           const SparseRowMatrix<T, I>& Win,
                           const SparseRowMatrix<T, I>& Wout, Tensor<T>* Z) {
  DXASSERT(Win.col() == Wout.col());
  DXASSERT(Z->same_shape(Xin.row(), 1));

  int Xrow = Xin.row();
  int col = Win.col();
  for (int i = 0; i < Xrow; ++i) {
    const auto* src = Win.get_row_no_init(Xin.col(i));
    const auto* dst = Wout.get_row_no_init(Xout.col(i));
    Z->data(i) = deepx_core::LLMath<T>::dot(col, src, dst);
  }
}

/************************************************************************/
/* backward functions */
/************************************************************************/

// Win: tsr_hidden; Wout: tsr_hidden
// gWin: tsr; gWout: tsr
template <typename T, typename I>
void BatchLookupDotBackward(const CSRMatrix<T, I>& Xin,
                            const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                            const Tensor<T>& Wout, const Tensor<T>& /*Z*/,
                            const Tensor<T>& gZ, Tensor<T>* gWin,
                            Tensor<T>* gWout) {
  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int Wout_row = Wout.dim(0);
  int col = gWin->dim(1);
  const auto* _Win = Win.data();
  const auto* _Wout = Wout.data();
  auto* _gWin = gWin->data();
  auto* _gWout = gWout->data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = _Wout + (Xout.col(i) % Wout_row) * col;
    auto* gsrc = _gWin + (Xin.col(i) % Win_row) * col;
    auto* gdst = _gWout + (Xout.col(i) % Wout_row) * col;
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), src, gdst);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), dst, gsrc);
  }
}

// Win: tsr_hidden; Wout: tsr_param
// gWin: tsr; gWout: srm
template <typename T, typename I>
void BatchLookupDotBackward(const CSRMatrix<T, I>& Xin,
                            const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                            const Tensor<T>& Wout, const Tensor<T>& /*Z*/,
                            const Tensor<T>& gZ, Tensor<T>* gWin,
                            SparseRowMatrix<T, I>* gWout) {
  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int Wout_row = Wout.dim(0);
  int col = gWin->dim(1);
  const auto* _Win = Win.data();
  const auto* _Wout = Wout.data();
  auto* _gWin = gWin->data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = _Wout + (Xout.col(i) % Wout_row) * col;
    auto* gsrc = _gWin + (Xin.col(i) % Win_row) * col;
    auto* gdst = gWout->get_row_no_init(Xout.col(i) % Wout_row);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), src, gdst);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), dst, gsrc);
  }
}

// Win: tsr_hidden; Wout: srm
// gWin: tsr; gWout: srm
template <typename T, typename I>
void BatchLookupDotBackward(const CSRMatrix<T, I>& Xin,
                            const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                            const SparseRowMatrix<T, I>& Wout,
                            const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                            Tensor<T>* gWin, SparseRowMatrix<T, I>* gWout) {
  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int col = gWin->dim(1);
  const auto* _Win = Win.data();
  auto* _gWin = gWin->data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = Wout.get_row_no_init(Xout.col(i));
    auto* gsrc = _gWin + (Xin.col(i) % Win_row) * col;
    auto* gdst = gWout->get_row_no_init(Xout.col(i));
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), src, gdst);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), dst, gsrc);
  }
}

// Win: tsr_param; Wout: tsr_param
// gWin: srm; gWout: srm
template <typename T, typename I>
void BatchLookupDotBackward(const CSRMatrix<T, I>& Xin,
                            const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                            const Tensor<T>& Wout, const Tensor<T>& /*Z*/,
                            const Tensor<T>& gZ, SparseRowMatrix<T, I>* gWin,
                            SparseRowMatrix<T, I>* gWout) {
  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int Wout_row = Wout.dim(0);
  int col = gWin->col();
  const auto* _Win = Win.data();
  const auto* _Wout = Wout.data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = _Wout + (Xout.col(i) % Wout_row) * col;
    auto* gsrc = gWin->get_row_no_init(Xin.col(i) % Win_row);
    auto* gdst = gWout->get_row_no_init(Xout.col(i) % Wout_row);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), src, gdst);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), dst, gsrc);
  }
}

// Win: tsr_param; Wout: srm
// gWin: srm; gWout: srm
template <typename T, typename I>
void BatchLookupDotBackward(const CSRMatrix<T, I>& Xin,
                            const CSRMatrix<T, I>& Xout, const Tensor<T>& Win,
                            const SparseRowMatrix<T, I>& Wout,
                            const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                            SparseRowMatrix<T, I>* gWin,
                            SparseRowMatrix<T, I>* gWout) {
  int Xrow = Xin.row();
  int Win_row = Win.dim(0);
  int col = gWin->col();
  const auto* _Win = Win.data();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = _Win + (Xin.col(i) % Win_row) * col;
    const auto* dst = Wout.get_row_no_init(Xout.col(i));
    auto* gsrc = gWin->get_row_no_init(Xin.col(i) % Win_row);
    auto* gdst = gWout->get_row_no_init(Xout.col(i));
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), src, gdst);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), dst, gsrc);
  }
}

// Win: srm; Wout: srm
// gWin: srm; gWout: srm
template <typename T, typename I>
void BatchLookupDotBackward(const CSRMatrix<T, I>& Xin,
                            const CSRMatrix<T, I>& Xout,
                            const SparseRowMatrix<T, I>& Win,
                            const SparseRowMatrix<T, I>& Wout,
                            const Tensor<T>& /*Z*/, const Tensor<T>& gZ,
                            SparseRowMatrix<T, I>* gWin,
                            SparseRowMatrix<T, I>* gWout) noexcept {
  int Xrow = Xin.row();
  int col = gWin->col();

  for (int i = 0; i < Xrow; ++i) {
    const auto* src = Win.get_row_no_init(Xin.col(i));
    const auto* dst = Wout.get_row_no_init(Xout.col(i));
    auto* gsrc = gWin->get_row_no_init(Xin.col(i));
    auto* gdst = gWout->get_row_no_init(Xout.col(i));
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), src, gdst);
    deepx_core::LLMath<T>::axpy(col, gZ.data(i), dst, gsrc);
  }
}

BatchLookupDotNode::BatchLookupDotNode(std::string name, GraphNode* Xin,
                                       GraphNode* Xout, GraphNode* Win,
                                       GraphNode* Wout)
    : GraphNode(std::move(name)) {
  DXCHECK_THROW(Xin->node_type() == deepx_core::GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(Xout->node_type() == deepx_core::GRAPH_NODE_TYPE_INSTANCE);
  DXCHECK_THROW(Xin->tensor_type() == deepx_core::TENSOR_TYPE_CSR);
  DXCHECK_THROW(Xout->tensor_type() == deepx_core::TENSOR_TYPE_CSR);
  DXCHECK_THROW(Win->tensor_type() == deepx_core::TENSOR_TYPE_TSR ||
                Win->tensor_type() == deepx_core::TENSOR_TYPE_SRM);
  DXCHECK_THROW(Wout->tensor_type() == deepx_core::TENSOR_TYPE_TSR ||
                Wout->tensor_type() == deepx_core::TENSOR_TYPE_SRM);

  input_ = {Xin, Xout, Win, Wout};
  node_type_ = deepx_core::GRAPH_NODE_TYPE_HIDDEN;
  tensor_type_ = deepx_core::TENSOR_TYPE_TSR;

  if (Xin->shape().is_rank(2)) {
    BatchLookupDotInferShape(Xin->shape()[0], &shape_);
  }
}

class BatchLookupDotOp : public deepx_core::OpImpl {
 private:
  const GraphNode* Xin_node_ = nullptr;
  const GraphNode* Xout_node_ = nullptr;
  const GraphNode* Win_node_ = nullptr;
  const GraphNode* Wout_node_ = nullptr;
  int Win_node_type_ = deepx_core::GRAPH_NODE_TYPE_NONE;
  int Win_tensor_type_ = deepx_core::TENSOR_TYPE_NONE;
  int Wout_node_type_ = deepx_core::GRAPH_NODE_TYPE_NONE;
  int Wout_tensor_type_ = deepx_core::TENSOR_TYPE_NONE;

  const csr_t* Xin_ = nullptr;
  const csr_t* Xout_ = nullptr;
  const tsr_t* Win_tsr_ = nullptr;
  const srm_t* Win_srm_ = nullptr;
  const tsr_t* Wout_tsr_ = nullptr;
  const srm_t* Wout_srm_ = nullptr;

  Shape Zshape_;
  tsr_t* Z_ = nullptr;
  tsr_t* gZ_ = nullptr;
  tsr_t* gWin_tsr_ = nullptr;
  tsr_t* gWout_tsr_ = nullptr;
  srm_t* gWin_srm_ = nullptr;
  srm_t* gWout_srm_ = nullptr;

 public:
  DEFINE_OP_LIKE(BatchLookupDotOp);

  void InitForward() override {
    Xin_node_ = node_->input(0);
    DXCHECK_THROW(!Xin_node_->need_grad());
    Xout_node_ = node_->input(1);
    DXCHECK_THROW(!Xout_node_->need_grad());
    Win_node_ = node_->input(2);
    Win_node_type_ = Win_node_->node_type();
    Win_tensor_type_ = Win_node_->tensor_type();
    Wout_node_ = node_->input(3);
    Wout_node_type_ = Wout_node_->node_type();
    Wout_tensor_type_ = Wout_node_->tensor_type();

    Xin_ = GetPtrCSR(Xin_node_);
    Xout_ = GetPtrCSR(Xout_node_);
    Win_tsr_ = nullptr;
    Win_srm_ = nullptr;
    Wout_tsr_ = nullptr;
    Wout_srm_ = nullptr;

    switch (Win_tensor_type_) {
      case deepx_core::TENSOR_TYPE_TSR:
        Win_tsr_ = GetPtrTSR(Win_node_);
        break;
      case deepx_core::TENSOR_TYPE_SRM:
        Win_srm_ = GetPtrSRM(Win_node_);
        break;
    }
    switch (Wout_tensor_type_) {
      case deepx_core::TENSOR_TYPE_TSR:
        Wout_tsr_ = GetPtrTSR(Wout_node_);
        break;
      case deepx_core::TENSOR_TYPE_SRM:
        Wout_srm_ = GetPtrSRM(Wout_node_);
        break;
    }

    DXCHECK_THROW(Xin_->row() == (int)Xin_->col_size());
    DXCHECK_THROW(Xout_->row() == (int)Xout_->col_size());
    DXCHECK_THROW(Xin_->row() == Xout_->row());
    BatchLookupDotInferShape(Xin_->row(), &Zshape_);
    Z_ = InitHiddenTSR(node_, Zshape_);
  }

  void InitBackward() override {
    gZ_ = GetGradPtrTSR(node_);

    if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
      gWin_tsr_ = InitGradTSR(Win_node_, Win_tsr_->shape());
    } else if (Win_tensor_type_ == deepx_core::TENSOR_TYPE_TSR) {
      gWin_srm_ = InitGradSRM(Win_node_, Win_tsr_->shape());
    } else {
      gWin_srm_ = InitGradSRM(Win_node_, Win_srm_->shape());
    }

    if (Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
      gWout_tsr_ = InitGradTSR(Wout_node_, Wout_tsr_->shape());
    } else if (Wout_tensor_type_ == deepx_core::TENSOR_TYPE_TSR) {
      gWout_srm_ = InitGradSRM(Wout_node_, Wout_tsr_->shape());
    } else {
      gWout_srm_ = InitGradSRM(Wout_node_, Wout_srm_->shape());
    }
  }

  void Forward() override {
    if (Win_tensor_type_ == deepx_core::TENSOR_TYPE_TSR &&
        Wout_tensor_type_ == deepx_core::TENSOR_TYPE_TSR) {
      BatchLookupDotForward(*Xin_, *Xout_, *Win_tsr_, *Wout_tsr_, Z_);
    } else if (Win_tensor_type_ == deepx_core::TENSOR_TYPE_SRM &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotForward(*Xin_, *Xout_, *Win_srm_, *Wout_srm_, Z_);
    } else if (Win_tensor_type_ == deepx_core::TENSOR_TYPE_TSR &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotForward(*Xin_, *Xout_, *Win_tsr_, *Wout_srm_, Z_);
    } else {
      BatchLookupDotForward(*Xout_, *Xin_, *Wout_tsr_, *Win_srm_, Z_);
    }
  }

  void Backward() override {
    // TODO(succ9420): simplify code
    if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN &&
        Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN) {
      BatchLookupDotBackward(*Xin_, *Xout_, *Win_tsr_, *Wout_tsr_, *Z_, *gZ_,
                             gWin_tsr_, gWout_tsr_);
    } else if (Win_tensor_type_ == deepx_core::TENSOR_TYPE_SRM &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotBackward(*Xin_, *Xout_, *Win_srm_, *Wout_srm_, *Z_, *gZ_,
                             gWin_srm_, gWout_srm_);
    } else if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN &&
               Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_TSR) {
      BatchLookupDotBackward(*Xin_, *Xout_, *Win_tsr_, *Wout_tsr_, *Z_, *gZ_,
                             gWin_tsr_, gWout_srm_);
    } else if (Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN &&
               Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM &&
               Win_tensor_type_ == deepx_core::TENSOR_TYPE_TSR) {
      BatchLookupDotBackward(*Xout_, *Xin_, *Wout_tsr_, *Win_tsr_, *Z_, *gZ_,
                             gWout_tsr_, gWin_srm_);
    } else if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotBackward(*Xin_, *Xout_, *Win_tsr_, *Wout_srm_, *Z_, *gZ_,
                             gWin_tsr_, gWout_srm_);
    } else if (Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_HIDDEN &&
               Win_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotBackward(*Xout_, *Xin_, *Wout_tsr_, *Win_srm_, *Z_, *gZ_,
                             gWout_tsr_, gWin_srm_);
    } else if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM &&
               Win_tensor_type_ == deepx_core::TENSOR_TYPE_TSR &&
               Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_TSR) {
      BatchLookupDotBackward(*Xin_, *Xout_, *Win_tsr_, *Wout_tsr_, *Z_, *gZ_,
                             gWin_srm_, gWout_srm_);
    } else if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM &&
               Win_tensor_type_ == deepx_core::TENSOR_TYPE_TSR &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotBackward(*Xin_, *Xout_, *Win_tsr_, *Wout_srm_, *Z_, *gZ_,
                             gWin_srm_, gWout_srm_);
    } else if (Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM &&
               Wout_tensor_type_ == deepx_core::TENSOR_TYPE_TSR &&
               Win_tensor_type_ == deepx_core::TENSOR_TYPE_SRM) {
      BatchLookupDotBackward(*Xout_, *Xin_, *Wout_tsr_, *Win_srm_, *Z_, *gZ_,
                             gWout_srm_, gWin_srm_);
    } else {
      DXERROR("BatchLokkupOp got unsupported input type.");
    }
  }

  void GetPullRequest(deepx_core::PullRequest* pull_request) const override {
    if (Win_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM) {
      const std::string& Win_name = Win_node_->name();
      switch (Win_tensor_type_) {
        case deepx_core::TENSOR_TYPE_TSR:
          pull_request->tsr_set.emplace(Win_name);
          break;
        case deepx_core::TENSOR_TYPE_SRM:
          pull_request->srm_map[Win_name].insert(Xin_->col_begin(),
                                                 Xin_->col_end());
          break;
      }
    }

    if (Wout_node_type_ == deepx_core::GRAPH_NODE_TYPE_PARAM) {
      const std::string& Wout_name = Wout_node_->name();
      switch (Wout_tensor_type_) {
        case deepx_core::TENSOR_TYPE_TSR:
          pull_request->tsr_set.emplace(Wout_name);
          break;
        case deepx_core::TENSOR_TYPE_SRM:
          pull_request->srm_map[Wout_name].insert(Xout_->col_begin(),
                                                  Xout_->col_end());
          break;
      }
    }
  }
};

GRAPH_NODE_REGISTER(BatchLookupDotNode);
OP_REGISTER(BatchLookupDotOp, "BatchLookupDotNode");

}  // namespace embedx
