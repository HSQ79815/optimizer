/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseReshapeTransposeReshape final : public PredicateBasedPass {
  explicit FuseReshapeTransposeReshape()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_reshape_transpose_reshape";
  }

  bool patternMatchPredicate(Node* node) override {
    return CheckKind(node, kReshape) && CheckKind(node->input(0), kTranspose) &&
           CheckKind(node->input(0)->node()->input(0), kReshape);
  }

  bool runTransform(Node* node, Graph& g,
                    NodeDestroyType& destroy_current) override {
    auto* reshape_0 = node->input(0)->node()->input(0)->node();
    auto* transpose = node->input(0)->node();
    auto* reshape_1 = node;

    if (!IsConstantTensor(reshape_0, 1) || !IsConstantTensor(reshape_1, 1) ||
        !transpose->hasAttribute(kperm)) {
      return false;
    }

    const Tensor* shape_tensor_of_reshape_0 =
        FetchConstantTensor(reshape_0->input(1));
    const Tensor* shape_tensor_of_reshape_1 =
        FetchConstantTensor(reshape_1->input(1));
    ONNX_ASSERT(shape_tensor_of_reshape_0->elem_type() ==
                ONNX_NAMESPACE::TensorProto_DataType_INT64);
    ONNX_ASSERT(shape_tensor_of_reshape_1->elem_type() ==
                ONNX_NAMESPACE::TensorProto_DataType_INT64);
    const auto& perm = transpose->is(kperm);
    const auto shape_of_reshape_0 =
        ParseData<int64_t>(shape_tensor_of_reshape_0);
    const auto shape_of_reshape_1 =
        ParseData<int64_t>(shape_tensor_of_reshape_1);
    ONNX_ASSERT(shape_of_reshape_0.size() == perm.size());

    if (shape_of_reshape_1.size() >= perm.size()) {
      return false;
    }

    int unchanged = 0, k = 0;
    std::vector<int64_t> new_shape, new_perm;
    for (int i = 0; i < perm.size(); ++i) {
      if (i == perm[i] && unchanged == 0) {
        unchanged++;
        continue;
      }
      if (k >= shape_of_reshape_1.size() ||
          shape_of_reshape_0[perm[i]] != shape_of_reshape_1[k++]) {
        return false;
      }
      new_shape.push_back(shape_of_reshape_0[i]);
      new_perm.push_back(perm[i] - unchanged);
    }
    if (new_perm.empty()) {
      // now transpose does nothing, it can be eliminate by
      // eliminate_nop_transpose
      return false;
    }

    Tensor shape_tensor;
    shape_tensor.sizes().push_back(new_shape.size());
    shape_tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    shape_tensor.int64s().swap(new_shape);

    auto* new_reshape_node = g.create(kReshape, 1);
    new_reshape_node->addInput(reshape_0->input(0));
    new_reshape_node->addInput(g.addInitializerAndCreateValue(shape_tensor));
    new_reshape_node->insertBefore(reshape_1);
    Symbol allowzero(std::string("allowzero"));
    if (reshape_1->hasAttribute(allowzero)) {
      new_reshape_node->i_(allowzero, reshape_1->i(allowzero));
    }

    auto* new_transpose_node = g.create(kTranspose, 1);
    new_transpose_node->is_(kperm, std::move(new_perm));
    new_transpose_node->addInput(new_reshape_node->output());
    new_transpose_node->insertBefore(reshape_1);

    const bool replacing_success =
        tryReplacingAllUsesWith(reshape_1, new_transpose_node);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
