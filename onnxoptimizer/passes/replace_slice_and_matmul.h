/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

/*
Before
  Y = Matmul(Slice(data, start, end, axes) ,rhs) , where data and rhs are
constant tensor, axes of slice should be constant tensor as well as the value of
axes should not represent the last shape.

After
  Y = Slice(Matmul(data, rhs), start, end, axes) , where Matmul can be folded.
*/

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ReplaceSliceAndMatmul final : public PredicateBasedPass {
  explicit ReplaceSliceAndMatmul()
      : PredicateBasedPass(PassType::Replace, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "replace_slice_and_matmul";
  }

  bool patternMatchPredicate(Node* node) override {
    int64_t slice_axis;
    const bool result = CheckKind(node, kMatMul) &&
                        CheckKind(node->inputs()[0], kSlice) &&
                        // rhs should be constant tensor
                        IsConstantTensor(node, 1) &&
                        // lhs should be constant tensor
                        IsConstantTensor(node->inputs()[0]->node(), 0) &&
                        // slice should have explicit axes
                        node->inputs()[0]->node()->inputs().size() >= 4 &&
                        // axes of slice should be constant tensor
                        IsConstantTensor(node->inputs()[0]->node(), 3);
    if (!result) {
      return false;
    }
    const int64_t rank = node->inputs()[0]->node()->inputs()[0]->sizes().size();
    const Tensor* axes =
        FetchConstantTensor(node->inputs()[0]->node()->inputs()[3]);
    if (axes->elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      const auto data = ParseData<int32_t>(axes);
      return std::none_of(data.cbegin(), data.cend(), [&rank](int32_t d) {
        return AddYIfNegative<int>(d, rank) == rank - 1;
      });
    } else if (axes->elem_type() ==
               ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      const auto data = ParseData<int64_t>(axes);
      return std::none_of(data.cbegin(), data.cend(), [&rank](int64_t d) {
        return AddYIfNegative<int64_t>(d, rank) == rank - 1;
      });
    } else {
      return false;
    }
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Value* slice_value = n->inputs()[0];
    Value* mat_y = n->inputs()[1];

    Node* slice = slice_value->node();
    Value* mat_x = slice->inputs()[0];

    Node* new_matmul = graph.create(kMatMul, 1);
    new_matmul->addInput(mat_x);
    new_matmul->addInput(mat_y);

    Node* new_slice = graph.create(kSlice, 1);
    new_slice->addInput(new_matmul->output());
    for (int i = 1; i < slice->inputs().size(); ++i) {
      new_slice->addInput(slice->inputs()[i]);
    }

    new_slice->insertBefore(n);
    new_matmul->insertBefore(new_slice);

    const bool replacing_success = tryReplacingAllUsesWith(n, new_slice);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
