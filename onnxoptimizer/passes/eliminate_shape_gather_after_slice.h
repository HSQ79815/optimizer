/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

/*
Before
          Slice(data, start, end, axes)
                  |
                Shape
                  |
              Gather(indices)
                  |
                  Y
    where data tensor should have sizes and indices and axes are disjoint

After
  Y = constant tensor
*/

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateShapeGatherAfterSlice final : public PredicateBasedPass {
  explicit EliminateShapeGatherAfterSlice()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_shape_gather_after_slice";
  }

  bool patternMatchPredicate(Node *node) override {
    return CheckKind(node, "Gather") &&
           // indices of gather should be a constant tensor
           IsConstantTensor(node, 1) && CheckKind(node->inputs()[0], "Shape") &&
           CheckKind(node->inputs()[0]->node()->inputs()[0], kSlice) &&
           // axes of slice is explicit
           node->inputs()[0]->node()->inputs()[0]->node()->inputs().size() >=
               4 &&
           IsConstantTensor(node->inputs()[0]->node()->inputs()[0]->node(), 3);
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    const Node *shape = node->inputs()[0]->node();
    const Node *slice = shape->inputs()[0]->node();

    if (!slice->inputs()[0]->has_sizes()) {
      return false;
    }

    const auto &shaps_of_slice_data = slice->inputs()[0]->sizes();
    const int64_t rank = shaps_of_slice_data.size();

    // We have validated that indices and axes are constant in
    // patternMatchPredicate
    const Tensor *indices = FetchConstantTensor(node->inputs()[1]);

    std::vector<int64_t> indices_data, axes_data;
    FetchIntsOfTensor(indices, indices_data);
    FetchIntsOfTensor(slice->inputs()[3], axes_data);

    // explicit use rank instead of shape->input()->sizes().size()
    const auto [start, end] = FetchStartAndEndAttrOfShape(shape, rank);

    for (auto &indices_val : indices_data) {
      indices_val = AddYIfNegative<int64_t>(indices_val, end - start);
      indices_val += start;
    }

    for (auto &axis : axes_data) {
      axis = AddYIfNegative<int64_t>(axis, rank);
    }

    if (IsIntersection(indices_data, axes_data)) {
      return false;
    }

    std::vector<int64_t> gather_output;
    for (auto &indices_val : indices_data) {
      ONNX_ASSERT(indices_val < rank);
      if (!shaps_of_slice_data[indices_val].is_int) {
        return false;
      }
      gather_output.push_back(shaps_of_slice_data[indices_val].dim);
    }

    Tensor tensor;
    if (!indices->sizes().empty()) {
      // if the output of gather is a tensor, it should have shapes
      tensor.sizes().push_back(indices->sizes().size());
    }
    tensor.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    tensor.int64s().swap(gather_output);
    Value *value = graph.addInitializerAndCreateValue(tensor);

    const bool replacing_success =
        tryReplacingAllUsesWith(node->output(), value);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
