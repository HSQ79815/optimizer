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

struct FuseConcatReshapeSubgraph final : public PredicateBasedPass {
  explicit FuseConcatReshapeSubgraph()
      : PredicateBasedPass(PassType::Fuse, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_concat_reshape_subgraph";
  }

  inline bool hasCommonParent(const Node *parent, Node *concat) {
    for (const auto *v : concat->inputs()) {
      const auto *tensor = FetchConstantTensor(v);
      if (tensor) {
        if (tensor->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          return false;
        }
      } else {
        if (!CheckKind(v, kUnsqueeze) ||
            !CheckKind(v->node()->input(0), "Gather") ||
            !CheckKind(v->node()->input(0)->node()->input(0), "Shape") ||
            v->node()->input(0)->node()->input(0)->node()->input()->node() !=
                parent) {
          return false;
        }
      }
    }
    return true;
  }

  inline bool matchConcatReshape(Node *node) {
    return CheckKind(node, kReshape) && CheckKind(node->input(1), kConcat) &&
           node->input(1)->node()->i(kaxis) == 0 &&
           hasCommonParent(node->input(0)->node(), node->input(1)->node());
  }

  inline bool matchMatmulAddConcatReshape(Node *node) {
    bool result = CheckKind(node, kReshape, kAdd, kConcat) &&
                  node->input(1)->node()->i(kaxis) == 0 &&
                  (CheckKind(node->input(0)->node(), kMatMul, kParam) ||
                   CheckKind(node->input(0)->node(), kParam, kMatMul));
    if (!result) {
      return false;
    }
    Node *matmul_node = FetchParrentNode(node->input(0)->node(), kMatMul);
    Node *reshape_data = nullptr;
    if (IsConstantTensor(matmul_node, 0)) {
      reshape_data = matmul_node->input(1)->node();
    } else if (IsConstantTensor(matmul_node, 1)) {
      reshape_data = matmul_node->input(0)->node();
    } else {
      return false;
    }
    return hasCommonParent(reshape_data, node->input(1)->node());
  }

  inline bool matchTransposeConcatReshape(Node *node) {
    return CheckKind(node, kReshape, kTranspose, kConcat) &&
           node->input(1)->node()->i(kaxis) == 0 &&
           node->input(0)->node()->input()->has_sizes() &&
           hasCommonParent(node->input(0)->node()->input()->node(),
                           node->input(1)->node());
  }

  bool patternMatchPredicate(Node *node) override {
    return matchConcatReshape(node) || matchTransposeConcatReshape(node) ||
           matchMatmulAddConcatReshape(node);
  }

  bool runTransform(Node *node, Graph &graph,
                    NodeDestroyType &destroy_current) override {
    Value *reshape_value = node->input(0);
    Node *concat = concat = node->input(1)->node();

    std::vector<int64_t> perm;
    const bool has_transpose = matchTransposeConcatReshape(node);
    if (has_transpose) {
      Node *transpose = reshape_value->node();
      if (transpose->hasAttribute(kperm)) {
        perm = transpose->is(kperm);
      } else {
        for (int i = transpose->input()->sizes().size() - 1; i >= 0; --i) {
          perm.push_back(i);
        }
      }
    }

    std::vector<int64_t> shapes;
    int index = -1;
    for (const auto *v : concat->inputs()) {
      index++;
      const auto *tensor = FetchConstantTensor(v);
      if (tensor) {
        const auto data = ParseData<int64_t>(tensor);
        std::copy(data.cbegin(), data.cend(), std::back_inserter(shapes));
      } else {
        /*
                  data
                 /  \
                /   shape
               /      \
          transpose   gather
              |         |
              |       unsqueeze
              |         |
              |         concat
               \        /
                reshape
            or
                  data
                 /  \
                /   shape
               /      \
              |      gather
              |         |
              |       unsqueeze
              |         |
              |         concat
               \        /
                reshape
        */

        const Node *unsqueeze = v->node();
        const Node *gather = unsqueeze->input(0)->node();
        const Node *shape_node = gather->input(0)->node();
        if (unsqueeze->hasAttribute(kaxes)) {
          // opset 11 adn below
          const auto &unsqueeze_axes = unsqueeze->is(kaxes);
          if (unsqueeze_axes.size() != 1 || unsqueeze_axes[0] != 0) {
            return false;
          }
        } else {
          // opset 13
          int64_t unsqueeze_axes = -1;
          if (!FetchSoleIntValueOfTensor(unsqueeze->input(1), unsqueeze_axes) ||
              unsqueeze_axes != 0) {
            return false;
          }
        }

        int64_t gather_indices;
        if (!FetchSoleIntValueOfTensor(gather->input(1), gather_indices)) {
          return false;
        }

        if (shape_node->hasAttribute(Symbol("start")) ||
            shape_node->hasAttribute(Symbol("end"))) {
          if (!shape_node->input()->has_sizes()) {
            return false;
          }
          auto [start, end] = FetchStartAndEndAttrOfShape(shape_node);
          gather_indices = AddYIfNegative(gather_indices, end - start);
          gather_indices += start;
        }
        if ((has_transpose &&
             (perm.size() < index || gather_indices != perm[index])) ||
            (!has_transpose && gather_indices != index)) {
          return false;
        }
        shapes.push_back(0);
      }
    }

    Tensor t;
    t.sizes().push_back(shapes.size());
    t.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_INT64;
    t.int64s().swap(shapes);
    Value *value = graph.addInitializerAndCreateValue(t);

    const bool replacing_success =
        tryReplacingAllUsesWith(node->input(1), value);
    if (!replacing_success) {
      return false;
    }
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
