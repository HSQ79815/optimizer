/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

/*
before
          X
        /   \
       |    ReduceMean
        \  /
         Sub
        /   \
       |    Pow
       |     |
       |   ReduceMean
       |     |
       |    Add
       |     |
       |    Sqrt
        \   /
         Div
          |
         Mul
          |
         Add
          |
          Y
after:
  Y = LaryerNorm(X)
*/

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnxoptimizer/pass.h"
#include "pass_util.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ReplaceWithLayerNorm final : public PredicateBasedPass {
  explicit ReplaceWithLayerNorm()
      : PredicateBasedPass(PassType::Replace, PassEfficiency::Complete,
                           PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "replace_with_layernorm";
  }

  bool checkKind(Node* node, const Symbol& lhs_type, const Symbol& rhs_type) {
    return node->inputs().size() == 2 && CheckKind(node->input(0), lhs_type) &&
           CheckKind(node->input(1), rhs_type);
  }

  bool checkKind(Node* node, const Symbol& n_type, const Symbol& lhs_type,
                 const Symbol& rhs_type) {
    return CheckKind(node, n_type) && checkKind(node, lhs_type, rhs_type);
  }

  Node* fetchInputNode(Node* node, const Symbol& type) {
    for (auto* input : node->inputs()) {
      if (input->node()->kind() == type) {
        return input->node();
      }
    }
    return nullptr;
  }

  Value* fetchConstantInputValue(Node* node) {
    for (auto* input : node->inputs()) {
      if (input->node()->kind() == kConstant ||
          input->node()->kind() == kParam) {
        return &*input;
      }
    }
    return nullptr;
  }

  bool checkAxes(Node* n) {
    return n->hasAttribute(kaxes) && n->is(kaxes).size() == 1 &&
           n->is(kaxes)[0] == -1;
  }

  bool patternMatchPredicate(Node* node) override {
    if (!checkKind(node, kAdd, kConstant, kMul) &&
        !checkKind(node, kAdd, kMul, kConstant) &&
        !checkKind(node, kAdd, kParam, kMul) &&
        !checkKind(node, kAdd, kMul, kParam)) {
      return false;
    }
    if (FetchConstantTensor(fetchConstantInputValue(node))->elem_type() !=
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return false;
    }
    Node* mul_gamma = fetchInputNode(node, kMul);
    if (!checkKind(mul_gamma, kConstant, kDiv) && !checkKind(mul_gamma, kDiv, kConstant) &&
        !checkKind(mul_gamma, kParam, kDiv) && !checkKind(mul_gamma, kDiv, kParam)) {
      return false;
    }
    if (FetchConstantTensor(fetchConstantInputValue(mul_gamma))->elem_type() !=
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return false;
    }

    Node* div = fetchInputNode(mul_gamma, kDiv);
    if (!checkKind(div, kSub, kSqrt)) {
      return false;
    }

    Node* sub = fetchInputNode(div, kSub);
    Node* add = fetchInputNode(div, kSqrt)->input()->node();

    if (!checkKind(add, kAdd, kConstant, kReduceMean) &&
        !checkKind(add, kAdd, kReduceMean, kConstant) &&
        !checkKind(add, kAdd, kParam, kReduceMean) &&
        !checkKind(add, kAdd, kReduceMean, kParam)) {
      return false;
    }
    Node* reducemean = fetchInputNode(add, kReduceMean);
    Node* pow = reducemean->input()->node();
    if (!checkKind(pow, kPow, kSub, kConstant) &&
            !checkKind(pow, kPow, kSub, kParam) ||
        pow->inputs()[0]->node() != sub) {
      return false;
    }
    if (sub->inputs()[1]->node()->kind() != kReduceMean ||
        sub->inputs()[0]->node() != sub->inputs()[1]->node()->input()->node()) {
      return false;
    }
    {
      // check whether the second operand of power is a scalar 2 or not
      float pow_val;
      if (!FetchSoleValueOfTensor(pow->inputs()[1], pow_val) ||
          pow_val != 2.f) {
        return false;
      }
    }
    {
      // check whether the epsilon is a scalar or not
      float dummy;
      if (!FetchSoleValueOfTensor(fetchConstantInputValue(add), dummy)) {
        return false;
      }
    }
    return checkAxes(reducemean) && checkAxes(sub->inputs()[1]->node());
  }

  bool runTransform(Node* n, Graph& graph,
                    NodeDestroyType& destroy_current) override {
    Node* add_beta = n;
    const Tensor* beta = FetchConstantTensor(fetchConstantInputValue(add_beta));
    Node* mul_gamma = fetchInputNode(add_beta, kMul);
    const Tensor* gamma = FetchConstantTensor(fetchConstantInputValue(mul_gamma));
    Node* div = fetchInputNode(mul_gamma, kDiv);
    Node* sub = fetchInputNode(div, kSub);
    Node* add = fetchInputNode(div, kSqrt)->input()->node();

    float epsilon;
    FetchSoleValueOfTensor(fetchConstantInputValue(add), epsilon);

    Node* layer_norm = graph.create(Symbol("LayerNorm"), 1);
    layer_norm->f_(kepsilon, epsilon);
    layer_norm->t_(kbeta, *beta);
    layer_norm->t_(Symbol("gamma"), *gamma);

    layer_norm->addInput(sub->inputs()[0]);
    layer_norm->insertBefore(n);

    const std::string custom_domain("custom");
    const int64_t version = 1;
    layer_norm->setDomain(custom_domain);

    auto& opset_versions = graph.opset_versions_mutable();
    if (std::none_of(opset_versions.cbegin(), opset_versions.cend(),
                     [&custom_domain](const OpSetID& opset) {
                       return opset.domain() == custom_domain;
                     })) {
      opset_versions.emplace_back(OpSetID(custom_domain, version));
    }

    const bool replacing_success = tryReplacingAllUsesWith(n, layer_norm);
    if (!replacing_success) {
      return false;
    }
    if(layer_norm->input()->has_sizes() && !layer_norm->output()->has_sizes()){
      layer_norm->output()->setSizes(layer_norm->input()->sizes());
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

}  // namespace optimization
}  // namespace ONNX_NAMESPACE
