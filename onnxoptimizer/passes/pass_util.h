/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnxoptimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

template <typename T>
T AddYIfNegative(T x, T y) {
  return x < 0 ? x + y : x;
}

inline bool IsConstantTensor(const Value* v) {
  auto* graph = v->owningGraph();
  return v->node()->kind() == kConstant || graph->is_constant_initializer(v);
}

inline bool IsConstantTensor(const Node* n, size_t which_input) {
  ONNX_ASSERT(which_input < n->inputs().size());
  return IsConstantTensor(n->input(which_input));
}

inline const Tensor* FetchConstantTensor(const Value* v) {
  const uint32_t kind = v->node()->kind();
  auto* graph = v->owningGraph();
  if (kind == kConstant) {
    return &v->node()->t(kvalue);
  } else if (graph->is_constant_initializer(v)) {
    return &*graph->getInitializer(v->uniqueName());
  } else {
    return nullptr;
  }
}

// fetch the only element when the tensor is a scalar or a tensor that only has
// a element
template <typename T>
bool FetchSoleValueOfTensor(const Value* t, T& val);

// FetchSoleIntValueOfTensor is a wraper that fetchs int value(INT32 or INT64)
// easier. E.g: get axis from axes tensor
bool FetchSoleIntValueOfTensor(const Value* t, int64_t& val);

// FetchIntsOfTensor is a wraper that fetchs int value tensor(INT32 or INT64)
// easier. E.g: get data from axes tensor
bool FetchIntsOfTensor(const Value* t, std::vector<int64_t>& vals);

bool FetchIntsOfTensor(const Tensor* t, std::vector<int64_t>& vals);

inline bool CheckKind(const Value* v, const Symbol& symbol) {
  return v->node()->kind() == symbol;
}

inline bool CheckKind(const Value* v, const char* symbol) {
  return CheckKind(v, Symbol(symbol));
}

inline bool CheckKind(const Node* n, const Symbol& symbol) {
  return n->kind() == symbol;
}

inline bool CheckKind(const Node* n, const char* symbol) {
  return CheckKind(n, Symbol(symbol));
}

inline bool CheckKind(const Node* n, const Symbol& lhs_type,
                      const Symbol& rhs_type) {
  return n->inputs().size() == 2 && CheckKind(n->input(0), lhs_type) &&
         CheckKind(n->input(1), rhs_type);
}

inline bool CheckKind(const Node* n, const Symbol& n_type,
                      const Symbol& lhs_type, const Symbol& rhs_type) {
  return CheckKind(n, n_type) && CheckKind(n, lhs_type, rhs_type);
}

inline std::pair<int64_t, int64_t> FetchStartAndEndAttrOfShape(
    const Node* shape, const int64_t rank) {
  ONNX_ASSERT(CheckKind(shape, "Shape"));

  const int64_t start = AddYIfNegative<int64_t>(
      shape->hasAttribute(Symbol("start")) ? shape->i(Symbol("start")) : 0,
      rank);
  const int64_t end = AddYIfNegative<int64_t>(
      shape->hasAttribute(Symbol("end")) ? shape->i(Symbol("end")) : rank,
      rank);
  return {start, end};
}

inline std::pair<int64_t, int64_t> FetchStartAndEndAttrOfShape(
    const Node* shape) {
  ONNX_ASSERT(CheckKind(shape, "Shape") && shape->input()->has_sizes());
  return FetchStartAndEndAttrOfShape(shape, shape->input()->sizes().size());
}

inline Node* FetchParrentNode(Node* node, const Symbol& type) {
  for (auto* input : node->inputs()) {
    if (input->node()->kind() == type) {
      return input->node();
    }
  }
  return nullptr;
}

inline bool IsIntersection(const std::vector<int64_t>& v1,
                      const std::vector<int64_t>& v2) {
  std::vector<int64_t> intersect;
  std::set<int64_t> s1(v1.begin(), v1.end());
  std::set<int64_t> s2(v2.begin(), v2.end());
  std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
                        std::back_inserter(intersect));
  return !intersect.empty();
}

}  // namespace optimization
}  // namespace ONNX_NAMESPACE