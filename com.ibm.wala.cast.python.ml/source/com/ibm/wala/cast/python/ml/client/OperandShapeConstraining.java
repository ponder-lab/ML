package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.List;
import java.util.Map;

/**
 * A generator whose operation statically constrains its operands' shapes (wala/ML#704,
 * wala/ML#734): the einsum equation fixes each operand's rank and proves shared-label extents; a
 * constant transpose permutation fixes its input's rank. The engine refines — rather than pins —
 * each constrained operand destination with the proven shape, so unknown-shape members recover the
 * proven axes while concrete members pass through untouched.
 */
public interface OperandShapeConstraining {

  /**
   * Derives, for each operand whose own shape does not resolve, the shape the operation proves for
   * it. An axis the operation leaves unconstrained carries {@link
   * com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim} (wala/ML#721).
   *
   * @param builder The propagation call graph builder.
   * @return A map from each caller-side operand's {@link PointerKey} to the shape the operation
   *     proves for it; empty when nothing is proven. An operand whose call sites prove disagreeing
   *     constraints is omitted.
   */
  Map<PointerKey, List<Dimension<?>>> getOperandShapeConstraints(
      PropagationCallGraphBuilder builder);
}
