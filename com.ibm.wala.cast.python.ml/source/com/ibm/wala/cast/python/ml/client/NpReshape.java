package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Modeling of the function-style {@code numpy.reshape(x, shape)} call. Semantically identical to
 * {@code tf.reshape(tensor, shape)} for shape/dtype inference: the input tensor is positional arg 0
 * and the target shape is positional arg 1 (both after {@code self} in the XML summary). Extends
 * {@link Reshape} to reuse the {@code -1} inference logic, dtype passthrough, and parameter
 * positions. The distinct class is a dispatch marker so that {@link TensorGeneratorFactory} can
 * route based on the declaring class {@code Lnumpy/reshape} (vs {@code
 * Ltensorflow/functions/reshape} / {@code tensorflow/python/ops/array_ops/reshape}).
 *
 * <p>Overrides {@link #getShapes} to also accept a bare integer as the {@code shape} argument
 * &mdash; {@code np.reshape(y, -1)} (and the equivalent parenthesised form {@code np.reshape(y,
 * (-1))}, which is just {@code -1}, not a 1-tuple) is idiomatic in numpy code but not recognised by
 * the list/tuple-based {@link #getShapesFromShapeArgument} helper. Treats a scalar int as a
 * single-element shape {@code [n]} (or {@code [-1]}) before delegating to the super-class for the
 * rest of the refinement.
 *
 * <p>Counterpart to the method-form {@link NdarrayReshape} ({@code x.reshape(shape)}) and to {@link
 * NpArray}'s function-form modeling. See wala/ML#410.
 */
public class NpReshape extends Reshape {

  public NpReshape(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Resolves the output shapes. Handles the scalar-integer form of the {@code shape} argument
   * ({@code np.reshape(x, n)} or {@code np.reshape(x, -1)}) by synthesising a one-element shape;
   * for list/tuple {@code shape} args, delegates to {@link Reshape#getShapes}.
   *
   * @param builder The propagation call graph builder.
   * @return The set of possible output shapes, refined for {@code -1} against the input tensor's
   *     total element count where possible.
   */
  @Override
  public Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> scalar = shapesFromScalarArgument(builder);
    if (scalar != null) return scalar;
    return super.getShapes(builder);
  }

  /**
   * If the {@code shape} argument's points-to set consists of integer {@link ConstantKey}s, treats
   * each as a one-element shape and applies the same {@code -1} inference rule as {@link Reshape}
   * (total-elements / product-of-known-dims). Returns {@code null} if the PTS is empty or contains
   * any non-integer, letting the list/tuple path in {@link Reshape#getShapes} take over.
   *
   * @param builder The propagation call graph builder.
   * @return The inferred single-dim shapes, or {@code null} if the scalar form doesn't apply.
   */
  private Set<List<Dimension<?>>> shapesFromScalarArgument(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> shapePts =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());
    if (shapePts == null || shapePts.isEmpty()) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (InstanceKey ik : shapePts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (!(value instanceof Number)) return null;
      int dim = ((Number) value).intValue();

      if (dim == -1) {
        Set<List<Dimension<?>>> inputShapes = this.getDefaultShapes(builder);
        if (inputShapes != null && !inputShapes.isEmpty()) {
          for (List<Dimension<?>> inputShape : inputShapes) {
            long total = 1;
            boolean known = true;
            for (Dimension<?> d : inputShape) {
              if (d instanceof NumericDim) total *= ((NumericDim) d).value();
              else {
                known = false;
                break;
              }
            }
            List<Dimension<?>> out = new ArrayList<>(1);
            out.add(known ? new NumericDim((int) total) : new SymbolicDim("?"));
            ret.add(out);
          }
        } else {
          List<Dimension<?>> out = new ArrayList<>(1);
          out.add(new SymbolicDim("?"));
          ret.add(out);
        }
      } else {
        List<Dimension<?>> out = new ArrayList<>(1);
        out.add(new NumericDim(dim));
        ret.add(out);
      }
    }
    return ret.isEmpty() ? null : ret;
  }
}
