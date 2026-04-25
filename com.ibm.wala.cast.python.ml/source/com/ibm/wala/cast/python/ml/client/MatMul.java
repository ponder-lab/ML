package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.matmul}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/matmul">tf.matmul</a>
 */
public class MatMul extends TensorGenerator {

  public MatMul(PointsToSetVariable source) {
    super(source);
  }

  public MatMul(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> aShapes = shapesOfArg(builder, 0, "a");
    Set<List<Dimension<?>>> bShapes = shapesOfArg(builder, 1, "b");

    if (aShapes == null || aShapes.isEmpty() || bShapes == null || bShapes.isEmpty()) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> aShape : aShapes) {
      for (List<Dimension<?>> bShape : bShapes) {
        if (aShape.size() >= 2 && bShape.size() >= 2) {
          List<Dimension<?>> newShape = new ArrayList<>();
          newShape.add(aShape.get(aShape.size() - 2));
          newShape.add(bShape.get(bShape.size() - 1));
          ret.add(newShape);
        }
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> aDTypes = dtypesOfArg(builder, 0, "a");
    return aDTypes == null || aDTypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : aDTypes;
  }

  /**
   * Resolves shapes of the matmul arg at the given position. Tries the summary-local PTS first (via
   * {@link #getShapesOfValue}); if empty, falls back to {@link #getArgumentShapesViaCallers} which
   * walks the caller's call site for concrete arg shapes. The fallback is needed because the matmul
   * generator's node is the synthetic {@code matmul.do()} summary and the summary's param-slot PTS
   * is often empty even when the caller passes a concretely-shaped tensor.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg (0 for {@code a}, 1 for {@code b}).
   * @param paramName The keyword parameter name.
   * @return The resolved shapes, or {@code null} if neither path recovers a concrete shape.
   */
  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
  }

  /**
   * Dtype counterpart to {@link #shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved dtypes, or {@code null} if neither path recovers.
   */
  private Set<DType> dtypesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    return this.getArgumentDTypesViaCallers(builder, paramPos, paramName);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
