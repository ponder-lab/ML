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
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * Member-wise record view (wala/ML#718): the product shape composes per operand-member pair, and
   * either operand's unknown remainder rides through.
   *
   * @param builder The propagation call graph builder.
   * @return The composed result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    ShapeResult aShapes = shapeResultOfArg(builder, 0, "a");
    ShapeResult bShapes = shapeResultOfArg(builder, 1, "b");
    if (aShapes.members().isEmpty() || bShapes.members().isEmpty()) return ShapeResult.unknown();

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> aShape : aShapes.members()) {
      for (List<Dimension<?>> bShape : bShapes.members()) {
        if (aShape.size() >= 2 && bShape.size() >= 2) {
          // Batched semantics: the leading (batch) dimensions carry through and the trailing two
          // compose as the matrix product, so the rank is preserved rather than collapsing to two
          // (wala/ML#718). The batch dimensions are taken from the higher-rank operand, a sound
          // simplification of TensorFlow's batch broadcasting for the common equal-rank case.
          List<Dimension<?>> batched = aShape.size() >= bShape.size() ? aShape : bShape;
          List<Dimension<?>> newShape = new ArrayList<>(batched.subList(0, batched.size() - 2));
          newShape.add(aShape.get(aShape.size() - 2));
          newShape.add(bShape.get(bShape.size() - 1));
          ret.add(newShape);
        }
      }
    }
    return ret.isEmpty()
        ? ShapeResult.unknown()
        : new ShapeResult(ret, aShapes.hasUnknown() || bShapes.hasUnknown());
  }

  /**
   * Routes the output-shape resolution through {@link #getDefaultShapeResult} (this generator has
   * no {@code shape} parameter), so partial results cross the generator boundary (wala/ML#718).
   *
   * @param builder The propagation call graph builder.
   * @return The resolution result.
   */
  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder);
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
  private ShapeResult shapeResultOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    ShapeResult fromValue = ShapeResult.unknown();
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      fromValue = this.getShapeResultOfValue(builder, pts, true);
      if (!fromValue.members().isEmpty() && !fromValue.hasUnknown()) return fromValue;
    }
    // An incomplete points-to union commonly reflects context collapse, so the per-context
    // caller walk is preferred; the union's resolvable members are the floor when the walk
    // fails (wala/ML#716, wala/ML#718).
    ShapeResult viaCallers = this.getArgumentShapeResultViaCallers(builder, paramPos, paramName);
    if (!viaCallers.members().isEmpty()) return viaCallers;
    return fromValue.members().isEmpty() ? viaCallers : fromValue;
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
      // A non-empty result that is only ⊤ does not short-circuit: the synthetic `matmul.do` param
      // PTS is often context-collapsed to a union of unrelated bare tensors (all `UNKNOWN`), so
      // fall through to the per-context caller-walk, which resolves the actual argument at each
      // call site. See wala/ML#570.
      if (dtypes != null && !dtypes.isEmpty() && !dtypes.equals(EnumSet.of(DType.UNKNOWN)))
        return dtypes;
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
