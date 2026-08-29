package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for the {@code __call__} on a Keras layer whose output has the same shape and dtype as
 * its input: normalization, an activation applied as a layer, and masking. Both axes pass through
 * unchanged.
 *
 * <p>An unmodeled layer class drops the shape for everything downstream of it, because its call
 * produces no output shape and the rest of the chain inherits ⊤ (<a
 * href="https://github.com/wala/ML/issues/840">wala/ML#840</a>). A shape-preserving layer is the
 * most surprising member of that family: nothing about normalizing or applying an activation
 * suggests it should affect the result's type at all, yet one of them sitting mid-chain erases the
 * shape of every value after it.
 *
 * <p>This is deliberately one generator rather than one per class. The classes differ in what they
 * compute and not in how they transform the type, so a per-class generator would be the same body
 * repeated, and a reader comparing them would have to diff them to learn they agree.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ShapePreservingLayerCall extends TensorGenerator {

  /**
   * Constructs a {@code ShapePreservingLayerCall} from a caller-side {@link PointsToSetVariable}
   * (the result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a shape-preserving Keras layer instance.
   */
  public ShapePreservingLayerCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code ShapePreservingLayerCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public ShapePreservingLayerCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by passing the {@code inputs} argument's shapes through unchanged.
   *
   * @param builder The propagation call graph builder.
   * @return The input's shapes, or {@code null} when the input's shape does not resolve, which is ⊤
   *     rather than "not a tensor": the layer still returns a tensor.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The SSA-chain fallback covers an input with dataflow state but no points-to evidence, e.g. a
    // destructured tuple dataset element (wala/ML#855).
    Set<List<Dimension<?>>> inputShapes = this.getArgumentShapesWithFallback(builder, 1, "inputs");
    if (inputShapes == null || inputShapes.isEmpty()) return null;
    // A fresh set rather than the input's own: the XML models this call as `<new>` plus `<return>`,
    // so the result is a distinct allocation, and handing back the input's collection invites the
    // result to be treated as the input rather than as a value that merely agrees with it.
    return HashSetFactory.make(inputShapes);
  }

  /**
   * Resolves the output dtypes by passing the {@code inputs} argument's dtypes through unchanged.
   *
   * @param builder The propagation call graph builder.
   * @return The set of dtypes observed on the input, or {@code {UNKNOWN\}} if none can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // The dtype twin of the shape fallback above (wala/ML#855).
    Set<DType> dtypes = this.getArgumentDTypesWithFallback(builder, 1, "inputs");
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; the call takes no explicit shape parameter.
   */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; the call takes no explicit shape parameter.
   */
  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; the call takes no explicit dtype parameter.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; the call takes no explicit dtype parameter.
   */
  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
