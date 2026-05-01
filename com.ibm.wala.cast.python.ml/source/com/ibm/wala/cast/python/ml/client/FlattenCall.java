package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.CompoundDim;
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
 * Generator for the {@code __call__} on a {@code tf.keras.layers.Flatten} instance. Given an input
 * of shape {@code (B, d1, d2, ..., dN)}, returns a tensor of shape {@code (B, d1*d2*...*dN)}; dtype
 * passes through unchanged.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten">tf.keras.layers.Flatten</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class FlattenCall extends TensorGenerator {

  /**
   * Constructs a {@code FlattenCall} from a caller-side {@link PointsToSetVariable} (the result of
   * the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code tf.keras.layers.Flatten} instance.
   */
  public FlattenCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code FlattenCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public FlattenCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by reading the {@code inputs} argument's shapes and collapsing all
   * trailing dimensions into a single {@link CompoundDim}.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, one per input shape of rank &ge; 2, or {@code null} if the
   *     input has no known shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 1, "inputs");
    if (inputPts == null || inputPts.isEmpty()) return null;
    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);
    if (inputShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.size() >= 2) {
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(inputShape.get(0));
        List<Dimension<?>> remaining = new ArrayList<>(inputShape.subList(1, inputShape.size()));
        newShape.add(new CompoundDim(remaining));
        ret.add(newShape);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Resolves the output dtypes by passing the {@code inputs} argument's dtypes through unchanged.
   *
   * @param builder The propagation call graph builder.
   * @return The set of dtypes observed on the input, or {@code {UNKNOWN\}} if none can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 1, "inputs");
    if (inputPts == null || inputPts.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = this.getDTypesOfValue(builder, inputPts);
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; {@code Flatten.__call__} takes no explicit shape
   *     parameter.
   */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; {@code Flatten.__call__} takes no explicit shape parameter.
   */
  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; {@code Flatten.__call__} takes no explicit dtype
   *     parameter.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; {@code Flatten.__call__} takes no explicit dtype parameter.
   */
  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
