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
 * Generator for the {@code __call__} on a {@code tf.keras.layers.GlobalAveragePooling1D} instance.
 * Given an input of shape {@code (B, steps, features)}, returns a tensor of shape {@code (B,
 * features)} (the temporal axis is averaged away); dtype passes through unchanged. See <a
 * href="https://github.com/wala/ML/issues/670">wala/ML#670</a>.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/GlobalAveragePooling1D">tf.keras.layers.GlobalAveragePooling1D</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class GlobalAveragePooling1DCall extends TensorGenerator {

  /**
   * Constructs a {@code GlobalAveragePooling1DCall} from a caller-side {@link PointsToSetVariable}
   * (the result of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code tf.keras.layers.GlobalAveragePooling1D} instance.
   */
  public GlobalAveragePooling1DCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code GlobalAveragePooling1DCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public GlobalAveragePooling1DCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by reading the {@code inputs} argument's shapes and dropping the
   * temporal (middle) axis of each rank-3 input.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, one per rank-3 input shape, or {@code null} if the input has no
   *     known shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 1, "inputs");
    if (inputPts == null || inputPts.isEmpty()) return null;
    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);
    if (inputShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.size() == 3) {
        List<Dimension<?>> newShape = new ArrayList<>();
        newShape.add(inputShape.get(0));
        newShape.add(inputShape.get(2));
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
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; {@code GlobalAveragePooling1D.__call__} takes no
   *     explicit shape parameter.
   */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; {@code GlobalAveragePooling1D.__call__} takes no explicit shape
   *     parameter.
   */
  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; {@code GlobalAveragePooling1D.__call__} takes no
   *     explicit dtype parameter.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; {@code GlobalAveragePooling1D.__call__} takes no explicit dtype
   *     parameter.
   */
  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
