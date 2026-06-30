package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for the SparseTensor a {@code tf.io.VarLenFeature(dtype)} represents: when a
 * variable-length feature is parsed (e.g. by {@code tf.io.parse_single_example}) it yields a {@code
 * tf.sparse.SparseTensor} whose values carry the feature's dtype and whose dense shape is dynamic
 * (⊤). Modeling it as that SparseTensor lets {@code parse_single_example} (a pass-through of its
 * feature dict) and a downstream {@code tf.sparse.to_dense} type the parsed value (wala/ML#645).
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/io/VarLenFeature">tf.io.VarLenFeature</a>.
 */
public class VarLenFeature extends TensorTypeAllocator {

  @Override
  protected boolean producesSparseTensor() {
    return true;
  }

  protected enum Parameters {
    DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public VarLenFeature(PointsToSetVariable source) {
    super(source);
  }

  /** Variable-length: the dense shape is dynamic (⊤). */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  /** The dtype comes from the {@code dtype} argument. */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, Parameters.DTYPE.getIndex(), Parameters.DTYPE.getName());

    if (pointsToSet == null || pointsToSet.isEmpty()) return EnumSet.of(DType.UNKNOWN);

    return this.getDTypesFromDTypeArgument(builder, pointsToSet);
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
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
