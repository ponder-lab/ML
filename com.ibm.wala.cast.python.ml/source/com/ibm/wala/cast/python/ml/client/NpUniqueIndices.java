package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for the index, inverse, and counts outputs of {@code np.unique} (slots 1&ndash;3 of the
 * flag-form tuple): positional indices into (or occurrence counts over) the input, always {@code
 * int64} regardless of the input dtype, rank 1 with a statically unresolvable length. The corpus
 * idiom is label densification via the inverse output, {@code _, _, y = np.unique(y,
 * return_index=True, return_inverse=True)}. See <a
 * href="https://github.com/wala/ML/issues/799">wala/ML#799</a>.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpUniqueIndices extends TensorGenerator {

  public NpUniqueIndices(PointsToSetVariable source) {
    super(source);
  }

  public NpUniqueIndices(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // Rank 1; the length (the number of uniques, the input's size, or the count vector's length)
    // is a fixed runtime integer the analysis cannot compute, so the axis is Unresolved rather
    // than Dynamic (wala/ML#721).
    return Set.of(List.of(UnresolvedDim.INSTANCE));
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(INT64);
  }

  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
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
