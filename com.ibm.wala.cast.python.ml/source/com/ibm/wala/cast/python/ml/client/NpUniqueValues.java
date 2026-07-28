package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for the unique-values output of {@code np.unique} (slot 0 of the flag-form tuple): the
 * sorted unique elements of the input, dtype preserved from {@code ar}, flattened to rank 1 with a
 * statically unresolvable length. See <a
 * href="https://github.com/wala/ML/issues/799">wala/ML#799</a>.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpUniqueValues extends TensorGenerator {

  public NpUniqueValues(PointsToSetVariable source) {
    super(source);
  }

  public NpUniqueValues(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // `np.unique` flattens: rank 1, with the number of uniques a fixed runtime integer the
    // analysis cannot compute, so the axis is Unresolved rather than Dynamic (wala/ML#721).
    return Set.of(List.of(UnresolvedDim.INSTANCE));
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> arPts = getArgumentPointsToSet(builder, 0, "ar");
    Set<DType> preserved = getDTypesOfValue(builder, arPts);
    return preserved == null || preserved.isEmpty() ? EnumSet.of(DType.UNKNOWN) : preserved;
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
