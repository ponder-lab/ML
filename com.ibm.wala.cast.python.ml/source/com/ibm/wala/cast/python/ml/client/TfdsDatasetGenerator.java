package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for the elements of a dataset that TensorFlow Datasets' {@code tfds.load} returns (<a
 * href="https://github.com/wala/ML/issues/1155">wala/ML#1155</a>). An element's types depend on the
 * named dataset's features, which the program does not state, so each component is a tensor of
 * unknown shape and dtype. With {@code as_supervised=True}, an element is a pair {@code (input,
 * label)}; otherwise it is a dictionary of features, each such a tensor. See <a
 * href="https://www.tensorflow.org/datasets/api_docs/python/tfds/load">tfds.load</a>.
 *
 * <p>Same intrinsic-element-type approach as {@link TFRecordDatasetGenerator}: the element type is
 * a property of the API, with no upstream tensor allocation to peel. The generator is anchored on
 * the {@code TfdsDataset} constructor that the {@code load} summary calls, whose one argument is
 * {@code load}'s {@code as_supervised}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TfdsDatasetGenerator extends DatasetGenerator {

  /** The number of components of a supervised element, {@code (input, label)}. */
  private static final int SUPERVISED_ARITY = 2;

  /** Parameter positions and keyword names for the {@code TfdsDataset} constructor. */
  protected enum Parameters {
    /** Whether an element is an {@code (input, label)} pair. */
    AS_SUPERVISED;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public TfdsDatasetGenerator(PointsToSetVariable source) {
    super(source);
  }

  public TfdsDatasetGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote An element is a pair when {@code as_supervised} may be the literal {@code True}.
   *     Absent or false, it is a dictionary of features. A non-literal {@code as_supervised}, such
   *     as a flag, also reads as a dictionary: its points-to set holds at most a {@code bool}'s
   *     type-level key, which argument reads drop (wala/ML#1148), so it cannot be told from an
   *     omitted argument.
   */
  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(
            builder, Parameters.AS_SUPERVISED.getIndex(), Parameters.AS_SUPERVISED.getName());
    if (pts == null) return false;
    for (InstanceKey ik : pts)
      if (ik instanceof ConstantKey<?> constant && Boolean.TRUE.equals(constant.getValue()))
        return true;
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A pair's members by index, or a dictionary's features by key, one level deep.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() != 1) return false;
    Object selector = path.get(0);
    if (this.yieldsTuple(builder))
      return selector instanceof Integer index && index >= 0 && index < SUPERVISED_ARITY;
    return selector instanceof String;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Every component's shape is unknown.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Every component's dtype is unknown.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return null;
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return Set.of(new TensorType(DType.UNKNOWN.name().toLowerCase(Locale.ROOT), null));
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    return null;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    return EnumSet.of(DType.UNKNOWN);
  }
}
