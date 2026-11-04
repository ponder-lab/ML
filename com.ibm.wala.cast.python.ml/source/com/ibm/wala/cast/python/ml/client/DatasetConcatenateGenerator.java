package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.concatenate} (wala/ML#1215). The
 * element is either its receiver's element or its argument's, so both are datasets the element
 * follows, and a read of the element joins over them.
 */
public class DatasetConcatenateGenerator extends DatasetGenerator {

  /** The position of the concatenated dataset in a {@code concatenate} call. */
  private static final int DATASET_PARAMETER_POSITION = 0;

  /** The name of the concatenated dataset's parameter. */
  private static final String DATASET_PARAMETER_NAME = "dataset";

  public DatasetConcatenateGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetConcatenateGenerator(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The receiver's datasets and the concatenated dataset's, since the element is either
   *     one's element.
   */
  @Override
  public List<TensorGenerator> getReceiverGenerators(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> receivers = super.getReceiverGenerators(builder);
    if (receivers == null) return null;
    OrdinalSet<InstanceKey> datasets =
        this.getArgumentPointsToSet(builder, DATASET_PARAMETER_POSITION, DATASET_PARAMETER_NAME);
    if (datasets == null || datasets.isEmpty()) return null;
    List<TensorGenerator> ret = new ArrayList<>(receivers);
    for (InstanceKey dataset : datasets) {
      TensorGenerator generator = receiverGenerator(builder, dataset);
      if (generator == null) return null;
      if (!ret.contains(generator)) ret.add(generator);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The element is one of the datasets' elements, unchanged.
   */
  @Override
  public boolean elementFollowsReceiver() {
    return true;
  }

  /**
   * The datasets whose elements this one's is, or {@code null} for a view fixed to one of them,
   * which reads that one alone.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The datasets' generators, or {@code null}.
   */
  private List<TensorGenerator> joined(PropagationCallGraphBuilder builder) {
    return this.getFixedReceiver() != null ? null : this.getReceiverGenerators(builder);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDefaultShapes(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : datasets) {
      Set<List<Dimension<?>>> shapes = dataset.getShapes(builder);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDefaultDTypes(builder);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : datasets) {
      Set<DType> dtypes = dataset.getDTypes(builder);
      if (dtypes == null || dtypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getShapesForPath(builder, List.of(index));
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getDTypesForPath(builder, List.of(index));
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getTensorTypesForPath(builder, List.of(index));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A path resolves where some dataset whose element this one's is resolves it.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.resolvesPath(builder, path);
    for (TensorGenerator dataset : datasets)
      if (dataset instanceof TupleElementProvider tep && tep.resolvesPath(builder, path))
        return true;
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A length is declared only where each dataset declares the same one.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.declaredArity(builder, path);
    int ret = -1;
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof TupleElementProvider tep)) return OptionalInt.empty();
      OptionalInt arity = tep.declaredArity(builder, path);
      if (arity.isEmpty() || (ret >= 0 && ret != arity.getAsInt())) return OptionalInt.empty();
      ret = arity.getAsInt();
    }
    return ret < 0 ? OptionalInt.empty() : OptionalInt.of(ret);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets of their shapes at the path, unknown when any is.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getShapesForPath(builder, path);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof TupleElementProvider tep)) return null;
      Set<List<Dimension<?>>> shapes = tep.getShapesForPath(builder, path);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over the datasets of their dtypes at the path, unknown when any is.
   */
  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDTypesForPath(builder, path);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof TupleElementProvider tep)) return EnumSet.of(DType.UNKNOWN);
      Set<DType> dtypes = tep.getDTypesForPath(builder, path);
      if (dtypes == null || dtypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Each dataset's types at the path, so a shape stays paired with its own dataset's
   *     dtype, and the joined shapes and dtypes crossed when a dataset provides no path read.
   */
  @Override
  public Set<TensorType> getTensorTypesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> datasets = this.joined(builder);
    Set<TensorType> ret = HashSetFactory.make();
    if (datasets != null && datasets.stream().allMatch(TupleElementProvider.class::isInstance)) {
      for (TensorGenerator dataset : datasets)
        ret.addAll(((TupleElementProvider) dataset).getTensorTypesForPath(builder, path));
      return ret;
    }
    Set<List<Dimension<?>>> shapes = this.getShapesForPath(builder, path);
    for (DType dtype : this.getDTypesForPath(builder, path)) {
      String cellType = dtype.name().toLowerCase(Locale.ROOT);
      if (shapes == null) ret.add(new TensorType(cellType, null));
      else for (List<Dimension<?>> shape : shapes) ret.add(new TensorType(cellType, shape));
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A concatenation has as many elements as its datasets together: each size it may have
   *     is the sum of one size of each, and unknown when any dataset's is.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> datasets = this.joined(builder);
    if (datasets == null) return super.getDatasetSizes(builder);
    Set<Long> ret = null;
    for (TensorGenerator dataset : datasets) {
      if (!(dataset instanceof DatasetGenerator generator)) return Collections.emptySet();
      Set<Long> sizes = generator.getDatasetSizes(builder);
      if (sizes == null || sizes.isEmpty()) return Collections.emptySet();
      if (ret == null) ret = sizes;
      else {
        Set<Long> sums = HashSetFactory.make();
        for (Long a : ret) for (Long b : sizes) sums.add(a + b);
        ret = sums;
      }
    }
    return ret == null ? Collections.emptySet() : ret;
  }
}
