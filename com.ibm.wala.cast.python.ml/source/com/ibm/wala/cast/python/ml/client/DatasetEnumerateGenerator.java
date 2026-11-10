package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
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
 * A generator for tensors created by {@code tf.data.Dataset.enumerate}. The element is the tuple
 * {@code (index, element)}: an int64 scalar index at position 0 and the receiver's element at
 * position 1, so a path below {@code [1]} reads the receiver's element at the rest of the path
 * (wala/ML#1190).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetEnumerateGenerator extends DatasetGenerator {

  protected DatasetEnumerateGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetEnumerateGenerator(CGNode node) {
    super(node);
  }

  /** The type of a tensor of unknown shape and dtype, read where a receiver does not resolve. */
  private static final TensorType UNKNOWN_TENSOR =
      new TensorType(DType.UNKNOWN.name().toLowerCase(Locale.ROOT), null);

  @Override
  protected Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<TensorType> types = HashSetFactory.make();

    // Add the index type, a scalar.
    types.add(new TensorType(this.getIndexDType().name().toLowerCase(Locale.ROOT), INDEX_SHAPE));

    // Add the underlying datasets' types
    Receivers receivers = this.receivers(builder);
    for (TensorGenerator underlying : receivers.generators())
      types.addAll(underlying.getTensorTypes(builder));
    if (receivers.partial()) types.add(UNKNOWN_TENSOR);

    return types;
  }

  /** The position of the index in the element. */
  private static final int INDEX_POSITION = 0;

  /** The position of the receiver's element in the element. */
  private static final int ELEMENT_POSITION = 1;

  /** The index's shape, a scalar. */
  private static final List<Dimension<?>> INDEX_SHAPE = Collections.emptyList();

  /**
   * The dtype of the scalar at position 0 of the element.
   *
   * @return {@link DType#INT64}, an index's.
   */
  protected DType getIndexDType() {
    return DType.INT64;
  }

  /**
   * A generator for the scalar at position 0 of an element of this dataset that is not copied.
   *
   * @param source The element's variable.
   * @return The index's generator.
   */
  public TensorGenerator getIndexGenerator(PointsToSetVariable source) {
    return new EnumerateIndexGenerator(source);
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The receiver's element is the join of every dataset the receiver may be: each
   *     dataset's types are read on their own, so a shape stays paired with its own dtype.
   */
  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index != ELEMENT_POSITION) return super.getTensorTypesForIndex(builder, index);
    Receivers receivers = this.receivers(builder);
    if (receivers.isEmpty()) return super.getTensorTypesForIndex(builder, index);
    Set<TensorType> ret = HashSetFactory.make();
    for (TensorGenerator underlying : receivers.generators())
      ret.addAll(underlying.getTensorTypes(builder));
    if (receivers.partial()) ret.add(UNKNOWN_TENSOR);
    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index == INDEX_POSITION) return Set.of(INDEX_SHAPE);
    if (index == ELEMENT_POSITION) {
      Receivers receivers = this.receivers(builder);
      if (receivers.isEmpty() || receivers.partial()) return null;
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (TensorGenerator underlying : receivers.generators()) {
        Set<List<Dimension<?>>> shapes = underlying.getShapes(builder);
        if (shapes == null) return null;
        ret.addAll(shapes);
      }
      return ret;
    }
    return Collections.emptySet();
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index == INDEX_POSITION) return EnumSet.of(this.getIndexDType());
    if (index == ELEMENT_POSITION) {
      Receivers receivers = this.receivers(builder);
      Set<DType> ret = EnumSet.noneOf(DType.class);
      for (TensorGenerator underlying : receivers.generators())
        ret.addAll(underlying.getDTypes(builder));
      if (receivers.partial() || ret.isEmpty()) ret.add(DType.UNKNOWN);
      return ret;
    }
    return Collections.emptySet();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A path below the receiver's element resolves where some dataset the receiver may be
   *     resolves its rest, as an iterator's element over several datasets does.
   */
  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1)
      return path.get(0) instanceof Integer index
          && (index == INDEX_POSITION || index == ELEMENT_POSITION);
    if (!isElementPosition(path.get(0))) return false;
    List<Object> rest = path.subList(1, path.size());
    for (TensorGenerator underlying : this.receivers(builder).generators())
      if (underlying instanceof TupleElementProvider tep && tep.resolvesPath(builder, rest))
        return true;
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Below the receiver's element, there is a structure only where each dataset the
   *     receiver may be has one, as for any transformation of several datasets.
   */
  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.isEmpty() || !isElementPosition(path.get(0))) return false;
    Receivers receivers = this.receivers(builder);
    if (receivers.isEmpty() || receivers.partial()) return false;
    List<Object> rest = path.subList(1, path.size());
    for (TensorGenerator underlying : receivers.generators())
      if (!(underlying instanceof TupleElementProvider tep && tep.isStructureAt(builder, rest)))
        return false;
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Below the receiver's element, a length is declared only where each dataset the
   *     receiver may be declares the same one.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.isEmpty()) return OptionalInt.of(2);
    if (!isElementPosition(path.get(0))) return OptionalInt.empty();
    Receivers receivers = this.receivers(builder);
    if (receivers.isEmpty() || receivers.partial()) return OptionalInt.empty();
    List<Object> rest = path.subList(1, path.size());
    int ret = -1;
    for (TensorGenerator underlying : receivers.generators()) {
      if (!(underlying instanceof TupleElementProvider tep)) return OptionalInt.empty();
      OptionalInt arity = tep.declaredArity(builder, rest);
      if (arity.isEmpty() || (ret >= 0 && ret != arity.getAsInt())) return OptionalInt.empty();
      ret = arity.getAsInt();
    }
    return OptionalInt.of(ret);
  }

  @Override
  public Set<TensorType> getTensorTypesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getTensorTypesForIndex(builder, index);
    if (!isElementPosition(path.get(0))) return Collections.emptySet();
    Receivers receivers = this.receivers(builder);
    if (receivers.isEmpty()) return Set.of(UNKNOWN_TENSOR);
    List<Object> rest = path.subList(1, path.size());
    Set<TensorType> ret = HashSetFactory.make();
    for (TensorGenerator underlying : receivers.generators())
      if (underlying instanceof TupleElementProvider tep)
        ret.addAll(tep.getTensorTypesForPath(builder, rest));
      else ret.add(UNKNOWN_TENSOR);
    if (receivers.partial()) ret.add(UNKNOWN_TENSOR);
    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getShapesForIndex(builder, index);
    if (!isElementPosition(path.get(0))) return Collections.emptySet();
    Receivers receivers = this.receivers(builder);
    if (receivers.isEmpty() || receivers.partial()) return null;
    List<Object> rest = path.subList(1, path.size());
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TensorGenerator underlying : receivers.generators()) {
      if (!(underlying instanceof TupleElementProvider tep)) return null;
      Set<List<Dimension<?>>> shapes = tep.getShapesForPath(builder, rest);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (path.size() == 1 && path.get(0) instanceof Integer index)
      return this.getDTypesForIndex(builder, index);
    if (!isElementPosition(path.get(0))) return Collections.emptySet();
    Receivers receivers = this.receivers(builder);
    List<Object> rest = path.subList(1, path.size());
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TensorGenerator underlying : receivers.generators())
      if (underlying instanceof TupleElementProvider tep)
        ret.addAll(tep.getDTypesForPath(builder, rest));
      else ret.add(DType.UNKNOWN);
    if (receivers.partial() || ret.isEmpty()) ret.add(DType.UNKNOWN);
    return ret;
  }

  /**
   * Whether a selector is the position of the receiver's element.
   *
   * @param selector A path's selector.
   * @return {@code true} iff it is the integer {@value #ELEMENT_POSITION}.
   */
  private static boolean isElementPosition(Object selector) {
    return selector instanceof Integer index && index == ELEMENT_POSITION;
  }

  /**
   * The datasets being enumerated (wala/ML#1204): a receiver may be several datasets, as {@code (a
   * if flag else b).enumerate()}'s is, and its element is then the join of theirs.
   *
   * @param generators The datasets' generators, each spread over the datasets its own receivers may
   *     be ({@link DatasetGenerator#spread}), without duplicates.
   * @param partial Whether some dataset the receiver may be does not resolve, so the element may
   *     also be one of unknown type.
   */
  record Receivers(List<TensorGenerator> generators, boolean partial) {

    /**
     * Whether no dataset the receiver may be is known.
     *
     * @return {@code true} iff no generator resolves and none is unresolved.
     */
    boolean isEmpty() {
      return this.generators.isEmpty() && !this.partial;
    }
  }

  /**
   * Resolves every dataset the receiver may be, where {@link #getUnderlyingGenerator} resolves the
   * first (wala/ML#1204).
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The datasets being enumerated.
   */
  Receivers receivers(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    List<TensorGenerator> generators = new ArrayList<>();
    boolean partial = false;
    if (receiverPTS != null)
      for (InstanceKey valueIK : receiverPTS) {
        TensorGenerator underlying = underlyingGenerator(builder, valueIK);
        if (underlying == null) {
          // A member that is no dataset allocation, as the import-block class singleton is, is not
          // a dataset being enumerated.
          if (getAllocationSiteInNode(valueIK) != null) partial = true;
          continue;
        }
        List<TensorGenerator> parts =
            DatasetGenerator.spread(builder, underlying, new ArrayList<>());
        for (TensorGenerator part : parts == null ? List.of(underlying) : parts)
          if (!generators.contains(part)) generators.add(part);
      }
    return new Receivers(generators, partial);
  }

  /**
   * Retrieves the generator for the dataset being enumerated: the first that resolves of the
   * datasets the receiver may be. {@link #receivers} resolves them all.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The underlying dataset generator, or null if not found.
   */
  public TensorGenerator getUnderlyingGenerator(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null)
      for (InstanceKey valueIK : receiverPTS) {
        TensorGenerator underlying = underlyingGenerator(builder, valueIK);
        if (underlying != null) return underlying;
      }
    return null;
  }

  /**
   * The generator of one dataset the receiver may be.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param valueIK A member of the receiver's points-to set.
   * @return The dataset's generator, or {@code null} if the member is not an allocation or its
   *     defining value is not found.
   */
  private TensorGenerator underlyingGenerator(
      PropagationCallGraphBuilder builder, InstanceKey valueIK) {
    AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
    if (asin == null) return null;
    // A mapped dataset is read through the mapped element, as a transformation's receiver is
    // (wala/ML#649).
    if (asin.getNode()
        .getMethod()
        .getDeclaringClass()
        .getReference()
        .equals(TensorFlowTypes.DATASET_MAP_TYPE))
      return DatasetGenerator.receiverGenerator(builder, valueIK);
    int vn = findDefinition(asin.getNode(), asin);
    if (vn <= 0) return null;
    PointerKey pk =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(asin.getNode(), vn);
    if (!builder.getPropagationSystem().isImplicit(pk))
      return TensorGeneratorFactory.getGenerator(
          builder.getPropagationSystem().findOrCreatePointsToSet(pk), builder);
    return createManualGenerator(asin.getNode(), builder);
  }
}
