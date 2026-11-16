package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**
 * A generator representing a specific element within a tuple produced by a dataset. This is used
 * when a dataset yields structured elements (like tuples) and we need to track the shape and type
 * of individual components within that structure, delegating the lookup to the underlying generator
 * based on the tuple index.
 */
public class DatasetTupleElementGenerator extends TensorGenerator
    implements DelegatingTensorGenerator {

  /** The generator representing the underlying dataset this element belongs to. */
  private final TupleElementProvider underlying;

  /** The selectors into the element, outermost first (wala/ML#993). */
  private final List<Object> path;

  /**
   * What identifies this element's reads among the resolver's tracked computations, or {@code null}
   * when each read is computed ({@link #sharingReads}).
   */
  private Object sharedReads;

  /** What a read of the element gives. */
  private enum Axis {
    /** Its tensor types. */
    TYPES,
    /** Its shapes. */
    SHAPES,
    /** Its dtypes. */
    DTYPES
  }

  /**
   * The key of an element's read among the resolver's tracked computations.
   *
   * @param element What identifies the element ({@link #sharedReads}).
   * @param axis What the read gives.
   */
  private record SharedRead(Object element, Axis axis) {}

  /**
   * Constructs a new {@code DatasetTupleElementGenerator}.
   *
   * @param source the points-to set variable representing the source of the element
   * @param underlying the generator representing the underlying dataset
   * @param index the index of this element within the tuple
   */
  protected DatasetTupleElementGenerator(
      PointsToSetVariable source, TupleElementProvider underlying, int index) {
    this(source, underlying, List.<Object>of(index));
  }

  /**
   * Constructs a generator for the component a path of selectors reaches within the element
   * (wala/ML#993): integer indexes into tuples and string keys into dicts, outermost first.
   *
   * @param source the points-to set variable representing the source of the read
   * @param underlying the generator representing the underlying dataset
   * @param path the selectors, outermost first; never empty
   */
  protected DatasetTupleElementGenerator(
      PointsToSetVariable source, TupleElementProvider underlying, List<Object> path) {
    super(source);
    this.underlying = underlying;
    this.path = List.copyOf(path);
  }

  /**
   * Shares this element's reads with every other generator given an equal key: the element is then
   * read once for as long as the queries the read made keep their values ({@link
   * WorklistTypeResolver#tracked}), not once per generator. A value that holds a dataset element's
   * leaves gets a generator per leaf and path, and each value holding a leaf reads the same element
   * at the same path, so unshared, the element's datasets were walked once per value and leaf, for
   * each read of the value (wala/ML#1273). The key must determine the underlying generator and the
   * path.
   *
   * @param key The key.
   * @return This generator.
   */
  DatasetTupleElementGenerator sharingReads(Object key) {
    this.sharedReads = key;
    return this;
  }

  /**
   * Makes a read of the element, from a kept value where the reads are shared ({@link
   * #sharingReads}). The read is computed when no analysis resolves, when this generator carries a
   * binding or a composed call's arguments, and when a generator guard is in progress on the
   * thread, since the computation may then answer for the frame it is asked in; a computation that
   * did not answer for itself alone ({@link #answeredForItself}) is not kept.
   *
   * @param <S> The type of the read's value.
   * @param builder The propagation call graph builder.
   * @param axis What the read gives.
   * @param computation The read.
   * @param copy Copies a value, so neither the kept one nor the computed one is handed out.
   * @return The value, in a set of the caller's own, or {@code null} where the read gives that.
   */
  private <S> S read(
      PropagationCallGraphBuilder builder,
      Axis axis,
      Supplier<S> computation,
      UnaryOperator<S> copy) {
    Supplier<S> counted =
        () -> {
          synchronized (READ_COMPUTATIONS) {
            READ_COMPUTATIONS.computeIfAbsent(builder, b -> new AtomicLong()).incrementAndGet();
          }
          return computation.get();
        };
    WorklistTypeResolver engine =
        this.sharedReads == null ? null : WorklistTypeResolver.active(builder);
    if (engine == null
        || this.getValueBinding() != null
        || this.getComposedArguments() != null
        || guardsInProgress()) return counted.get();
    BooleanSupplier storable = answeredForItself();
    @SuppressWarnings("unchecked")
    S kept =
        (S)
            engine.tracked(
                new SharedRead(this.sharedReads, axis),
                () -> {
                  S computed = counted.get();
                  return computed == null ? null : copy.apply(computed);
                },
                storable);
    return kept == null ? null : copy.apply(kept);
  }

  /**
   * How many reads of an element were computed per builder, shared or not, for tests that an
   * element is not read once per value holding it. The counts hold nothing of the builder, so its
   * entry expires with it.
   */
  private static final Map<PropagationCallGraphBuilder, AtomicLong> READ_COMPUTATIONS =
      new WeakHashMap<>();

  /**
   * How many reads of an element were computed for a builder, whether the read is shared ({@link
   * #sharingReads}) or not; a read answered from a kept value is not counted.
   *
   * @param builder The builder.
   * @return The count.
   */
  static long readComputations(PropagationCallGraphBuilder builder) {
    synchronized (READ_COMPUTATIONS) {
      AtomicLong count = READ_COMPUTATIONS.get(builder);
      return count == null ? 0 : count.get();
    }
  }

  /**
   * The selectors into the element this generator reads, outermost first.
   *
   * @return the path; never empty
   */
  public List<Object> getPath() {
    return this.path;
  }

  /** Whether the path is a single integer index into a tuple element, the original form. */
  private boolean isIndex() {
    return this.path.size() == 1 && this.path.get(0) instanceof Integer;
  }

  private int index() {
    return (Integer) this.path.get(0);
  }

  /**
   * Constructs a new {@code DatasetTupleElementGenerator} anchored on a call-graph node
   * (manual-generator path, used when no points-to source is available — e.g., producer delegation
   * for a tuple-element allocation inside a synthetic {@code do()} body; wala/ML#830).
   *
   * @param node the CG node the generator is anchored at
   * @param underlying the generator representing the underlying dataset
   * @param index the index of this element within the tuple
   */
  protected DatasetTupleElementGenerator(CGNode node, TupleElementProvider underlying, int index) {
    super(node);
    this.underlying = underlying;
    this.path = List.of(index);
  }

  /**
   * Retrieves the underlying generator.
   *
   * @return the generator representing the underlying dataset
   */
  @Override
  public TensorGenerator getUnderlying() {
    return (TensorGenerator) underlying;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The tuple index: two positions of one batch tuple share the allocating {@code do()}
   *     node as their manual anchor, so class plus anchor alone would key both positions' memoized
   *     results (and the same-operation guard) identically, replaying one position's shapes for the
   *     other (wala/ML#830).
   */
  @Override
  protected Object operationDiscriminator() {
    return this.path;
  }

  @Override
  public String toString() {
    return "DatasetTupleElementGenerator(" + underlying + ", path=" + path + ")";
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation delegates to the underlying dataset generator for the specific
   *     tuple index.
   */
  @Override
  protected Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return this.<Set<TensorType>>read(
          builder,
          Axis.TYPES,
          () ->
              isIndex()
                  ? underlying.getTensorTypesForIndex(builder, index())
                  : underlying.getTensorTypesForPath(builder, path),
          HashSetFactory::make);
    }
    return super.getTensorTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation delegates to the underlying dataset generator for the specific
   *     tuple index.
   */
  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return this.<Set<List<Dimension<?>>>>read(
          builder,
          Axis.SHAPES,
          () ->
              isIndex()
                  ? underlying.getShapesForIndex(builder, index())
                  : underlying.getShapesForPath(builder, path),
          HashSetFactory::make);
    }
    return super.getShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation delegates to the underlying dataset generator for the specific
   *     tuple index.
   */
  @Override
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    if (underlying != null) {
      return this.<Set<DType>>read(
          builder,
          Axis.DTYPES,
          () ->
              isIndex()
                  ? underlying.getDTypesForIndex(builder, index())
                  : underlying.getDTypesForPath(builder, path),
          dtypes -> {
            Set<DType> ret = EnumSet.noneOf(DType.class);
            ret.addAll(dtypes);
            return ret;
          });
    }
    return super.getDTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation returns {@code null} since properties are fully delegated and the
   *     shape cannot be determined here.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return null;
  }

  @Override
  protected int getShapeParameterPosition() {
    return -1;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote This implementation returns {@link DType#UNKNOWN} since properties are fully
   *     delegated and dtype cannot be determined here.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return -1;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
