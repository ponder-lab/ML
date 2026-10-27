package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
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
 * A generator for tensors created by {@code tf.data.Dataset} transformations.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetGenerator extends TensorGenerator implements TupleElementProvider {

  /**
   * The receiver this view is fixed to ({@link #overReceiver}), or {@code null} to resolve it from
   * the call.
   */
  private TensorGenerator receiver;

  public DatasetGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetGenerator(CGNode node) {
    super(node);
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

  /**
   * By default, dataset transformations inherit the element shapes of the dataset they are called
   * on (the receiver). This method looks up the receiver dataset and infers shapes from it.
   *
   * @param builder the propagation call graph builder used for the analysis
   * @return a set of possible element shapes, or an empty set if unknown
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For dataset transformations, default to shapes of the input dataset (the receiver).
    // The receiver is 'self' (arg0 in IR).
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, receiverPTS);
      // An API-produced dataset receiver's elements are tensors by construction, so an empty
      // resolution here means the upstream element shape did not resolve, not that the elements
      // are non-tensors. The default-mode delegation walk flattens an all-unknown union to ⊥ (the
      // wala/ML#718 resolvable-subset contract keeps the unknown remainder only in exact mode),
      // which without this floor kills the whole transformation chain at its first unresolved
      // hop: ⊥ shapes compose with any resolved dtype to no types at all (wala/ML#776). The
      // modeled-producer gate keeps a receiver with no synthetic-`do` allocation (e.g. the
      // import-block `Dataset` singleton reached by the illegal direct construction in
      // `tf2_test_dataset3.py`) at ⊥.
      if ((shapes == null || shapes.isEmpty()) && hasModeledProducer(receiverPTS)) return null;
      return shapes;
    }
    return null;
  }

  /**
   * Returns whether any member of the given points-to set is allocated in a synthetic {@code do}
   * method, i.e., was produced by a modeled dataset API rather than reached some other way (such as
   * the import-block class singleton).
   *
   * @param pointsToSet The receiver's points-to set.
   * @return {@code true} iff some member's allocation node is a {@code do} method.
   */
  private static boolean hasModeledProducer(OrdinalSet<InstanceKey> pointsToSet) {
    for (InstanceKey ik : pointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin != null && asin.getNode().getMethod().getName().toString().equals(DO_METHOD_NAME)) {
        return true;
      }
    }
    return false;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For dataset transformations, default to dtypes of the input dataset (the receiver).
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      Set<DType> dTypes = this.getDTypesOfValue(builder, receiverPTS);
      // The dtype counterpart of the shape floor above: ⊥ on one axis must pair with ⊥ on the
      // other, so an unresolved element dtype degrades to UNKNOWN (⊤) rather than emptying the
      // set, under the same modeled-producer gate (wala/ML#776).
      if ((dTypes == null || dTypes.isEmpty()) && hasModeledProducer(receiverPTS))
        return EnumSet.of(DType.UNKNOWN);
      return dTypes == null ? EnumSet.of(DType.UNKNOWN) : dTypes;
    }
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> receivers = this.receivers(builder);
    if (receivers.isEmpty()) return false;
    for (TensorGenerator receiver : receivers)
      if (!(receiver instanceof TupleElementProvider tep && tep.yieldsTuple(builder))) return false;
    return true;
  }

  /**
   * The receivers this transformation's element follows: the one this view is fixed to, every
   * dataset the receiver may be ({@link #getReceiverGenerators}), or, when those do not all
   * resolve, the first that does.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The receivers' generators, empty when none resolves.
   */
  private List<TensorGenerator> receivers(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> receivers = this.getReceiverGenerators(builder);
    if (receivers != null && !receivers.isEmpty()) return receivers;
    TensorGenerator receiver = this.getReceiverGenerator(builder);
    return receiver == null ? List.of() : List.of(receiver);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    TensorGenerator receiver = getReceiverGenerator(builder);
    if (receiver instanceof TupleElementProvider tep) {
      return tep.getShapesForIndex(builder, index);
    }
    return this.getShapes(builder);
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    TensorGenerator receiver = getReceiverGenerator(builder);
    if (receiver instanceof TupleElementProvider tep) {
      return tep.getDTypesForIndex(builder, index);
    }
    return this.getDTypes(builder);
  }

  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    TensorGenerator receiver = getReceiverGenerator(builder);
    return receiver instanceof TupleElementProvider tep && tep.resolvesPath(builder, path);
  }

  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    // A receiver that may be several datasets has a structure at the path only where each of them
    // does; where they disagree, the component is either, which a reader of its parts marks.
    List<TensorGenerator> receivers = this.receivers(builder);
    if (receivers.isEmpty()) return false;
    for (TensorGenerator receiver : receivers)
      if (!(receiver instanceof TupleElementProvider tep && tep.isStructureAt(builder, path)))
        return false;
    return true;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A receiver that may be several datasets declares a length at the path only where each
   *     of them declares the same one; where they disagree, or one declares none, the component may
   *     be either, so none is declared.
   */
  @Override
  public OptionalInt declaredArity(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TensorGenerator> receivers = this.receivers(builder);
    if (receivers.isEmpty()) return OptionalInt.empty();
    int ret = -1;
    for (TensorGenerator receiver : receivers) {
      if (!(receiver instanceof TupleElementProvider tep)) return OptionalInt.empty();
      OptionalInt arity = tep.declaredArity(builder, path);
      if (arity.isEmpty() || (ret >= 0 && ret != arity.getAsInt())) return OptionalInt.empty();
      ret = arity.getAsInt();
    }
    return OptionalInt.of(ret);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    TensorGenerator receiver = getReceiverGenerator(builder);
    if (receiver instanceof TupleElementProvider tep) return tep.getShapesForPath(builder, path);
    return this.getShapes(builder);
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    TensorGenerator receiver = getReceiverGenerator(builder);
    if (receiver instanceof TupleElementProvider tep) return tep.getDTypesForPath(builder, path);
    return this.getDTypes(builder);
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    // Null shapes signal "unknown per-index shape" (⊤). Emit one ⊤-shaped TensorType per dtype
    // rather than falling through to the aggregate {@code getTensorTypes}, which would silently
    // leak sibling fields' shapes. See wala/ML#396.
    if (shapes == null) {
      for (DType dtype : dTypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), null));
      return ret;
    }

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), dimensionList));

    return ret;
  }

  /**
   * Resolves the receiver (the dataset this one is derived from) to a {@link TensorGenerator}: the
   * receiver this view is fixed to ({@link #overReceiver}), if any, and otherwise the first member
   * of the receiver's points-to set that resolves; {@link #getReceiverGenerators} resolves them
   * all.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The generator for the receiver dataset, or {@code null} if it cannot be resolved.
   */
  public TensorGenerator getReceiverGenerator(PropagationCallGraphBuilder builder) {
    if (this.receiver != null) return this.receiver;
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      for (InstanceKey valueIK : receiverPTS) {
        TensorGenerator generator = receiverGenerator(builder, valueIK);
        if (generator != null) return generator;
      }
    }
    return null;
  }

  /**
   * Resolves every dataset the receiver may be to its {@link TensorGenerator}, where {@link
   * #getReceiverGenerator} resolves the first: a transformation applied to {@code a if flag else b}
   * has a receiver that is either dataset, and its element is the join of the two (wala/ML#1009).
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The receivers' generators, without duplicates, or {@code null} when this view is fixed
   *     to one receiver, the receiver has no member, or a member does not resolve.
   */
  public List<TensorGenerator> getReceiverGenerators(PropagationCallGraphBuilder builder) {
    if (this.receiver != null) return null;
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS == null || receiverPTS.isEmpty()) return null;
    List<TensorGenerator> ret = new ArrayList<>();
    for (InstanceKey valueIK : receiverPTS) {
      TensorGenerator generator = receiverGenerator(builder, valueIK);
      if (generator == null) return null;
      if (!ret.contains(generator)) ret.add(generator);
    }
    return ret;
  }

  /**
   * Whether this transformation's element is its receiver's element, transformed per component
   * through {@link #getReceiverGenerator}, so that it applies to each dataset the receiver may be
   * on its own ({@link #overReceiver}). A pass-through transformation ({@code shuffle}, {@code
   * prefetch}, {@code repeat}, ...) and a batch are; a transformation building its own element
   * ({@code map}, {@code zip}, {@code enumerate}) or a source is not.
   *
   * @return {@code true} iff the element follows the receiver's.
   */
  public boolean elementFollowsReceiver() {
    return this.getClass() == DatasetGenerator.class;
  }

  /**
   * This transformation applied to the given receiver alone, one of {@link #getReceiverGenerators}.
   *
   * @param receiver The receiver's generator.
   * @return A view of this generator whose receiver is the given one.
   */
  public DatasetGenerator overReceiver(TensorGenerator receiver) {
    DatasetGenerator ret = (DatasetGenerator) this.copy();
    ret.receiver = receiver;
    return ret;
  }

  /**
   * The generator of a receiver allocation.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param valueIK A member of the receiver's points-to set.
   * @return The generator, or {@code null} if the member does not resolve to one.
   */
  private static TensorGenerator receiverGenerator(
      PropagationCallGraphBuilder builder, InstanceKey valueIK) {
    AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
    if (asin == null) return null;
    // A mapped-dataset receiver is allocated in `map.do()` as a generic `Dataset`, so the default
    // dispatch below would resolve it to a plain `DatasetGenerator` that inherits from `map.do()`'s
    // own receiver (the upstream base), dropping `map_func`'s return. Resolve it to a
    // `DatasetMapGenerator` reading the `element` field off this receiver instance so the mapped
    // type survives a downstream pass-through transform. wala/ML#649.
    if (asin.getNode()
        .getMethod()
        .getDeclaringClass()
        .getReference()
        .equals(TensorFlowTypes.DATASET_MAP_TYPE)) {
      return new DatasetMapGenerator(asin.getNode(), asin);
    }
    // A dataset is allocated in a library summary, whose allocating local has an implicit pointer
    // key, so its generator is the summary node's own.
    return createManualGenerator(asin.getNode(), builder);
  }

  @Override
  public boolean equals(Object obj) {
    return super.equals(obj) && this.receiver == ((DatasetGenerator) obj).receiver;
  }

  @Override
  public int hashCode() {
    return 31 * super.hashCode() + System.identityHashCode(this.receiver);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A view fixed to one receiver ({@link #overReceiver}) is keyed on that receiver, by
   *     identity as {@link #equals} compares it, so views over different receivers, and the view
   *     and its unfixed original, never share a memoized result. Today every memoized read of a
   *     view reads the union of its receivers and so agrees across them; the key keeps that true
   *     once a whole-element read honors the fixed receiver.
   */
  @Override
  protected Object operationDiscriminator() {
    return this.receiver == null
        ? super.operationDiscriminator()
        : new FixedReceiver(this.receiver);
  }

  /**
   * The receiver a view is fixed to, compared by identity.
   *
   * @param receiver The receiver.
   */
  private record FixedReceiver(TensorGenerator receiver) {
    @Override
    public boolean equals(Object obj) {
      return obj instanceof FixedReceiver other && this.receiver == other.receiver;
    }

    @Override
    public int hashCode() {
      return System.identityHashCode(this.receiver);
    }
  }

  /**
   * Retrieves the sizes (number of elements) of the dataset represented by this generator. By
   * default, it recursively queries the receiver (the dataset this one is derived from).
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return A set of possible dataset sizes, or an empty set if unknown.
   */
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      Set<Long> ret = HashSetFactory.make();
      for (InstanceKey valueIK : receiverPTS) {
        if (getAllocationSiteInNode(valueIK) != null) {
          AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
          int vn = findDefinition(asin.getNode(), asin);
          if (vn > 0) {
            PointerKey pk =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(asin.getNode(), vn);
            PointsToSetVariable var = null;
            if (!builder.getPropagationSystem().isImplicit(pk)) {
              var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
            }
            TensorGenerator generator = null;
            if (var != null) {
              generator = TensorGeneratorFactory.getGenerator(var, builder);
            } else {
              generator = createManualGenerator(asin.getNode(), builder);
            }

            if (generator instanceof DatasetGenerator
                && !generator.getClass().equals(this.getClass())) {
              ret.addAll(((DatasetGenerator) generator).getDatasetSizes(builder));
            }
          }
        }
      }
      return ret;
    }
    return Collections.emptySet();
  }

  /**
   * Returns {@code true} iff the upstream dataset chain is guaranteed to be infinite &mdash; i.e.,
   * the nearest finite-ness-affecting ancestor is {@code .repeat()} rather than {@code .take(N)} or
   * a finite source.
   *
   * <p>Walks upstream via {@link #getReceiverGenerator}, only following plain {@link
   * DatasetGenerator} instances (the pass-through layer the factory creates for
   * shuffle/map/repeat/take/prefetch/etc.; see {@code TensorGeneratorFactory}'s dispatch for {@link
   * TensorFlowTypes#DATASET_REPEAT_TYPE} and friends). Each visited generator's invoke instruction
   * is inspected by resolving the call's callees and checking the declaring class of each callee.
   * The walk terminates at:
   *
   * <ul>
   *   <li>The first {@link TensorFlowTypes#DATASET_REPEAT_TYPE} callee &rarr; returns {@code true}
   *       (upstream is infinite; partial batches impossible downstream).
   *   <li>The first {@link TensorFlowTypes#DATASET_TAKE_TYPE} callee &rarr; returns {@code false}
   *       (take bounds an infinite stream to finite; partial batches possible).
   *   <li>A specific subclass of {@link DatasetGenerator} (e.g., {@link
   *       DatasetFromTensorSlicesGenerator}, {@link DatasetBatchGenerator}) or a {@code null}
   *       receiver &rarr; returns {@code false} (finite source; partial batches possible).
   * </ul>
   *
   * <p>Used by {@link DatasetBatchGenerator} to suppress spurious partial-batch shape siblings when
   * the upstream chain can't produce partial batches at runtime (e.g., {@code
   * from_tensor_slices(...).repeat().shuffle().batch(256)} &mdash; Python runtime only ever yields
   * the full batch shape; see verification in the plan).
   *
   * @param builder The propagation call graph builder used for call-graph and PA lookups.
   * @return {@code true} iff the nearest finite-ness-affecting upstream op is {@code .repeat()}.
   */
  protected boolean upstreamIsInfinite(PropagationCallGraphBuilder builder) {
    TensorGenerator up = this.getReceiverGenerator(builder);
    while (up != null && up.getClass() == DatasetGenerator.class) {
      DatasetGenerator dg = (DatasetGenerator) up;
      TypeReference declaring = upstreamOpClass(dg, builder);
      if (declaring != null) {
        if (declaring.equals(TensorFlowTypes.DATASET_REPEAT_TYPE)) return true;
        if (declaring.equals(TensorFlowTypes.DATASET_TAKE_TYPE)) return false;
      }
      up = dg.getReceiverGenerator(builder);
    }
    return false;
  }

  /**
   * Identifies the dataset operation this generator represents. Prefers the invoke instruction's
   * callee declaring class when the generator has a points-to source; falls back to the generator's
   * CG node's method-declaring class for manual (node-only) generators produced by {@link
   * DatasetGenerator#getReceiverGenerator} when the upstream PK was implicit and {@code
   * createManualGenerator} was used.
   *
   * @param dg The upstream generator to identify.
   * @param builder The propagation call graph builder.
   * @return The {@link TypeReference} of the dataset op (e.g., {@code Ltensorflow/data/repeat}), or
   *     {@code null} if neither path yields one.
   */
  private static TypeReference upstreamOpClass(
      DatasetGenerator dg, PropagationCallGraphBuilder builder) {
    PythonInvokeInstruction invoke = dg.getInvokeInstruction();
    if (invoke != null) {
      for (CGNode callee :
          builder.getCallGraph().getPossibleTargets(dg.getNode(), invoke.getCallSite())) {
        TypeReference declaring = callee.getMethod().getReference().getDeclaringClass();
        // Return the first callee's declaring class; dispatch is typically unique for these ops.
        return declaring;
      }
    }
    // Fall back to the node's method declaring class. For manual generators produced from a
    // synthetic method's allocation site (e.g., `tensorflow.data.repeat.do`), the node's method
    // IS the op.
    CGNode node = dg.getNode();
    if (node != null) {
      return node.getMethod().getDeclaringClass().getReference();
    }
    return null;
  }
}
