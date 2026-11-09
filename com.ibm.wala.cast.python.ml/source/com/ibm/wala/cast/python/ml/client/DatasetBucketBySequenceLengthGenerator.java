package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.bucket_by_sequence_length}
 * (wala/ML#1214). The element is a batch of the receiver's, each element padded to the longest in
 * its batch. The batch size depends on the bucket, so it is not counted. {@code
 * element_length_func} is not called.
 */
public class DatasetBucketBySequenceLengthGenerator extends DatasetBatchGenerator {

  /** The position of the {@code padded_shapes} argument, excluding the receiver. */
  private static final int PADDED_SHAPES_PARAMETER_POSITION = 3;

  /** The keyword name of the {@code padded_shapes} argument. */
  private static final String PADDED_SHAPES_PARAMETER_NAME = "padded_shapes";

  /** The position of the {@code pad_to_bucket_boundary} argument, excluding the receiver. */
  private static final int PAD_TO_BUCKET_BOUNDARY_PARAMETER_POSITION = 5;

  /** The keyword name of the {@code pad_to_bucket_boundary} argument. */
  private static final String PAD_TO_BUCKET_BOUNDARY_PARAMETER_NAME = "pad_to_bucket_boundary";

  /** The position of the {@code drop_remainder} argument, excluding the receiver. */
  private static final int DROP_REMAINDER_PARAMETER_POSITION = 7;

  protected DatasetBucketBySequenceLengthGenerator(PointsToSetVariable source) {
    super(source);
  }

  protected DatasetBucketBySequenceLengthGenerator(CGNode node) {
    super(node);
  }

  /**
   * Whether a node is the summary of the transformation {@code
   * tf.data.experimental.bucket_by_sequence_length} returns, which forwards every argument it was
   * made with to the dataset's own method, supplied or not.
   *
   * @param node The node.
   * @return {@code true} iff the node forwards the arguments.
   */
  private static boolean isForwarder(CGNode node) {
    return node.getMethod()
        .getDeclaringClass()
        .getReference()
        .equals(TensorFlowTypes.EXPERIMENTAL_BUCKET_BY_SEQUENCE_LENGTH_FN_TYPE);
  }

  /**
   * Whether the program supplies an argument to the {@code
   * tf.data.experimental.bucket_by_sequence_length} calls that made the transformations a
   * forwarding node runs as. The function's parameters are the method's, in the same order.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param forwarder The forwarding node.
   * @param paramPos The argument's position, excluding the receiver.
   * @param paramName The argument's keyword name.
   * @return {@code TRUE}, {@code FALSE}, or {@code null}, as {@link
   *     #isArgumentSyntacticallySupplied} answers.
   */
  private static Boolean suppliedToTransformation(
      PropagationCallGraphBuilder builder, CGNode forwarder, int paramPos, String paramName) {
    PointerKey self =
        builder
            .getPointerAnalysis()
            .getHeapModel()
            .getPointerKeyForLocal(forwarder, forwarder.getIR().getParameter(0));
    boolean sawMaker = false;
    boolean indeterminate = false;
    for (InstanceKey transformation : builder.getPointerAnalysis().getPointsToSet(self)) {
      AllocationSiteInNode asin = getAllocationSiteInNode(transformation);
      if (asin == null) return null;
      sawMaker = true;
      Boolean here =
          isArgumentSyntacticallySuppliedAt(builder, asin.getNode(), paramPos, paramName);
      if (Boolean.TRUE.equals(here)) return true;
      if (here == null) indeterminate = true;
    }
    if (indeterminate) return null;
    return sawMaker ? Boolean.FALSE : null;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A call from the transformation {@code tf.data.experimental.bucket_by_sequence_length}
   *     returns, which {@code apply} runs, passes every argument whether the program supplied it or
   *     not, so there the program's calls of {@code tf.data.experimental.bucket_by_sequence_length}
   *     decide (wala/ML#1214).
   */
  @Override
  protected Boolean isArgumentSyntacticallySupplied(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    CGNode node = this.getNode();
    if (node == null) return super.isArgumentSyntacticallySupplied(builder, paramPos, paramName);
    PythonInvokeInstruction call = this.getInvokeInstruction();
    if (call != null)
      return isForwarder(node)
          ? suppliedToTransformation(builder, node, paramPos, paramName)
          : super.isArgumentSyntacticallySupplied(builder, paramPos, paramName);
    boolean sawSite = false;
    boolean indeterminate = false;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, node)) {
      Boolean here;
      if (isForwarder(callerInvoke.fst))
        here = suppliedToTransformation(builder, callerInvoke.fst, paramPos, paramName);
      else if (callerInvoke.snd instanceof PythonInvokeInstruction invoke)
        here = invokeSuppliesArgument(invoke, paramPos, paramName);
      else here = null;
      sawSite = true;
      if (Boolean.TRUE.equals(here)) return true;
      if (here == null) indeterminate = true;
    }
    if (indeterminate) return null;
    return sawSite ? Boolean.FALSE : null;
  }

  @Override
  protected int getDropRemainderParameterIndex() {
    return DROP_REMAINDER_PARAMETER_POSITION;
  }

  /**
   * Whether padding to the longest element in a batch leaves each element's shape as it is: only
   * when no {@code padded_shapes} is supplied, which declares the padded shape, and {@code
   * pad_to_bucket_boundary} is omitted or provably {@code False}, since padding to a bucket's
   * boundary changes the padded axis.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @return {@code true} iff the padded element has the receiver's element's shapes.
   */
  private boolean keepsElementShapes(PropagationCallGraphBuilder builder) {
    if (!Boolean.FALSE.equals(
        this.isArgumentSyntacticallySupplied(
            builder, PADDED_SHAPES_PARAMETER_POSITION, PADDED_SHAPES_PARAMETER_NAME))) return false;
    Boolean toBoundary =
        this.isArgumentSyntacticallySupplied(
            builder,
            PAD_TO_BUCKET_BOUNDARY_PARAMETER_POSITION,
            PAD_TO_BUCKET_BOUNDARY_PARAMETER_NAME);
    if (Boolean.FALSE.equals(toBoundary)) return true;
    if (toBoundary == null) return false;
    Set<Boolean> values =
        getPossibleBooleanValues(
            this.getArgumentPointsToSet(
                builder,
                PAD_TO_BUCKET_BOUNDARY_PARAMETER_POSITION,
                PAD_TO_BUCKET_BOUNDARY_PARAMETER_NAME));
    return values != null && values.equals(Collections.singleton(Boolean.FALSE));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Each shape behind a batch dimension of unknown size, where padding to the longest
   *     element leaves the shapes as they are ({@link #keepsElementShapes}), and unknown otherwise.
   *     The shapes given are already padded ({@link #padAgainst}).
   */
  @Override
  protected Set<List<Dimension<?>>> applyBatching(
      Set<List<Dimension<?>>> inputShapes, PropagationCallGraphBuilder builder) {
    if (inputShapes == null || !this.keepsElementShapes(builder)) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      List<Dimension<?>> batched = new ArrayList<>();
      batched.add(new SymbolicDim("?"));
      batched.addAll(shape);
      ret.add(batched);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The receiver's element shapes, padded against every shape the receiver may yield.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = this.getReceiverElementShapes(builder);
    if (inputShapes == null || inputShapes.isEmpty()) return null;
    return this.applyBatching(padAgainst(inputShapes, this.paddingContext(builder, null)), builder);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return this.getShapesForPath(builder, List.of(index));
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The receiver's component's shapes, padded against every shape the receiver may yield
   *     there.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    TensorGenerator receiver = this.getReceiverGenerator(builder);
    if (receiver instanceof TupleElementProvider tep) {
      Set<List<Dimension<?>>> componentShapes = tep.getShapesForPath(builder, path);
      if (componentShapes == null) return null;
      if (!componentShapes.isEmpty())
        return this.applyBatching(
            padAgainst(componentShapes, this.paddingContext(builder, path)), builder);
    }
    return this.getShapes(builder);
  }

  /**
   * Every shape the receiver may yield, whole or at a path, read over the receiver as a whole and
   * not over the one dataset a view of this transformation is fixed to ({@link #overReceiver}). A
   * batch pads elements that arrive together, and the elements of every dataset a concatenation
   * joins arrive in one run, though a view is fixed to one of them. Reading every dataset the
   * receiver may be over-approximates where the receiver is one of several in a run, and only where
   * their shapes differ on more than one axis.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param path The path of a component, or {@code null} for the whole element.
   * @return The shapes, or {@code null} where some do not resolve.
   */
  private Set<List<Dimension<?>>> paddingContext(
      PropagationCallGraphBuilder builder, List<Object> path) {
    OrdinalSet<InstanceKey> receiverPTS = this.getReceiverPTS(builder);
    if (receiverPTS == null || receiverPTS.isEmpty()) return null;
    if (path == null) return this.getShapesOfValue(builder, receiverPTS);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (InstanceKey valueIK : receiverPTS) {
      TensorGenerator receiver = DatasetGenerator.receiverGenerator(builder, valueIK);
      if (!(receiver instanceof TupleElementProvider tep)) return null;
      Set<List<Dimension<?>>> shapes = tep.getShapesForPath(builder, path);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * The shapes elements take once padded to the longest in their batch, which pads each axis to
   * that axis's longest extent. Where the shapes of one rank that may share a batch differ on one
   * axis at most, the padded shape is one of them, so each stands. Where they differ on more, the
   * padded shape may combine the axes of different elements, so the axes they differ on are {@link
   * DynamicDim}: the padded extent varies per batch.
   *
   * @param shapes The shapes to pad.
   * @param context Every shape that may share a batch with them, or {@code null} where it is
   *     unknown.
   * @return The padded shapes, or {@code null} where the context is unknown.
   */
  private static Set<List<Dimension<?>>> padAgainst(
      Set<List<Dimension<?>>> shapes, Set<List<Dimension<?>>> context) {
    if (context == null) return null;
    Map<Integer, List<List<Dimension<?>>>> byRank = new HashMap<>();
    for (List<Dimension<?>> shape : context)
      byRank.computeIfAbsent(shape.size(), k -> new ArrayList<>()).add(shape);
    Map<Integer, List<Dimension<?>>> joinedByRank = new HashMap<>();
    for (Map.Entry<Integer, List<List<Dimension<?>>>> entry : byRank.entrySet()) {
      List<List<Dimension<?>>> group = entry.getValue();
      List<Dimension<?>> first = group.get(0);
      List<Dimension<?>> joined = new ArrayList<>(first);
      int differing = 0;
      for (int axis = 0; axis < first.size(); axis++) {
        Dimension<?> dim = first.get(axis);
        final int at = axis;
        if (group.stream().anyMatch(shape -> !shape.get(at).equals(dim))) {
          differing++;
          joined.set(axis, DynamicDim.INSTANCE);
        }
      }
      if (differing > 1) joinedByRank.put(entry.getKey(), joined);
    }
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : shapes) ret.add(joinedByRank.getOrDefault(shape.size(), shape));
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown: the number of batches depends on how the elements fall into the buckets.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
