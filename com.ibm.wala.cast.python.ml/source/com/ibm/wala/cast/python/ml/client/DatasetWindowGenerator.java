package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAGetInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Iterator2Iterable;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for a window of a dataset, which {@code tf.data.Dataset.window} yields
 * (wala/ML#1201): a dataset of its receiver's elements, or, for a tuple element, of one of their
 * components. A window may hold fewer elements than its size, so its size is unknown. Each window
 * is allocated and returned by its own maker, whose node it is anchored on, and the maker's {@code
 * source} carries the receiver; a window of a component is made by {@code make_window_component},
 * and its position is the field {@code window} read the component from (wala/ML#1221).
 */
public class DatasetWindowGenerator extends DatasetGenerator {

  /**
   * The position of a window maker's {@code source}: the method object {@code window} was called
   * on.
   */
  private static final int SOURCE_PARAMETER_POSITION = 0;

  /** The name of a window maker's {@code source} parameter. */
  private static final String SOURCE_PARAMETER_NAME = "source";

  /**
   * The use of a call of a window maker that passes its {@code element}, the component: after the
   * maker itself and {@code source}.
   */
  private static final int ELEMENT_USE = 2;

  /**
   * The part of the receiver's element this window holds: {@link
   * TensorFlowTypes#DATASET_WINDOW_WHOLE} for the whole, or {@link
   * TensorFlowTypes#DATASET_WINDOW_OF_COMPONENT} for a component, whose position {@link #position}
   * reads.
   */
  private final int component;

  /**
   * {@inheritDoc}
   *
   * <p>A shallow copy suffices: {@link #component} is a primitive.
   */
  @Override
  protected DatasetWindowGenerator clone() throws CloneNotSupportedException {
    return (DatasetWindowGenerator) super.clone();
  }

  /**
   * Constructs a generator for a window that the summary of {@code window} allocates.
   *
   * @param node The {@code window} node.
   * @param component The part of the receiver's element the window holds: {@link
   *     TensorFlowTypes#DATASET_WINDOW_WHOLE} or {@link
   *     TensorFlowTypes#DATASET_WINDOW_OF_COMPONENT}.
   */
  protected DatasetWindowGenerator(CGNode node, int component) {
    super(node);
    this.component = component;
  }

  @Override
  public boolean equals(Object obj) {
    return super.equals(obj) && this.component == ((DatasetWindowGenerator) obj).component;
  }

  @Override
  public int hashCode() {
    return 31 * super.hashCode() + this.component;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Keyed on the component too, so a window of the whole element and one of a component
   *     never share a memoized result.
   */
  @Override
  protected Object operationDiscriminator() {
    return Pair.make(super.operationDiscriminator(), this.component);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The maker's {@code source}, the method object {@code window} was called on, which
   *     carries the dataset {@code window} was called on as a transformation's own {@code self}
   *     does.
   */
  @Override
  protected OrdinalSet<InstanceKey> receiverPointsToSet(PropagationCallGraphBuilder builder) {
    return this.getArgumentPointsToSet(builder, SOURCE_PARAMETER_POSITION, SOURCE_PARAMETER_NAME);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A window of the whole element hands on the receiver's element.
   */
  @Override
  public boolean elementFollowsReceiver() {
    return this.isWhole();
  }

  /**
   * Whether this window holds the receiver's whole element.
   *
   * @return {@code true} iff so.
   */
  private boolean isWhole() {
    return this.component == TensorFlowTypes.DATASET_WINDOW_WHOLE;
  }

  /**
   * The position of the component this window holds: the field {@code window} read the component
   * from, which it passes its maker as {@code element} (wala/ML#1221).
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The position, or {@code null} when the calls do not pass a component read at one
   *     position.
   */
  private Integer position(PropagationCallGraphBuilder builder) {
    CallGraph cg = builder.getCallGraph();
    CGNode node = this.getNode();
    Set<Integer> positions = HashSetFactory.make();
    for (CGNode caller : Iterator2Iterable.make(cg.getPredNodes(node))) {
      IR ir = caller.getIR();
      if (ir == null || caller.getDU() == null) return null;
      for (CallSiteReference site : Iterator2Iterable.make(cg.getPossibleSites(caller, node)))
        for (SSAAbstractInvokeInstruction call : ir.getCalls(site)) {
          if (ELEMENT_USE >= call.getNumberOfUses()) return null;
          if (!(caller.getDU().getDef(call.getUse(ELEMENT_USE)) instanceof SSAGetInstruction get))
            return null;
          try {
            positions.add(Integer.parseInt(get.getDeclaredField().getName().toString()));
          } catch (NumberFormatException e) {
            return null;
          }
        }
    }
    if (positions.size() != 1) return null;
    int position = positions.iterator().next();
    return position >= 0 ? position : null;
  }

  /**
   * The path of a component of this window's element within the receiver's element.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param path The path within this window's element.
   * @return The path within the receiver's element, or {@code null} when the component's position
   *     does not resolve.
   */
  private List<Object> inReceiver(PropagationCallGraphBuilder builder, List<Object> path) {
    if (this.isWhole()) return path;
    Integer position = this.position(builder);
    if (position == null) return null;
    List<Object> ret = new ArrayList<>(path.size() + 1);
    ret.add(position);
    ret.addAll(path);
    return ret;
  }

  /**
   * The receivers whose element's component this window reads: each dataset the receiver may be, or
   * the first that resolves.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The receivers that provide tuple elements, empty when none resolves.
   */
  private List<TupleElementProvider> componentReceivers(PropagationCallGraphBuilder builder) {
    List<TensorGenerator> receivers = this.getReceiverGenerators(builder);
    if (receivers == null || receivers.isEmpty()) {
      TensorGenerator receiver = this.getReceiverGenerator(builder);
      receivers = receiver == null ? List.of() : List.of(receiver);
    }
    List<TupleElementProvider> ret = new ArrayList<>();
    for (TensorGenerator receiver : receivers)
      if (receiver instanceof TupleElementProvider tep) ret.add(tep);
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    if (this.isWhole()) return super.getDefaultShapes(builder);
    return this.componentShapes(builder, List.of());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    if (this.isWhole()) return super.getDefaultDTypes(builder);
    return this.componentDTypes(builder, List.of());
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (this.isWhole()) return super.getShapesForIndex(builder, index);
    return this.getShapesForPath(builder, List.of(index));
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (this.isWhole()) return super.getDTypesForIndex(builder, index);
    return this.getDTypesForPath(builder, List.of(index));
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    if (this.isWhole()) return super.getShapesForPath(builder, path);
    return this.componentShapes(builder, path);
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (this.isWhole()) return super.getDTypesForPath(builder, path);
    return this.componentDTypes(builder, path);
  }

  /**
   * The shapes of a window of a component, at a path within its element: the union over the
   * receivers of their shapes at the component's path, unknown where any is, and no tensor where no
   * receiver resolves. A receiver that reads no tensor there adds nothing.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param path The path within the window's element, empty for the whole element.
   * @return The shapes.
   */
  private Set<List<Dimension<?>>> componentShapes(
      PropagationCallGraphBuilder builder, List<Object> path) {
    List<TupleElementProvider> receivers = this.componentReceivers(builder);
    if (receivers.isEmpty()) return Collections.emptySet();
    List<Object> at = this.inReceiver(builder, path);
    if (at == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (TupleElementProvider receiver : receivers) {
      Set<List<Dimension<?>>> shapes = receiver.getShapesForPath(builder, at);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * The dtypes of a window of a component, at a path within its element, read as {@link
   * #componentShapes} reads the shapes: a receiver that reads no tensor there adds nothing.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param path The path within the window's element, empty for the whole element.
   * @return The dtypes.
   */
  private Set<DType> componentDTypes(PropagationCallGraphBuilder builder, List<Object> path) {
    List<TupleElementProvider> receivers = this.componentReceivers(builder);
    if (receivers.isEmpty()) return Collections.emptySet();
    List<Object> at = this.inReceiver(builder, path);
    if (at == null) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (TupleElementProvider receiver : receivers) {
      Set<DType> dtypes = receiver.getDTypesForPath(builder, at);
      if (dtypes == null || dtypes.contains(DType.UNKNOWN)) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    if (this.isWhole()) return super.yieldsTuple(builder);
    return this.isStructureAt(builder, List.of());
  }

  @Override
  public boolean yieldsOnlyTuples(PropagationCallGraphBuilder builder) {
    return this.isWhole() && super.yieldsOnlyTuples(builder);
  }

  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    if (this.isWhole()) return super.isStructureAt(builder, path);
    List<TupleElementProvider> receivers = this.componentReceivers(builder);
    List<Object> at = this.inReceiver(builder, path);
    if (receivers.isEmpty() || at == null) return false;
    for (TupleElementProvider receiver : receivers)
      if (!receiver.isStructureAt(builder, at)) return false;
    return true;
  }

  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    if (this.isWhole()) return super.resolvesPath(builder, path);
    List<Object> at = this.inReceiver(builder, path);
    if (at == null) return false;
    for (TupleElementProvider receiver : this.componentReceivers(builder))
      if (receiver.resolvesPath(builder, at)) return true;
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown: a window holds at most its size, and the last may hold fewer.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
