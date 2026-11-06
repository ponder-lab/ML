package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SCAN_TYPE;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for the dataset {@code tf.data.Dataset.scan} returns (wala/ML#1214): a dataset of the
 * second member of each pair {@code scan_func} returns. The summary of {@code scan} stores that
 * member on the result's {@code mirrored_element}, so the element is the value that store writes,
 * read in the {@code scan} node. The scan keeps every element, so its size is its receiver's.
 *
 * <p>An element that is itself a structure, such as a tuple returned as the second member, is not
 * modeled, so it is read as no tensor, and so is each of its members, rather than as a tensor of
 * unknown type that would stand in for the missing model.
 */
public class DatasetScanGenerator extends DatasetGenerator {

  /** The field the summary of {@code scan} stores the element on. */
  private static final String ELEMENT_FIELD = "mirrored_element";

  public DatasetScanGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetScanGenerator(CGNode node) {
    super(node);
  }

  /**
   * The {@code scan} nodes that made this dataset: the targets of the call that returned it for a
   * source-based generator, and the generator's own node for a manual one.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The {@code scan} nodes, empty when none resolves.
   */
  private Set<CGNode> scanNodes(PropagationCallGraphBuilder builder) {
    Set<CGNode> ret = HashSetFactory.make();
    CGNode node = this.getNode();
    if (node == null) return ret;
    if (this.getSource() == null || this.getInvokeInstruction() == null) {
      if (isScan(node)) ret.add(node);
      return ret;
    }
    for (CGNode callee :
        builder.getCallGraph().getPossibleTargets(node, this.getInvokeInstruction().getCallSite()))
      if (isScan(callee)) ret.add(callee);
    return ret;
  }

  private static boolean isScan(CGNode node) {
    return node.getMethod().getDeclaringClass().getReference().equals(DATASET_SCAN_TYPE);
  }

  /**
   * The values each {@code scan} node stores as its result's element.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The (node, value number) pairs, empty when none resolves.
   */
  private Set<Pair<CGNode, Integer>> elementValues(PropagationCallGraphBuilder builder) {
    Set<Pair<CGNode, Integer>> ret = HashSetFactory.make();
    for (CGNode node : this.scanNodes(builder)) {
      IR ir = node.getIR();
      if (ir == null) continue;
      for (SSAInstruction instruction : ir.getInstructions())
        if (instruction instanceof SSAPutInstruction put
            && !put.isStatic()
            && put.getDeclaredField().getName().toString().equals(ELEMENT_FIELD))
          ret.add(Pair.make(node, put.getVal()));
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The element is {@code scan_func}'s return, not the receiver's element.
   */
  @Override
  public boolean elementFollowsReceiver() {
    return false;
  }

  /**
   * Whether every object a stored element value may be is a structure, such as a tuple, which this
   * generator does not model.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param value The (node, value number) of the stored value.
   * @return {@code true} iff the value's points-to set is non-empty and holds only structures.
   */
  private static boolean isStructure(
      PropagationCallGraphBuilder builder, Pair<CGNode, Integer> value) {
    boolean any = false;
    for (InstanceKey ik :
        builder
            .getPointerAnalysis()
            .getPointsToSet(
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(value.fst, value.snd))) {
      if (!PythonTypes.isContainer(ik.concreteType().getReference())) return false;
      any = true;
    }
    return any;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union of the shapes of the stored element values. A value that is a structure is
   *     not modeled and contributes nothing, and so does no value at all, so a union that is empty
   *     is no tensor. It is unknown only when a value's own reading is.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Pair<CGNode, Integer> value : this.elementValues(builder)) {
      if (isStructure(builder, value)) continue;
      Set<List<Dimension<?>>> shapes = this.getShapes(builder, value.fst, value.snd);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union of the dtypes of the stored element values, paired with {@link
   *     #getDefaultShapes}: no tensor where it reads none, and unknown when a value's own reading
   *     is.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = this.getDefaultShapes(builder);
    if (shapes != null && shapes.isEmpty()) return Collections.emptySet();
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Pair<CGNode, Integer> value : this.elementValues(builder)) {
      if (isStructure(builder, value)) continue;
      Set<DType> dtypes = this.getDTypes(builder, value.fst, value.snd);
      if (dtypes == null || dtypes.contains(DType.UNKNOWN)) return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    return false;
  }

  @Override
  public boolean yieldsOnlyTuples(PropagationCallGraphBuilder builder) {
    return false;
  }

  @Override
  public boolean isStructureAt(PropagationCallGraphBuilder builder, List<Object> path) {
    return false;
  }

  @Override
  public boolean resolvesPath(PropagationCallGraphBuilder builder, List<Object> path) {
    return path.isEmpty();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote No tensor: a member of an element that is a structure is not modeled.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return Collections.emptySet();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote No tensor, as {@link #getShapesForIndex} reads it.
   */
  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return Collections.emptySet();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The element for the empty path, and no tensor for a component, as {@link
   *     #getShapesForIndex} reads it.
   */
  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    return path.isEmpty() ? this.getDefaultShapes(builder) : Collections.emptySet();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Paired with {@link #getShapesForPath}.
   */
  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    return path.isEmpty() ? this.getDefaultDTypes(builder) : Collections.emptySet();
  }
}
