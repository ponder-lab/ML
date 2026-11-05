package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SCAN_TYPE;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for the dataset {@code tf.data.Dataset.scan} returns (wala/ML#1214): a dataset of the
 * second member of each pair {@code scan_func} returns. The summary of {@code scan} stores that
 * member on the result's {@code mirrored_element}, so the element is the value that store writes,
 * read in the {@code scan} node. The scan keeps every element, so its size is its receiver's.
 *
 * <p>An element that is itself a structure is read as one unknown tensor.
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
   * {@inheritDoc}
   *
   * @implNote The union of the shapes of the stored element values, unknown when any is or when
   *     none resolves.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<Pair<CGNode, Integer>> values = this.elementValues(builder);
    if (values.isEmpty()) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Pair<CGNode, Integer> value : values) {
      Set<List<Dimension<?>>> shapes = this.getShapes(builder, value.fst, value.snd);
      if (shapes == null) return null;
      ret.addAll(shapes);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union of the dtypes of the stored element values, unknown when any is or when
   *     none resolves.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<Pair<CGNode, Integer>> values = this.elementValues(builder);
    if (values.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Pair<CGNode, Integer> value : values) {
      Set<DType> dtypes = this.getDTypes(builder, value.fst, value.snd);
      if (dtypes == null || dtypes.isEmpty() || dtypes.contains(DType.UNKNOWN))
        return EnumSet.of(DType.UNKNOWN);
      ret.addAll(dtypes);
    }
    return ret;
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

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    return null;
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForPath(
      PropagationCallGraphBuilder builder, List<Object> path) {
    return path.isEmpty() ? this.getDefaultShapes(builder) : null;
  }

  @Override
  public Set<DType> getDTypesForPath(PropagationCallGraphBuilder builder, List<Object> path) {
    return path.isEmpty() ? this.getDefaultDTypes(builder) : EnumSet.of(DType.UNKNOWN);
  }
}
