package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.withoutClientOnlyKeys;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for datasets created by {@code tf.data.Dataset.flat_map} and {@code interleave}
 * (wala/ML#1214). Each calls its function on the receiver's element, as {@code map} does, and its
 * element is any element of the datasets the function returns, so those datasets are the ones a
 * read of the element joins over. The summary stores them on the dataset's {@code element} field.
 */
public class DatasetFlatMapGenerator extends DatasetJoinGenerator {

  /** The field the summary stores the datasets the function returns into. */
  private static final String RETURNED_FIELD = "element";

  /** A method every summary dataset carries on a field of its own. */
  private static final String METHOD_FIELD = "batch";

  public DatasetFlatMapGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetFlatMapGenerator(CGNode node) {
    super(node);
  }

  /**
   * The instances of this dataset: the source's points-to set, or, for a generator anchored on the
   * summary's node, the dataset the node allocates.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The instances, or {@code null} when none resolves.
   */
  private OrdinalSet<InstanceKey> selfInstances(PropagationCallGraphBuilder builder) {
    if (this.getSource() != null)
      return withoutClientOnlyKeys(
          builder.getPointerAnalysis().getPointsToSet(this.getSource().getPointerKey()));
    CGNode node = this.getNode();
    IR ir = node == null ? null : node.getIR();
    if (ir == null) return null;
    List<InstanceKey> ret = new ArrayList<>();
    for (SSAInstruction inst : ir.getInstructions())
      if (inst instanceof SSANewInstruction allocation
          && allocation.getConcreteType().equals(TensorFlowTypes.DATASET)) {
        InstanceKey ik =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getInstanceKeyForAllocation(node, allocation.getNewSite());
        if (ik != null) ret.add(ik);
      }
    return ret.isEmpty()
        ? null
        : OrdinalSet.toOrdinalSet(ret, builder.getPointerAnalysis().getInstanceKeyMapping());
  }

  /**
   * Whether the function's arguments are unknown, because the receiver's element is a tuple longer
   * than the summary's calls spread into (wala/ML#164): no call is made, so the datasets it returns
   * say nothing about the element.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return {@code true} iff an instance of this dataset is such a one.
   */
  private boolean overflows(PropagationCallGraphBuilder builder) {
    if (!(builder instanceof PythonSSAPropagationCallGraphBuilder python)) return false;
    OrdinalSet<InstanceKey> self = this.selfInstances(builder);
    if (self == null) return false;
    for (InstanceKey ik : self) if (python.isMapOverflow(ik)) return true;
    return false;
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The datasets the function returns, which the summary stores on each instance's {@code
   *     element} field. The receiver is not among them: the function may return a dataset of the
   *     receiver's element, but then it is that dataset, not the receiver, that the element
   *     follows. A returned value that resolves to no dataset generator, such as a dataset from an
   *     API with no summary or a value that is no dataset, is left out, so its elements are no
   *     tensor (⊥) and the missing modeling stays visible rather than read as a tensor of unknown
   *     type.
   */
  @Override
  public List<TensorGenerator> getReceiverGenerators(PropagationCallGraphBuilder builder) {
    if (this.getFixedReceiver() != null || this.overflows(builder)) return null;
    OrdinalSet<InstanceKey> self = this.selfInstances(builder);
    if (self == null || self.isEmpty()) return null;
    IField field =
        builder
            .getClassHierarchy()
            .resolveField(
                FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(RETURNED_FIELD), Root));
    if (field == null) return null;
    List<TensorGenerator> ret = new ArrayList<>();
    for (InstanceKey ik : self) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) return null;
      for (InstanceKey returned :
          withoutClientOnlyKeys(
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(builder.getPointerKeyForInstanceField(asin, field)))) {
        // Only a dataset has elements; a function returning anything else fails at run time. The
        // key's own class decides, since a tensor allocated in a dataset's summary, as a mirrored
        // element's leaf is, resolves to that dataset's generator.
        if (!isDataset(builder, returned)) continue;
        TensorGenerator generator = receiverGenerator(builder, returned);
        if (generator != null && !ret.contains(generator)) ret.add(generator);
      }
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Whether a key is a dataset: of the dataset class or a subclass of it, or one a summary makes,
   * which, as every summary dataset does, carries its methods on its own fields, so its {@code
   * batch} field is not empty. A summary dataset such as {@code TextLineDataset} has a class of its
   * own, and a tensor, a method object or a summary's helper carries no {@code batch}.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @param key The key.
   * @return {@code true} iff the key is a dataset.
   */
  private static boolean isDataset(PropagationCallGraphBuilder builder, InstanceKey key) {
    if (PythonTensorAnalysisEngine.isDataset(builder, key)) return true;
    AllocationSiteInNode asin;
    try {
      asin = getAllocationSiteInNode(key);
    } catch (IllegalArgumentException e) {
      // A key that is not an allocation, such as a client-only constant, is no dataset.
      return false;
    }
    if (asin == null) return false;
    IField batch =
        builder
            .getClassHierarchy()
            .resolveField(
                FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(METHOD_FIELD), Root));
    return batch != null
        && !builder
            .getPointerAnalysis()
            .getPointsToSet(builder.getPointerKeyForInstanceField(asin, batch))
            .isEmpty();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The dataset a view is fixed to, or the first dataset the function returns, never the
   *     receiver: the receiver's element is the function's argument, not this dataset's element.
   */
  @Override
  public TensorGenerator getReceiverGenerator(PropagationCallGraphBuilder builder) {
    TensorGenerator fixed = this.getFixedReceiver();
    if (fixed != null) return fixed;
    List<TensorGenerator> returned = this.getReceiverGenerators(builder);
    return returned == null ? null : returned.get(0);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Never the receiver's element, which is the function's argument, not this dataset's
   *     element. Unknown (⊤) where the receiver's element is a tuple longer than the summary's
   *     calls spread into, as for {@code map}, and no tensor (⊥) where no dataset the function
   *     returns resolves.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    if (this.getFixedReceiver() == null) {
      if (this.overflows(builder)) return null;
      if (this.joined(builder) == null) return Collections.emptySet();
    }
    return super.getDefaultShapes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown, or no tensor, where the shapes are.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    if (this.getFixedReceiver() == null) {
      if (this.overflows(builder)) return EnumSet.of(DType.UNKNOWN);
      if (this.joined(builder) == null) return Collections.emptySet();
    }
    return super.getDefaultDTypes(builder);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote Unknown: how many elements the datasets the function returns hold, summed over the
   *     receiver's elements, is not computed, and {@code tf.data} reports it as unknown too.
   */
  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
