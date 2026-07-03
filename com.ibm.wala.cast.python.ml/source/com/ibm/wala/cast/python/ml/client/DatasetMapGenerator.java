package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ir.ssa.AstPropertyWrite;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for {@code tf.data.Dataset.map(map_func)}. The default {@link DatasetGenerator}
 * inherits its element type from the receiver dataset, but a mapped dataset's element type is
 * {@code map_func}'s return type. The {@code map.do()} summary invokes {@code map_func} (passing
 * the receiver, whose tensor type is its element type) and stores the result in the mapped
 * dataset's {@code element} field; this generator reads that field. Falls back to
 * receiver-inheritance when the field is empty. Tracked by <a
 * href="https://github.com/wala/ML/issues/506">wala/ML#506</a>.
 */
public class DatasetMapGenerator extends DatasetGenerator {

  /** The field {@code map.do()} stores {@code map_func}'s return (the mapped element) into. */
  private static final String ELEMENT_FIELD = "element";

  /**
   * The specific mapped-dataset instance whose {@code element} field carries the mapped type, when
   * this generator is resolved from a receiver instance rather than a source variable (e.g. a
   * downstream {@code repeat}/{@code prefetch} inheriting from a {@code map} receiver). {@code
   * null} for the source- and node-based constructions.
   */
  private final AllocationSiteInNode mapResultInstance;

  public DatasetMapGenerator(PointsToSetVariable source) {
    super(source);
    this.mapResultInstance = null;
  }

  public DatasetMapGenerator(CGNode node) {
    super(node);
    this.mapResultInstance = null;
  }

  /**
   * Constructs a generator for a specific mapped-dataset instance, used when a downstream
   * pass-through transform inherits its element type from a {@code map} receiver. The {@code
   * element} field is read off {@code mapResultInstance} directly, so the mapped type survives the
   * pass-through. wala/ML#649.
   *
   * @param node The {@code map.do()} node that allocated the instance.
   * @param mapResultInstance The mapped-dataset instance carrying the {@code element} field.
   */
  public DatasetMapGenerator(CGNode node, AllocationSiteInNode mapResultInstance) {
    super(node);
    this.mapResultInstance = mapResultInstance;
  }

  /**
   * Returns the instances whose {@code element} field holds the mapped type: the source variable's
   * points-to set, or the single {@code mapResultInstance} when this generator was resolved from a
   * receiver instance.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The mapped-dataset instances, or {@code null} if neither is available.
   */
  private OrdinalSet<InstanceKey> selfInstances(PropagationCallGraphBuilder builder) {
    if (this.getSource() != null) {
      return builder.getPointerAnalysis().getPointsToSet(this.getSource().getPointerKey());
    }
    if (this.mapResultInstance != null) {
      return OrdinalSet.toOrdinalSet(
          Collections.singleton((InstanceKey) this.mapResultInstance),
          builder.getPointerAnalysis().getInstanceKeyMapping());
    }
    return null;
  }

  /**
   * Returns the points-to set of the {@code element} field of the mapped dataset, i.e. of {@code
   * map_func}'s return value that {@code map.do()} stored there.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The points-to set of the mapped element, or {@code null} if it cannot be resolved.
   */
  private OrdinalSet<InstanceKey> getMappedElementPointsToSet(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    OrdinalSet<InstanceKey> selfPTS = selfInstances(builder);
    if (selfPTS == null || selfPTS.isEmpty()) return null;

    IField field =
        builder
            .getClassHierarchy()
            .resolveField(
                FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(ELEMENT_FIELD), Root));
    if (field == null) return null;

    OrdinalSet<InstanceKey> ret = OrdinalSet.empty();
    for (InstanceKey ik : selfPTS) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      OrdinalSet<InstanceKey> fieldPTS =
          pa.getPointsToSet(builder.getPointerKeyForInstanceField(asin, field));
      if (fieldPTS != null) ret = OrdinalSet.unify(ret, fieldPTS);
    }
    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> elementPTS = getMappedElementPointsToSet(builder);
    if (elementPTS != null && !elementPTS.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, elementPTS);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    // No mapped element resolved; fall back to the receiver's element shapes.
    return super.getDefaultShapes(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> elementPTS = getMappedElementPointsToSet(builder);
    if (elementPTS != null && !elementPTS.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, elementPTS);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    // No mapped element resolved; fall back to the receiver's element dtypes.
    return super.getDefaultDTypes(builder);
  }

  /**
   * Returns the points-to set of the {@code index}-th component of the mapped element, when {@code
   * map_func} returns a tuple (e.g. {@code return inputs, targets}).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param index The 0-based tuple-component index.
   * @return The points-to set of that component, or {@code null} if it cannot be resolved.
   */
  private OrdinalSet<InstanceKey> getMappedElementComponentPointsToSet(
      PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> elementPTS = getMappedElementPointsToSet(builder);
    if (elementPTS == null || elementPTS.isEmpty()) return null;
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    IField field =
        builder
            .getClassHierarchy()
            .resolveField(
                FieldReference.findOrCreate(
                    Root, findOrCreateAsciiAtom(String.valueOf(index)), Root));
    if (field == null) return null;
    OrdinalSet<InstanceKey> ret = OrdinalSet.empty();
    for (InstanceKey ik : elementPTS) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      OrdinalSet<InstanceKey> componentPTS =
          pa.getPointsToSet(builder.getPointerKeyForInstanceField(asin, field));
      if (componentPTS != null) ret = OrdinalSet.unify(ret, componentPTS);
    }
    return ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> elementPTS = getMappedElementPointsToSet(builder);
    if (elementPTS != null && !elementPTS.isEmpty()) {
      for (InstanceKey ik : elementPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null) continue;
        TypeReference reference = asin.concreteType().getReference();
        if (reference.equals(tuple) || reference.equals(list)) return true;
      }
    }
    return super.yieldsTuple(builder);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> componentPTS = getMappedElementComponentPointsToSet(builder, index);
    if (componentPTS != null && !componentPTS.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, componentPTS);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }

    // A computed tuple member (e.g. a binary-op result in `map_func`) has no allocation, so the
    // tuple field's points-to set is empty; walk the callback's store into the field at the SSA
    // level instead (wala/ML#688).
    Set<List<Dimension<?>>> ssaShapes = HashSetFactory.make();
    for (Pair<CGNode, Integer> componentValue :
        getMappedElementComponentValueNumbers(builder, index)) {
      Set<List<Dimension<?>>> shapes =
          this.getShapes(builder, componentValue.fst, componentValue.snd);
      if (shapes != null) ssaShapes.addAll(shapes);
    }
    if (!ssaShapes.isEmpty()) return ssaShapes;

    return super.getShapesForIndex(builder, index);
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> componentPTS = getMappedElementComponentPointsToSet(builder, index);
    if (componentPTS != null && !componentPTS.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, componentPTS);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }

    // The SSA fallback for computed tuple members; see getShapesForIndex (wala/ML#688).
    Set<DType> ssaDTypes = EnumSet.noneOf(DType.class);
    for (Pair<CGNode, Integer> componentValue :
        getMappedElementComponentValueNumbers(builder, index)) {
      Set<DType> dtypes = this.getDTypes(builder, componentValue.fst, componentValue.snd);
      if (dtypes != null) ssaDTypes.addAll(dtypes);
    }
    if (!ssaDTypes.isEmpty()) return ssaDTypes;

    return super.getDTypesForIndex(builder, index);
  }

  /**
   * Finds the SSA values stored into the given numeric field of the mapped element tuple, per
   * containing node: for each tuple allocation flowing out of {@code map_func}, locates its
   * allocation's definition and the instruction storing {@code <field index>} on it, and returns
   * the stored value number. This recovers computed tuple members (binary-op results and other
   * allocation-less values) whose points-to sets are empty (<a
   * href="https://github.com/wala/ML/issues/688">wala/ML#688</a>).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @param index The tuple index whose stores to find.
   * @return The (node, value number) pairs stored into the field; empty if none resolve.
   */
  private Set<Pair<CGNode, Integer>> getMappedElementComponentValueNumbers(
      PropagationCallGraphBuilder builder, int index) {
    Set<Pair<CGNode, Integer>> ret = HashSetFactory.make();
    OrdinalSet<InstanceKey> elementPTS = getMappedElementPointsToSet(builder);
    if (elementPTS == null || elementPTS.isEmpty()) return ret;

    String fieldName = String.valueOf(index);
    for (InstanceKey ik : elementPTS) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      CGNode node = asin.getNode();
      IR ir = node.getIR();
      if (ir == null) continue;

      // Find the allocation's definition.
      int allocVn = -1;
      for (SSAInstruction inst : ir.getInstructions())
        if (inst instanceof SSANewInstruction newInst
            && newInst.getNewSite().equals(asin.getSite())) {
          allocVn = newInst.getDef();
          break;
        }
      if (allocVn < 0) continue;

      // Find the stores into `<field index>` on that allocation.
      for (SSAInstruction inst : ir.getInstructions()) {
        if (inst instanceof SSAPutInstruction put
            && !put.isStatic()
            && put.getRef() == allocVn
            && put.getDeclaredField().getName().toString().equals(fieldName))
          ret.add(Pair.make(node, put.getVal()));
        else if (inst instanceof AstPropertyWrite write && write.getObjectRef() == allocVn) {
          SymbolTable symbolTable = ir.getSymbolTable();
          int memberVn = write.getMemberRef();
          // The member constant may live in the symbol table or flow as a ConstantKey through the
          // points-to set; check both, mirroring TensorGeneratorFactory.constantMemberEquals.
          if ((symbolTable.isConstant(memberVn)
                  && fieldName.equals(String.valueOf(symbolTable.getConstantValue(memberVn))))
              || TensorGeneratorFactory.constantMemberEquals(node, memberVn, index, builder))
            ret.add(Pair.make(node, write.getValue()));
        }
      }
    }
    return ret;
  }
}
