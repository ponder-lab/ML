package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by {@code tf.data.Dataset.from_tensor_slices}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetFromTensorSlicesGenerator extends DatasetGenerator
    implements TupleElementProvider {

  private static final Logger LOGGER =
      Logger.getLogger(DatasetFromTensorSlicesGenerator.class.getName());

  protected enum Parameters {
    TENSORS,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DatasetFromTensorSlicesGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetFromTensorSlicesGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For tf.data.Dataset.from_tensor_slices(tensors), the dataset elements are created by
    // slicing the input tensors along their first dimension. Thus, the element shapes are
    // the input shapes with the first dimension removed.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    Set<List<Dimension<?>>> inputShapes = null;
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      inputShapes = this.getShapesOfTensorsArgument(builder, tensorsPTS);
    }
    final int tensorsPTSSize = tensorsPTS == null ? -1 : tensorsPTS.size();
    final Set<List<Dimension<?>>> ptsPathShapes = inputShapes;
    LOGGER.fine(
        () ->
            "DatasetFromTensorSlicesGenerator.getDefaultShapes: source="
                + this.getSource()
                + ", tensorsPTS size="
                + tensorsPTSSize
                + ", inputShapes via pts-path="
                + (ptsPathShapes == null ? "null" : ptsPathShapes.size() + " shapes"));

    // Fallback: if the points-to set for the argument is empty (e.g., `tensors` is the result of
    // a Python binary op, for which WALA does not allocate a trackable target), walk the call
    // string to resolve the argument value number in each caller and delegate to getShapes, which
    // knows how to construct an ElementWiseOperation generator for binop-def'd locals.
    if (inputShapes == null || inputShapes.isEmpty()) {
      inputShapes =
          this.getArgumentShapesViaCallers(
              builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
      final Set<List<Dimension<?>>> fallbackShapes = inputShapes;
      LOGGER.fine(
          () ->
              "DatasetFromTensorSlicesGenerator.getDefaultShapes: fallback inputShapes="
                  + (fallbackShapes == null ? "null" : fallbackShapes.size() + " shapes"));
    }

    if (inputShapes == null) return null;
    if (inputShapes.isEmpty()) return Collections.emptySet();

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      if (shape.size() > 0) {
        // Remove the first dimension to account for slicing.
        ret.add(new ArrayList<>(shape.subList(1, shape.size())));
      } else {
        // If the input is already a scalar (unexpected for from_tensor_slices),
        // the element shape is empty.
        ret.add(Collections.emptyList());
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For from_tensor_slices, element dtypes are the same as the input tensor(s)' dtypes.
    // The 'tensors' argument is at position 0 (args: this, tensors, name).
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfTensorsArgument(builder, tensorsPTS);
      if (!dtypes.isEmpty()) return dtypes;
    }

    // Fallback: walk the call string to resolve the argument in each caller.
    Set<DType> fallback =
        this.getArgumentDTypesViaCallers(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    if (fallback != null && !fallback.isEmpty()) return fallback;

    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    Set<List<Dimension<?>>> inputShapes = null;
    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      inputShapes = this.getShapesOfTensorsArgument(builder, tensorsPTS);
    }

    // Fallback mirroring `getDefaultShapes`: when the `tensors` argument has empty PTS (e.g.,
    // it's the SSA result of a binop chain feeding `from_tensor_slices`), walk callers to
    // resolve its shape. Without this, partial-batch size computation in
    // `DatasetBatchGenerator` degrades to a symbolic `?` even though the dataset's size is
    // statically knowable. See wala/ML#357.
    if (inputShapes == null || inputShapes.isEmpty()) {
      inputShapes =
          this.getArgumentShapesViaCallers(
              builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());
    }

    if (inputShapes == null) return Collections.emptySet();

    Set<Long> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : inputShapes) {
      if (!shape.isEmpty()) {
        Dimension<?> firstDim = shape.get(0);
        if (firstDim instanceof NumericDim) {
          ret.add(Long.valueOf(((NumericDim) firstDim).value()));
        }
      }
    }
    return ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
          return true;
        }
      }
    }
    return false;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null && fieldIndex == index) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.getConcreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                Set<List<Dimension<?>>> fieldShapes =
                    this.getShapesOfValue(builder, builder.getPointerAnalysis().getPointsToSet(pk));
                if (fieldShapes != null)
                  for (List<Dimension<?>> shape : fieldShapes) {
                    if (shape.size() > 0) {
                      ret.add(new ArrayList<>(shape.subList(1, shape.size())));
                    } else {
                      ret.add(Collections.emptyList());
                    }
                  }
              }
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
      // Per-index lookup saw a tuple but the requested field's PTS was empty — return unknown (⊤)
      // rather than falling through to the aggregate, which would silently return sibling fields'
      // shapes. See wala/ML#396.
      return null;
    }
    return this.getShapes(builder);
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    OrdinalSet<InstanceKey> tensorsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.TENSORS.getIndex(), Parameters.TENSORS.getName());

    if (tensorsPTS != null && !tensorsPTS.isEmpty()) {
      Set<DType> ret = HashSetFactory.make();
      for (InstanceKey ik : tensorsPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              builder
                  .getPointerAnalysis()
                  .getPointsToSet(
                      ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                          .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            Integer fieldIndex = getFieldIndex((ConstantKey<?>) catalogIK);
            if (fieldIndex != null && fieldIndex == index) {
              FieldReference subscript =
                  FieldReference.findOrCreate(
                      asin.getConcreteType().getReference(),
                      findOrCreateAsciiAtom(fieldIndex.toString()),
                      Root);
              IField f = builder.getClassHierarchy().resolveField(subscript);
              if (f != null) {
                PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
                Set<DType> fieldDTypes =
                    this.getDTypesOfValue(builder, builder.getPointerAnalysis().getPointsToSet(pk));
                if (fieldDTypes != null) ret.addAll(fieldDTypes);
              }
            }
          }
        }
      }
      if (!ret.isEmpty()) {
        return ret;
      }
      // Per-index lookup saw a tuple but the requested field's PTS was empty — return UNKNOWN (⊤)
      // rather than falling through to the aggregate. See wala/ML#396.
      return EnumSet.of(DType.UNKNOWN);
    }
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
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), null));
      return ret;
    }

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    return ret;
  }

  /**
   * Shape-walk helper for the {@code tensors} argument of {@code
   * tf.data.Dataset.from_tensor_slices}.
   *
   * <p>{@code from_tensor_slices} accepts a "nested structure" argument: when the argument is a
   * tuple like {@code (a, b)}, the two elements are not the outer dimensions of a single tensor —
   * they are independent tensors bundled together, and the dataset yields tuples of per-element
   * slices. The base {@link TensorGenerator#getShapesOfValue} treats tuple and list identically
   * (both as multi-dim tensor values), which is the right interpretation for {@code
   * tf.convert_to_tensor((1, 2, 3))} and for nested-inside-list cases like {@code [(7, 8), (9,
   * 10)]}, but wrong for a top-level tuple argument to this specific API.
   *
   * <p>This helper special-cases a top-level tuple: walk each field independently and union the
   * per-field shapes (obtained via the base {@link TensorGenerator#getShapesOfValue}, which still
   * treats nested tuples as multi-dim). For non-tuple top-level values it falls through to the base
   * implementation unchanged, so list literals and single-tensor arguments keep their current
   * behavior. See wala/ML#366.
   *
   * @param builder The propagation call graph builder used to resolve field pointer keys.
   * @param valuePointsToSet The points-to set of the {@code tensors} argument.
   * @return The union of per-field shapes when the argument is a top-level tuple, or the base
   *     {@link TensorGenerator#getShapesOfValue} result otherwise.
   */
  private Set<List<Dimension<?>>> getShapesOfTensorsArgument(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    boolean sawTuple = false;

    for (InstanceKey ik : valuePointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null || !asin.getConcreteType().getReference().equals(tuple)) {
        continue;
      }
      sawTuple = true;
      OrdinalSet<InstanceKey> objectCatalogPointsToSet =
          builder
              .getPointerAnalysis()
              .getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

      LOGGER.fine(
          DatasetFromTensorSlicesGenerator.class.getName()
              + ".getShapesOfTensorsArgument: top-level tuple catalog size="
              + objectCatalogPointsToSet.size());

      for (InstanceKey catalogIK : objectCatalogPointsToSet) {
        Integer fieldIndex =
            getFieldIndex((com.ibm.wala.ipa.callgraph.propagation.ConstantKey<?>) catalogIK);
        if (fieldIndex == null) continue;
        FieldReference subscript =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
        IField f = builder.getClassHierarchy().resolveField(subscript);
        if (f == null) continue;
        PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
        OrdinalSet<InstanceKey> fieldPts = builder.getPointerAnalysis().getPointsToSet(pk);
        Set<List<Dimension<?>>> fieldShapes = this.getShapesOfValue(builder, fieldPts);
        if (fieldShapes == null || fieldShapes.isEmpty()) {
          // Fallback: fieldPts is empty because the stored value has an implicit PK (e.g.,
          // a division-result post reshape chain: `x_train / 255.0`). Locate the tuple-field
          // putfield in the allocation node's IR and walk the stored vn via the shared
          // SSA-chain helper. See wala/WALA#1889.
          int storedVn = findTupleFieldStoreForIndex(asin.getNode(), asin, fieldIndex, builder);
          if (storedVn > 0) {
            try {
              Set<List<Dimension<?>>> viaChain =
                  this.getShapesOrSSAChain(builder, asin.getNode(), storedVn);
              if (viaChain != null && !viaChain.isEmpty()) {
                fieldShapes = viaChain;
                final int fi = fieldIndex;
                LOGGER.fine(
                    () ->
                        "getShapesOfTensorsArgument: recovered field="
                            + fi
                            + " via SSA-chain on storedVn="
                            + storedVn);
              }
            } catch (IllegalArgumentException e) {
              // leave as null/empty
            }
          }
        }
        if (fieldShapes != null) ret.addAll(fieldShapes);
      }
    }

    if (!sawTuple) {
      return this.getShapesOfValue(builder, valuePointsToSet);
    }
    return ret;
  }

  /**
   * Locates a {@link PythonPropertyWrite} in {@code node}'s IR that stores to the tuple-like
   * allocation {@code asin} at the given {@code fieldIndex}. Returns the stored SSA value number.
   *
   * <p>Used as a fallback when the tuple-field's points-to set is empty (the stored value has an
   * implicit PK from a summary-method return). The stored vn's shape can then be recovered via
   * {@link #shapesFromSSAChain(PropagationCallGraphBuilder, CGNode, int)}.
   *
   * @param node The CG node whose IR to scan.
   * @param asin The tuple-like allocation whose field was stored to.
   * @param fieldIndex The integer field index ({@code 0}, {@code 1}, ...).
   * @param builder The propagation call graph builder for PTS lookups.
   * @return The stored value's SSA value number, or {@code -1} if no matching store was found.
   */
  private static int findTupleFieldStoreForIndex(
      CGNode node, AllocationSiteInNode asin, int fieldIndex, PropagationCallGraphBuilder builder) {
    // Match writes by PTS (not by SSA vn) because the tuple's allocation site and the
    // putfield's objectRef may be different SSA values even in the same node — both point
    // to `asin` via PA but carry different vns. Same reasoning for the member constant.
    String targetMember = String.valueOf(fieldIndex);
    int found = -1;
    for (SSAInstruction inst : node.getIR().getInstructions()) {
      if (!(inst instanceof PythonPropertyWrite)) continue;
      PythonPropertyWrite write = (PythonPropertyWrite) inst;

      // Phase 1: does the write target `asin`? Check via PTS membership on the objectRef.
      int objectVn = write.getUse(0);
      PointerKey objPk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, objectVn);
      OrdinalSet<InstanceKey> objPts = builder.getPointerAnalysis().getPointsToSet(objPk);
      if (objPts.isEmpty()) continue;
      boolean matchesAsin = false;
      for (InstanceKey ik : objPts) {
        if (ik.equals(asin)) {
          matchesAsin = true;
          break;
        }
      }
      if (!matchesAsin) continue;

      // Phase 2: does the write's member match `fieldIndex`? Python tuple-store lowers the
      // field index (0, 1, ...) to a `ConstantKey<Integer>` (or `<String>` of the digits)
      // pointed to by the memberRef vn. Compare by `toString` so both representations match.
      int memberVn = write.getUse(1);
      PointerKey memberPk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, memberVn);
      boolean memberMatches = false;
      for (InstanceKey memberIk : builder.getPointerAnalysis().getPointsToSet(memberPk)) {
        if (!(memberIk instanceof ConstantKey)) continue;
        Object value = ((ConstantKey<?>) memberIk).getValue();
        if (value != null && targetMember.equals(value.toString())) {
          memberMatches = true;
          break;
        }
      }
      if (!memberMatches) continue;

      // Phase 3: record the stored value. If we see multiple distinct stores to the same
      // field (unlikely in straight-line tuple construction, but possible with conditional
      // writes), treat as ambiguous — the caller falls back to ⊤ rather than pick one.
      int stored = write.getUse(2);
      if (found != -1 && found != stored) return -1;
      found = stored;
    }
    return found;
  }

  /**
   * Dtype counterpart of {@link #getShapesOfTensorsArgument}. See that method for the rationale.
   *
   * @param builder The propagation call graph builder used to resolve field pointer keys.
   * @param valuePointsToSet The points-to set of the {@code tensors} argument.
   * @return The union of per-field dtypes when the argument is a top-level tuple, or the base
   *     {@link TensorGenerator#getDTypesOfValue} result otherwise.
   */
  private Set<DType> getDTypesOfTensorsArgument(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    boolean sawTuple = false;

    for (InstanceKey ik : valuePointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null || !asin.getConcreteType().getReference().equals(tuple)) {
        continue;
      }
      sawTuple = true;
      OrdinalSet<InstanceKey> objectCatalogPointsToSet =
          builder
              .getPointerAnalysis()
              .getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

      for (InstanceKey catalogIK : objectCatalogPointsToSet) {
        Integer fieldIndex =
            getFieldIndex((com.ibm.wala.ipa.callgraph.propagation.ConstantKey<?>) catalogIK);
        if (fieldIndex == null) continue;
        FieldReference subscript =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
        IField f = builder.getClassHierarchy().resolveField(subscript);
        if (f == null) continue;
        PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
        OrdinalSet<InstanceKey> fieldPts = builder.getPointerAnalysis().getPointsToSet(pk);
        Set<DType> fieldDTypes = this.getDTypesOfValue(builder, fieldPts);
        if (fieldDTypes == null || fieldDTypes.isEmpty()) {
          // Same fallback as for shapes: implicit-PK stored value, walk DU chain.
          int storedVn = findTupleFieldStoreForIndex(asin.getNode(), asin, fieldIndex, builder);
          if (storedVn > 0) {
            try {
              Set<DType> viaChain = this.getDTypesOrSSAChain(builder, asin.getNode(), storedVn);
              if (viaChain != null && !viaChain.isEmpty()) fieldDTypes = viaChain;
            } catch (IllegalArgumentException e) {
              // leave as null/empty
            }
          }
        }
        if (fieldDTypes != null) ret.addAll(fieldDTypes);
      }
    }

    if (!sawTuple) {
      return this.getDTypesOfValue(builder, valuePointsToSet);
    }
    return ret;
  }
}
