package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.ELLIPSIS;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Modeling of numpy-style ndarray subscript for the special case of ellipsis + newaxis patterns
 * that add dimensions without removing any (e.g., {@code x[..., None]}, {@code x[None, ...]},
 * {@code x[None]}).
 *
 * <p>Only the dimension-adding subset of subscript semantics is modeled here. Subscripts that
 * consume dimensions (integer indices, slices, boolean masks) are left unhandled &mdash; this class
 * returns {@code null} (⊤ unknown shape) for those cases so downstream analyses can fall through to
 * existing behavior.
 *
 * <p>Ellipsis is distinguished from {@code None} via the {@code PythonTypes.ELLIPSIS} sentinel
 * emitted by the parser's {@code visitEllipsis}. Without that distinction the analysis would be
 * unable to tell {@code x[..., None]} from {@code x[None, None]} (see wala/ML#356).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NdarraySubscriptOperation extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(NdarraySubscriptOperation.class.getName());

  public NdarraySubscriptOperation(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Checks whether {@code source}'s defining instruction is a {@link PythonPropertyRead} whose
   * member ref is a tuple allocation that contains only ellipsis and {@code None} elements (i.e., a
   * dim-adding subscript). Callers use this to decide whether to dispatch to this generator.
   */
  public static boolean isApplicable(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    PythonPropertyRead propRead = getPropertyRead(source);
    if (propRead == null) return false;
    CGNode node = ((LocalPointerKey) source.getPointerKey()).getNode();
    List<SubscriptField> fields = extractSubscriptFields(propRead, node, builder);
    return fields != null && !fields.isEmpty();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    PythonPropertyRead propRead = getPropertyRead(source);
    if (propRead == null) return null;

    List<SubscriptField> fields = extractSubscriptFields(propRead, getNode(), builder);
    if (fields == null) {
      LOGGER.fine(() -> "NdarraySubscriptOperation: unsupported subscript pattern for " + source);
      return null;
    }

    int objRef = propRead.getObjectRef();
    Set<List<Dimension<?>>> receiverShapes = getShapes(builder, getNode(), objRef);
    LOGGER.fine(
        () -> "NdarraySubscriptOperation: receiver vn=" + objRef + " shapes=" + receiverShapes);
    if (receiverShapes == null || receiverShapes.isEmpty()) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> receiverShape : receiverShapes) {
      List<Dimension<?>> outputShape = applySubscript(receiverShape, fields);
      if (outputShape == null) return null;
      ret.add(outputShape);
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    PythonPropertyRead propRead = getPropertyRead(source);
    if (propRead == null) return null;
    int objRef = propRead.getObjectRef();
    Set<DType> ret = getDTypes(builder, objRef);
    return ret.isEmpty() ? null : ret;
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
   * Applies the parsed subscript pattern to the input shape. Returns {@code null} for patterns that
   * consume existing dimensions (not in scope for this generator).
   *
   * <p>Semantics (matching numpy):
   *
   * <ul>
   *   <li>{@code None} (newaxis) at position p inserts a size-1 dim at p in the output.
   *   <li>{@code ...} (ellipsis) expands to fill the remaining unconsumed input dims at its
   *       position.
   *   <li>If no ellipsis is present, one is implicitly appended (i.e., {@code x[None]} behaves like
   *       {@code x[None, ...]}).
   * </ul>
   */
  private static List<Dimension<?>> applySubscript(
      List<Dimension<?>> input, List<SubscriptField> fields) {
    int ellipsisCount = 0;
    for (SubscriptField f : fields) if (f == SubscriptField.ELLIPSIS) ellipsisCount++;
    if (ellipsisCount > 1) return null;

    List<Dimension<?>> out = new ArrayList<>();
    boolean ellipsisSeen = false;
    for (SubscriptField f : fields) {
      switch (f) {
        case ELLIPSIS:
          out.addAll(input);
          ellipsisSeen = true;
          break;
        case NEWAXIS:
          out.add(new NumericDim(1));
          break;
      }
    }
    if (!ellipsisSeen) {
      // Implicit trailing ellipsis: append any remaining input dims.
      out.addAll(input);
    }
    return out;
  }

  /**
   * Extracts the subscript tuple's fields if they are all ellipsis or {@code None}. Returns {@code
   * null} if the subscript contains anything else (integer index, slice, etc.), signaling that this
   * generator does not apply.
   */
  private static List<SubscriptField> extractSubscriptFields(
      PythonPropertyRead propRead, CGNode node, PropagationCallGraphBuilder builder) {
    int memberVn = propRead.getMemberRef();
    PointerKey memberKey =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, memberVn);
    OrdinalSet<InstanceKey> memberPts = builder.getPointerAnalysis().getPointsToSet(memberKey);

    AllocationSiteInNode tupleAsin = null;
    for (InstanceKey ik : memberPts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin != null && asin.getConcreteType().getReference().equals(tuple)) {
        tupleAsin = asin;
        break;
      }
    }
    if (tupleAsin == null) return null;

    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    AstPointerKeyFactory pkf = (AstPointerKeyFactory) builder.getPointerKeyFactory();
    OrdinalSet<InstanceKey> catalog =
        pa.getPointsToSet(pkf.getPointerKeyForObjectCatalog(tupleAsin));
    int fieldCount = catalog.size();
    if (fieldCount == 0) return null;

    // The tuple's fields are indexed by name "0", "1", ...; we need them in order.
    List<SubscriptField> out = new ArrayList<>(fieldCount);
    for (int i = 0; i < fieldCount; i++) out.add(null);

    for (InstanceKey catalogIK : catalog) {
      ConstantKey<?> ck = (ConstantKey<?>) catalogIK;
      Integer fieldIndex = getFieldIndex(ck);
      if (fieldIndex == null || fieldIndex < 0 || fieldIndex >= fieldCount) return null;

      FieldReference subscriptField =
          FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
      IField f = builder.getClassHierarchy().resolveField(subscriptField);
      if (f == null) return null;

      PointerKey fieldPk = builder.getPointerKeyForInstanceField(tupleAsin, f);
      OrdinalSet<InstanceKey> fieldPts = pa.getPointsToSet(fieldPk);

      SubscriptField classified = classifyField(fieldPts);
      if (classified == null) return null; // unsupported element
      out.set(fieldIndex, classified);
    }
    for (SubscriptField f : out) if (f == null) return null;
    return out;
  }

  /**
   * Classifies a single subscript tuple element as ellipsis, newaxis ({@code None}), or an
   * unsupported value (integer index, slice, variable, etc.).
   */
  private static SubscriptField classifyField(OrdinalSet<InstanceKey> pts) {
    if (pts.isEmpty()) return null; // empty PTS is not None; unknown value
    boolean sawEllipsis = false;
    boolean sawNone = false;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object v = ((ConstantKey<?>) ik).getValue();
      if (ELLIPSIS.equals(v)) sawEllipsis = true;
      else if (v == null) sawNone = true;
      else return null; // integer index, string, etc. — not supported
    }
    if (sawEllipsis && sawNone) return null; // ambiguous; bail
    if (sawEllipsis) return SubscriptField.ELLIPSIS;
    if (sawNone) return SubscriptField.NEWAXIS;
    return null;
  }

  /**
   * Returns the {@link PythonPropertyRead} that defines {@code source}'s value number, or {@code
   * null} if {@code source} isn't a {@link LocalPointerKey} or its defining instruction isn't a
   * property read. Shared by {@link #isApplicable(PointsToSetVariable,
   * PropagationCallGraphBuilder)} and the {@code getDefault*} overrides; callers treat a {@code
   * null} return as "this generator does not apply."
   */
  private static PythonPropertyRead getPropertyRead(PointsToSetVariable source) {
    if (!(source.getPointerKey() instanceof LocalPointerKey)) return null;
    LocalPointerKey lpk = (LocalPointerKey) source.getPointerKey();
    SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
    return def instanceof PythonPropertyRead ? (PythonPropertyRead) def : null;
  }

  private enum SubscriptField {
    ELLIPSIS,
    NEWAXIS
  }
}
