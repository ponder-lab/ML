package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.ELLIPSIS;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
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
   *
   * <p>TODO(wala/WALA#1889): This syntactic dispatch pattern exists to work around WALA
   * representing synthetic-method return values as implicit {@code PointerKey}s, which breaks
   * class-type dispatch through the call graph for post-unpack receivers. Once that upstream bug is
   * fixed, this whole static predicate and its associated dispatch site can be deleted.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is being inspected.
   * @param builder The propagation call graph builder used to resolve the subscript's member ref.
   * @return {@code true} if this generator's modeling applies to {@code source}, {@code false}
   *     otherwise.
   */
  public static boolean isApplicable(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    PythonPropertyRead propRead = getPropertyRead(source);
    if (propRead == null) return false;
    CGNode node = ((LocalPointerKey) source.getPointerKey()).getNode();
    List<SubscriptField> fields = extractSubscriptFields(propRead, node, builder);
    boolean applies = fields != null && !fields.isEmpty();
    LOGGER.fine(
        () ->
            "isApplicable: source="
                + source
                + " propRead="
                + propRead
                + " fields="
                + fields
                + " applies="
                + applies);
    return applies;
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
   *
   * @param input The receiver's shape dims, in order.
   * @param fields The parsed subscript tuple elements, in order.
   * @return The output shape dims, or {@code null} if the subscript contains a dimension-consuming
   *     element (more than one ellipsis, or any shape not handled by this generator).
   */
  private static List<Dimension<?>> applySubscript(
      List<Dimension<?>> input, List<SubscriptField> fields) {
    int ellipsisCount = 0;
    for (SubscriptField f : fields) if (f instanceof Ellipsis) ellipsisCount++;
    if (ellipsisCount > 1) return null;

    List<Dimension<?>> out = new ArrayList<>();
    int inputConsumed = 0;
    boolean ellipsisSeen = false;
    for (SubscriptField f : fields) {
      if (f instanceof Ellipsis) {
        out.addAll(input);
        inputConsumed = input.size();
        ellipsisSeen = true;
      } else if (f instanceof Newaxis) {
        out.add(new NumericDim(1));
      } else if (f instanceof Slice) {
        // `[:k]` with constant `k` and default start/step: replace the leading input dim with
        // `NumericDim(k)` and preserve the rest. Broader slice patterns (`[a:b]`, `[::s]`, etc.)
        // are out of scope — see wala/ML#406.
        Slice sl = (Slice) f;
        if (!sl.isCanonicalStopOnly() || inputConsumed >= input.size()) return null;
        out.add(new NumericDim(sl.stop));
        inputConsumed++;
      } else {
        return null;
      }
    }
    if (!ellipsisSeen) {
      // Implicit trailing ellipsis: append any remaining input dims.
      for (int i = inputConsumed; i < input.size(); i++) out.add(input.get(i));
    }
    return out;
  }

  /**
   * Extracts the subscript tuple's fields if they are all ellipsis or {@code None}. Returns {@code
   * null} if the subscript contains anything else (integer index, slice, etc.), signaling that this
   * generator does not apply.
   *
   * @param propRead The subscript's {@link PythonPropertyRead} instruction.
   * @param node The {@link CGNode} containing {@code propRead}.
   * @param builder The propagation call graph builder used to resolve the subscript's member ref.
   * @return The parsed subscript tuple elements in order, or {@code null} if any element is
   *     unsupported (integer index, slice, variable, etc.).
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
    if (tupleAsin == null) {
      StringBuilder ptsDesc = new StringBuilder();
      for (InstanceKey ik : memberPts) {
        ptsDesc.append(" ").append(ik.getClass().getSimpleName());
        AllocationSiteInNode a = getAllocationSiteInNode(ik);
        if (a != null) ptsDesc.append("@").append(a.getConcreteType().getReference().getName());
      }
      LOGGER.fine(
          () ->
              "extractSubscriptFields: memberVn="
                  + memberVn
                  + " no tuple alloc in PTS (size="
                  + memberPts.size()
                  + ") — pts-kinds:"
                  + ptsDesc);
      return null;
    }

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

      SubscriptField classified = classifyField(fieldPts, builder);
      if (classified == null) return null; // unsupported element
      out.set(fieldIndex, classified);
    }
    for (SubscriptField f : out) if (f == null) return null;
    return out;
  }

  /**
   * Classifies a single subscript tuple element as ellipsis, newaxis ({@code None} or {@code
   * tf.newaxis}), a canonical {@code [:k]} slice (three-element list with {@code None} / constant
   * integer / {@code None}), or an unsupported value.
   *
   * <p>Recognises {@code tf.newaxis} via its modeled type {@link TensorFlowTypes#NEWAXIS} (see
   * {@code tensorflow.xml}'s {@code <new def="newaxis" class="Ltensorflow/newaxis">}). At Python
   * runtime {@code tf.newaxis is None}, but WALA sees it as an attribute-access result with a
   * synthetic allocation rather than a {@code ConstantKey<null>} — hence the separate branch.
   *
   * <p>Recognises slice expressions via the front-end's lowering of {@code x[lower:upper:step]}
   * into {@code OBJECT_REF(x, tuple(list(lower, upper, step)))}: the tuple element's PTS contains a
   * {@link com.ibm.wala.cast.python.types.PythonTypes#list} allocation whose fields {@code 0},
   * {@code 1}, {@code 2} hold the bounds. Only the canonical {@code [:k]} shape (start/step both
   * {@code None}, stop a constant integer) is accepted; other slice vocabulary is out of scope per
   * wala/ML#406.
   *
   * @param pts The points-to set of the tuple element being classified.
   * @param builder The propagation call graph builder, needed to resolve the list's field PTS when
   *     the element is a slice literal.
   * @return An {@link Ellipsis}, {@link Newaxis}, or {@link Slice} instance for a supported value;
   *     {@code null} if the element is empty, non-constant, or any other unsupported value.
   */
  private static SubscriptField classifyField(
      OrdinalSet<InstanceKey> pts, PropagationCallGraphBuilder builder) {
    if (pts.isEmpty()) return null; // empty PTS is not None; unknown value
    boolean sawEllipsis = false;
    boolean sawNone = false;
    Slice sawSlice = null;
    for (InstanceKey ik : pts) {
      if (ik instanceof ConstantKey) {
        Object v = ((ConstantKey<?>) ik).getValue();
        if (ELLIPSIS.equals(v)) {
          sawEllipsis = true;
          continue;
        }
        if (v == null) {
          sawNone = true;
          continue;
        }
        return null; // integer index, string, etc. — not supported
      }
      // Non-constant InstanceKey: `tf.newaxis` allocation (modeled) or a slice literal list.
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin != null && asin.getConcreteType().getReference().equals(TensorFlowTypes.NEWAXIS)) {
        sawNone = true;
        continue;
      }
      if (asin != null && asin.getConcreteType().getReference().equals(list)) {
        Slice slice = classifyListAsSlice(asin, builder);
        if (slice == null) return null;
        if (sawSlice != null) return null; // multiple list allocations — ambiguous.
        sawSlice = slice;
        continue;
      }
      return null; // unknown non-constant — bail
    }
    int kinds = (sawEllipsis ? 1 : 0) + (sawNone ? 1 : 0) + (sawSlice != null ? 1 : 0);
    if (kinds != 1) return null; // empty or ambiguous mix
    if (sawEllipsis) return Ellipsis.INSTANCE;
    if (sawNone) return Newaxis.INSTANCE;
    return sawSlice;
  }

  /**
   * Interprets a {@code list} allocation with fields {@code 0}, {@code 1}, {@code 2} as a slice
   * literal {@code [lower, upper, step]}. Returns a {@link Slice} only for the canonical {@code
   * [:k]} shape &mdash; {@code lower} is {@code None}, {@code upper} is a constant integer, and
   * {@code step} is either absent, {@code None}, or the constant {@code 1}.
   */
  private static Slice classifyListAsSlice(
      AllocationSiteInNode listAsin, PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pa = builder.getPointerAnalysis();
    OrdinalSet<InstanceKey> catalog =
        pa.getPointsToSet(
            ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                .getPointerKeyForObjectCatalog(listAsin));
    Integer[] fieldValues = new Integer[3]; // 0 = lower, 1 = upper, 2 = step
    boolean[] isNone = new boolean[3];
    boolean[] seen = new boolean[3];
    for (InstanceKey catalogIK : catalog) {
      ConstantKey<?> ck = (ConstantKey<?>) catalogIK;
      Integer fieldIndex = getFieldIndex(ck);
      if (fieldIndex == null || fieldIndex < 0 || fieldIndex > 2) return null;
      FieldReference fieldRef =
          FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);
      IField f = builder.getClassHierarchy().resolveField(fieldRef);
      if (f == null) return null;
      OrdinalSet<InstanceKey> fieldPts =
          pa.getPointsToSet(builder.getPointerKeyForInstanceField(listAsin, f));
      if (fieldPts == null) return null;
      seen[fieldIndex] = true;
      for (InstanceKey ik : fieldPts) {
        if (!(ik instanceof ConstantKey)) return null;
        Object v = ((ConstantKey<?>) ik).getValue();
        if (v == null) {
          isNone[fieldIndex] = true;
        } else if (v instanceof Integer) {
          if (fieldValues[fieldIndex] != null && !fieldValues[fieldIndex].equals(v)) return null;
          fieldValues[fieldIndex] = (Integer) v;
        } else if (v instanceof Long) {
          int asInt = ((Long) v).intValue();
          if (fieldValues[fieldIndex] != null && fieldValues[fieldIndex] != asInt) return null;
          fieldValues[fieldIndex] = asInt;
        } else {
          return null;
        }
      }
    }
    if (!seen[0] || !seen[1]) return null;
    // Canonical `[:k]` shape.
    boolean lowerOK = isNone[0] && fieldValues[0] == null;
    if (!lowerOK) return null;
    if (fieldValues[1] == null) return null; // non-constant upper bound
    boolean stepOK =
        !seen[2]
            || (isNone[2] && fieldValues[2] == null)
            || (fieldValues[2] != null && fieldValues[2] == 1);
    if (!stepOK) return null;
    return new Slice(fieldValues[1]);
  }

  /**
   * Returns the {@link PythonPropertyRead} that defines {@code source}'s value number.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is being inspected.
   * @return The defining {@link PythonPropertyRead}, or {@code null} if {@code source} isn't a
   *     {@link LocalPointerKey} or its defining instruction isn't a property read.
   */
  private static PythonPropertyRead getPropertyRead(PointsToSetVariable source) {
    if (!(source.getPointerKey() instanceof LocalPointerKey)) return null;
    LocalPointerKey lpk = (LocalPointerKey) source.getPointerKey();
    SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
    return def instanceof PythonPropertyRead ? (PythonPropertyRead) def : null;
  }

  /**
   * Kind of a single element in the subscript tuple. Concrete subclasses {@link Ellipsis}, {@link
   * Newaxis}, and {@link Slice} cover the currently-supported vocabulary. Any other subscript shape
   * (integer index, variable, non-canonical slice) causes {@link #classifyField(OrdinalSet,
   * PropagationCallGraphBuilder)} to return {@code null} and the generator declines to apply.
   */
  private abstract static class SubscriptField {}

  private static final class Ellipsis extends SubscriptField {
    static final Ellipsis INSTANCE = new Ellipsis();

    private Ellipsis() {}

    @Override
    public String toString() {
      return "...";
    }
  }

  private static final class Newaxis extends SubscriptField {
    static final Newaxis INSTANCE = new Newaxis();

    private Newaxis() {}

    @Override
    public String toString() {
      return "None";
    }
  }

  /**
   * A canonical {@code [:k]} slice: start and step are both {@code None}/{@code 0}/{@code 1} and
   * {@code stop} is a constant integer. Broader slice vocabulary (non-zero start, non-unit step,
   * non-constant bounds, negative bounds) is out of scope here &mdash; see wala/ML#406.
   */
  private static final class Slice extends SubscriptField {
    final int stop;

    Slice(int stop) {
      this.stop = stop;
    }

    boolean isCanonicalStopOnly() {
      return true;
    }

    @Override
    public String toString() {
      return ":" + stop;
    }
  }
}
