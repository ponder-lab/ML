package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for {@code tf.split(value, num_or_size_splits, axis=0, ...)} with an integer {@code
 * num_or_size_splits}: every piece has the {@code value} shape with the {@code axis} dimension
 * divided by the split count, so the single modeled piece stands for all of them. Output dtype is
 * inherited from {@code value}. A size-list {@code num_or_size_splits} produces differently-shaped
 * pieces the single-piece model cannot represent, and a non-constant count or axis leaves the
 * quotient unknown; in those cases the axis dimension degrades to dynamic (rank and the other
 * dimensions still transfer) or, without a resolvable input shape, the result is ⊤. See <a
 * href="https://github.com/wala/ML/issues/717">wala/ML#717</a>.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/split">tf.split</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Split extends PassThroughUnaryTensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(Split.class.getName());

  public Split(PointsToSetVariable source) {
    super(source);
  }

  public Split(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "value";
  }

  /**
   * Derives each piece's shape from the {@code value} (arg 0) shape: the {@code axis} (arg 2,
   * default 0) dimension is divided by the integer {@code num_or_size_splits} (arg 1); the other
   * dimensions transfer unchanged.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible piece shapes, or {@code null} (⊤) when the {@code value} shape is
   *     unknown or the {@code axis} is out of the shape's range.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes = packedInputShapes(builder);
    if (inputShapes == null) return null;

    Integer count = constantIntArgOrNull(builder, 1, "num_or_size_splits");

    // An absent axis defaults to 0; a passed one must resolve to a constant, since an empty
    // points-to set on a passed argument means a computed value (the PA does not fold
    // arithmetic), not an absent one, and assuming 0 for it would be unsound.
    Integer axis;
    if (isAxisPassed(builder)) {
      axis = constantIntArgOrNull(builder, 2, "axis");
      if (axis == null) {
        LOGGER.fine(() -> "Non-constant axis for " + describe(this.getSource()) + "; returning ⊤.");
        return null;
      }
    } else axis = 0;

    Set<List<Dimension<?>>> sizeLists = count == null ? sizeListsOrNull(builder) : null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> input : inputShapes) {
      int rank = input.size();
      if (rank == 0) return null; // Splitting a scalar is a runtime error.
      int normalized = axis < 0 ? axis + rank : axis;
      if (normalized < 0 || normalized >= rank) return null;

      Dimension<?> axisDim = input.get(normalized);
      if (sizeLists != null) {
        // A size list: each piece's extent along the axis is one of the listed sizes, so the
        // modeled piece is any of them (wala/ML#993).
        for (List<Dimension<?>> sizes : sizeLists)
          for (Dimension<?> extent : pieceExtents(sizes, axisDim)) {
            List<Dimension<?>> out = new ArrayList<>(input);
            out.set(normalized, extent);
            ret.add(out);
          }
        continue;
      }

      List<Dimension<?>> out = new ArrayList<>(input);
      if (count != null
          && axisDim instanceof NumericDim
          && count > 0
          && ((NumericDim) axisDim).value() % count == 0)
        out.set(normalized, new NumericDim(((NumericDim) axisDim).value() / count));
      // A size-list split, a non-constant count, a non-numeric axis dimension, or a non-exact
      // division: the quotient is unknown, but the rank and the other dimensions still hold. A
      // quotient of a `None` axis is itself `None` at run time; otherwise it is a fixed size the
      // analysis could not compute (wala/ML#721).
      else
        out.set(
            normalized,
            axisDim instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE);
      ret.add(out);
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * The {@code value} operand's shapes, with a list or tuple operand packed into one tensor
   * (wala/ML#1016): its leading extent is the container's length and its element shape is one every
   * element can have, so an element with several possible shapes contributes only the shapes the
   * other elements share. The pass-through read unions the fields instead, which is the container's
   * reading, not the packed tensor's. A tensor operand keeps the pass-through read; an operand none
   * of whose elements has a known shape, or a tensor member whose shape is unknown, falls back to
   * the pass-through read and its resolutions (the callers' arguments, a declared build contract);
   * a container no combination of whose elements packs contributes nothing.
   *
   * @param builder The propagation call graph builder.
   * @return The operand's shapes as the operation sees them, or {@code null} when unknown.
   */
  private Set<List<Dimension<?>>> packedInputShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts =
        this.getArgumentPointsToSet(builder, getInputParameterPosition(), getInputParameterName());
    boolean container = false;
    if (pts != null)
      for (InstanceKey ik : pts) {
        AllocationSiteInNode asin = getAllocationSiteInNode(ik);
        if (asin == null) continue;
        TypeReference ref = asin.concreteType().getReference();
        if (ref.equals(list) || ref.equals(tuple)) container = true;
      }
    if (!container) return super.getDefaultShapes(builder);
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (InstanceKey ik : pts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      TypeReference ref = asin.concreteType().getReference();
      if (ref.equals(list) || ref.equals(tuple)) {
        Set<List<Dimension<?>>> packed = packedElementShapes(builder, asin);
        // No element's shape is known here: the pass-through read's own resolutions (the callers'
        // arguments, a declared build contract) decide, as they did before.
        if (packed == null) return super.getDefaultShapes(builder);
        int count =
            integerCatalogSize(
                builder
                    .getPointerAnalysis()
                    .getPointsToSet(
                        ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                            .getPointerKeyForObjectCatalog(asin)));
        for (List<Dimension<?>> shape : packed) {
          List<Dimension<?>> out = new ArrayList<>(shape.size() + 1);
          out.add(new NumericDim(count));
          out.addAll(shape);
          ret.add(out);
        }
      } else {
        ShapeResult member = this.getShapeResultFromTensor(builder, asin, false);
        if (member.hasUnknown()) return super.getDefaultShapes(builder);
        ret.addAll(member.members());
      }
    }
    return ret;
  }

  /**
   * The size lists a {@code num_or_size_splits} argument may be (wala/ML#993): the possible
   * contents of a list or tuple of sizes, read as a shape argument is, each size a {@link
   * NumericDim}, an inferred {@code -1} included.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @return The possible size lists, or {@code null} when the argument is not a resolvable list or
   *     tuple of sizes.
   */
  private Set<List<Dimension<?>>> sizeListsOrNull(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, 1, "num_or_size_splits");
    if (pts == null || pts.isEmpty()) return null;
    Set<List<Dimension<?>>> lists = this.getShapesFromShapeArgument(builder, pts);
    return lists == null || lists.isEmpty() ? null : lists;
  }

  /**
   * The extents along the split axis of the pieces a size list makes. A listed size is its own
   * extent; an inferred size ({@code -1}) is what the other sizes leave of the axis, when the axis
   * and the other sizes are known, and otherwise unknown as an unknown quotient is.
   *
   * @param sizes One possible size list.
   * @param axisDim The input's dimension along the split axis.
   * @return The possible piece extents.
   */
  private static Set<Dimension<?>> pieceExtents(List<Dimension<?>> sizes, Dimension<?> axisDim) {
    Dimension<?> unknown =
        axisDim instanceof DynamicDim ? DynamicDim.INSTANCE : UnresolvedDim.INSTANCE;
    Set<Dimension<?>> extents = HashSetFactory.make();
    int listed = 0;
    boolean allListedKnown = true;
    boolean inferred = false;
    for (Dimension<?> size : sizes)
      if (size instanceof NumericDim numeric && numeric.value() >= 0) {
        extents.add(numeric);
        listed += numeric.value();
      } else if (size instanceof SymbolicDim
          || (size instanceof NumericDim numeric && numeric.value() == -1)) inferred = true;
      else {
        extents.add(unknown);
        allListedKnown = false;
      }
    if (inferred)
      extents.add(
          axisDim instanceof NumericDim total && allListedKnown && total.value() >= listed
              ? new NumericDim(total.value() - listed)
              : unknown);
    return extents;
  }

  /**
   * Whether the {@code axis} argument is passed at the call site, positionally (a fourth use beyond
   * the callable, {@code value}, and {@code num_or_size_splits}) or as a resolvable keyword. A
   * keyword-passed <em>computed</em> axis is indistinguishable from an absent one here and is
   * treated as absent.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for call graph and PA lookup.
   * @return {@code true} iff some call site passes {@code axis}.
   */
  private boolean isAxisPassed(PropagationCallGraphBuilder builder) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) return call.getNumberOfPositionalParameters() > 3;
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode()))
      if (callerInvoke.snd instanceof PythonInvokeInstruction
          && ((PythonInvokeInstruction) callerInvoke.snd).getNumberOfPositionalParameters() > 3)
        return true;
    OrdinalSet<InstanceKey> keywordPts = this.getArgumentPointsToSet(builder, 2, "axis");
    return keywordPts != null && !keywordPts.isEmpty();
  }

  /**
   * Resolves an argument to a constant integer via its points-to set.
   *
   * @param builder The {@link PropagationCallGraphBuilder} providing the pointer analysis.
   * @param paramPos The argument's positional index, excluding {@code self}.
   * @param paramName The argument's keyword name.
   * @return The constant value, or {@code null} when the argument is absent, non-constant, or not
   *     an integer.
   */
  private Integer constantIntArgOrNull(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts == null || pts.isEmpty()) return null;
    Integer found = null;
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (!(value instanceof Number)) return null;
      int intValue = ((Number) value).intValue();
      if (found != null && found != intValue) return null; // Ambiguous.
      found = intValue;
    }
    return found;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the pass-through identity record path would bypass, so the record
   * view routes through the legacy transform until a member-wise upgrade.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * This generator transforms its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side input keys, or {@code null} when none is
   *     located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }
}
