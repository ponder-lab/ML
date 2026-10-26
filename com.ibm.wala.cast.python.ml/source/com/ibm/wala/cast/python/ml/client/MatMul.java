package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.util.TensorShapeUtil;
import com.ibm.wala.cast.python.ml.util.TensorShapeUtil.MatMulReading;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for {@code tf.matmul}.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/matmul">tf.matmul</a>
 */
public class MatMul extends TensorGenerator {

  /** Parameter positions and keyword names of the operand flags (wala/ML#1152). */
  protected enum Parameters {
    A,
    B,
    TRANSPOSE_A,
    TRANSPOSE_B,
    ADJOINT_A,
    ADJOINT_B;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public MatMul(PointsToSetVariable source) {
    super(source);
  }

  public MatMul(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * Member-wise record view (wala/ML#718): the product shape composes per operand-member pair, and
   * either operand's unknown remainder rides through.
   *
   * @param builder The propagation call graph builder.
   * @return The composed result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    ShapeResult aShapes = shapeResultOfArg(builder, Parameters.A);
    ShapeResult bShapes = shapeResultOfArg(builder, Parameters.B);
    if (aShapes.members().isEmpty() || bShapes.members().isEmpty()) return ShapeResult.unknown();
    Set<MatMulReading> readings = this.getReadings(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> aShape : aShapes.members()) {
      for (List<Dimension<?>> bShape : bShapes.members()) {
        // Batched semantics: the leading (batch) dimensions BROADCAST and the trailing two compose
        // as the matrix product, so the rank is preserved rather than collapsing to two
        // (wala/ML#718, wala/ML#878), after each operand a flag transposes has its last two axes
        // swapped (wala/ML#1152). The type-feed composition shares the unswapped rule, so for a
        // call setting no flag one operation cannot get two answers depending on which path
        // resolved it.
        //
        // A pair that cannot compose contributes NO member. Batch extents that are unequal and
        // neither of them 1 would fail at run time, so the pair is not an execution the program
        // can have: it is a spurious crossing of union members that never co-occur, and turning
        // it into an unknown member would lose precision to describe something impossible.
        ret.addAll(TensorShapeUtil.matmulShapes(aShape, bShape, readings));
      }
    }
    return ret.isEmpty()
        ? ShapeResult.unknown()
        : new ShapeResult(ret, aShapes.hasUnknown() || bShapes.hasUnknown());
  }

  /**
   * Routes the output-shape resolution through {@link #getDefaultShapeResult} (this generator has
   * no {@code shape} parameter), so partial results cross the generator boundary (wala/ML#718).
   *
   * @param builder The propagation call graph builder.
   * @return The resolution result.
   */
  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder);
  }

  /**
   * Resolves the result dtype from either operand: the runtime requires {@code a} and {@code b} to
   * agree on dtype, so a resolved second operand proves the result dtype when the first operand's
   * walk does not (wala/ML#677; the equality constraint mirrors the einsum operand refinement,
   * wala/ML#704). The gpt-2 witness is {@code tf.matmul(h_flat, self.layer_weights)}: {@code
   * h_flat} flattens the decoder-stack output, whose producer walk does not resolve, while the
   * weight types {@code float32}.
   *
   * <p>When both operands resolve, the result is their common dtypes: a pairing whose dtypes differ
   * raises at run time ({@code tf.matmul} of an {@code int32} and a {@code float32} operand is an
   * {@code InvalidArgumentError}), so it yields no tensor. A layer whose undecided {@code mode}
   * branch also hands its integer ids to a projection against float weights has such a pairing.
   * Operands that share no dtype raise in every pairing, so the dtypes are empty and {@link
   * #getTensorTypes} yields no tensor type for the call. The shape query does not read the dtypes,
   * so the engine's shape dependencies stay as they were.
   *
   * <p>An operand whose dtypes are partial (wala/ML#862), the dtypes it has at the allocations that
   * resolved plus {@link DType#UNKNOWN} for the rest, is not resolved: a resolved other operand
   * still proves the result, and otherwise the partial operand's dtypes stand, so the members that
   * resolved survive into the result rather than reading as unknown.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The common dtypes when both operands resolve (empty when they share none), else the
   *     dtypes of the operand that resolves, else those of the first operand when it is partial or
   *     of the second when it reads anything, else {@code UNKNOWN}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> aDTypes = dtypesOfArg(builder, 0, "a");
    if (isResolvedDTypes(aDTypes)) {
      Set<DType> bDTypes = dtypesOfArg(builder, 1, "b");
      if (isResolvedDTypes(bDTypes)) return commonDTypes(aDTypes, bDTypes);
      return aDTypes;
    }
    Set<DType> bDTypes = dtypesOfArg(builder, 1, "b");
    if (isResolvedDTypes(bDTypes)) return bDTypes;
    if (hasKnownDType(aDTypes)) return aDTypes;
    if (bDTypes != null && !bDTypes.isEmpty()) return bDTypes;
    return aDTypes == null || aDTypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : aDTypes;
  }

  /**
   * The dtypes two operands share.
   *
   * @param aDTypes The first operand's dtypes.
   * @param bDTypes The second operand's dtypes.
   * @return Their intersection.
   */
  private static Set<DType> commonDTypes(Set<DType> aDTypes, Set<DType> bDTypes) {
    Set<DType> common = EnumSet.noneOf(DType.class);
    common.addAll(aDTypes);
    common.retainAll(bDTypes);
    return common;
  }

  /**
   * The possible readings of the operand flags (wala/ML#1152): {@code a} enters the product
   * transposed when {@code transpose_a} or {@code adjoint_a} is true, and likewise {@code b}. A
   * reading setting both flags of one operand raises at run time, so it is not among them.
   *
   * @param builder The propagation call graph builder.
   * @return The readings, never empty.
   */
  private Set<MatMulReading> getReadings(PropagationCallGraphBuilder builder) {
    Set<Boolean> transposeA = this.getPossibleFlagValues(builder, Parameters.TRANSPOSE_A);
    Set<Boolean> transposeB = this.getPossibleFlagValues(builder, Parameters.TRANSPOSE_B);
    Set<Boolean> adjointA = this.getPossibleFlagValues(builder, Parameters.ADJOINT_A);
    Set<Boolean> adjointB = this.getPossibleFlagValues(builder, Parameters.ADJOINT_B);
    Set<Boolean> swapsA = swaps(transposeA, adjointA);
    Set<Boolean> swapsB = swaps(transposeB, adjointB);
    Set<MatMulReading> ret = HashSetFactory.make();
    for (Boolean a : swapsA) for (Boolean b : swapsB) ret.add(new MatMulReading(a, b));
    return ret;
  }

  /**
   * Whether an operand enters the product swapped, over the possible values of its two flags. Both
   * set raises at run time, so that pairing contributes nothing.
   *
   * @param transpose The possible values of the operand's {@code transpose_x}.
   * @param adjoint The possible values of the operand's {@code adjoint_x}.
   * @return The possible swaps, never empty.
   */
  private static Set<Boolean> swaps(Set<Boolean> transpose, Set<Boolean> adjoint) {
    Set<Boolean> ret = HashSetFactory.make();
    for (Boolean t : transpose) for (Boolean j : adjoint) if (!(t && j)) ret.add(t || j);
    // Every pairing raising leaves no execution to describe; read the operand both ways rather
    // than claim one.
    if (ret.isEmpty()) ret.addAll(Set.of(false, true));
    return ret;
  }

  /**
   * The possible values of a flag: its literal values, {@code false} when no call site supplies it
   * (the API default), and both when a site supplies a value that doesn't resolve to a literal
   * {@code bool}, such as a computed {@code bool}, whose points-to set reads as empty here just as
   * an omitted argument's does (wala/ML#865).
   *
   * @param builder The propagation call graph builder.
   * @param flag The flag.
   * @return The possible values, never empty.
   */
  private Set<Boolean> getPossibleFlagValues(PropagationCallGraphBuilder builder, Parameters flag) {
    Set<Boolean> values =
        getPossibleBooleanValues(
            this.getArgumentPointsToSet(builder, flag.getIndex(), flag.getName()));
    if (values != null && !values.isEmpty()) return values;
    if (values != null
        && Boolean.FALSE.equals(
            this.isArgumentSyntacticallySupplied(builder, flag.getIndex(), flag.getName())))
      return Set.of(false);
    return Set.of(false, true);
  }

  /**
   * Resolves shapes of the matmul arg at the given position. Tries the summary-local PTS first (via
   * {@link #getShapesOfValue}); if empty, falls back to {@link #getArgumentShapesViaCallers} which
   * walks the caller's call site for concrete arg shapes. The fallback is needed because the matmul
   * generator's node is the synthetic {@code matmul.do()} summary and the summary's param-slot PTS
   * is often empty even when the caller passes a concretely-shaped tensor.
   *
   * @param builder The propagation call graph builder.
   * @param operand The operand, {@link Parameters#A} or {@link Parameters#B}.
   * @return The resolved shapes, or {@code null} if neither path recovers a concrete shape.
   */
  private ShapeResult shapeResultOfArg(PropagationCallGraphBuilder builder, Parameters operand) {
    int paramPos = operand.getIndex();
    String paramName = operand.getName();
    // The per-context caller walk resolves the argument at its call site, where a φ-defined
    // argument's arm feasibility prunes (wala/ML#746); the argument's points-to union cannot
    // distinguish arms (it is the union over all of them), so it serves as the floor rather than
    // the short-circuit (wala/ML#716, wala/ML#718).
    ShapeResult viaCallers = this.getArgumentShapeResultViaCallers(builder, paramPos, paramName);
    if (!viaCallers.members().isEmpty()) return viaCallers;
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      ShapeResult fromValue = this.getShapeResultOfValue(builder, pts, true);
      if (!fromValue.members().isEmpty()) return fromValue;
    }
    return viaCallers;
  }

  /**
   * Dtype counterpart to {@link #shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved dtypes, or {@code null} if neither path recovers.
   */
  private Set<DType> dtypesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    Set<DType> dtypes = pts == null || pts.isEmpty() ? null : this.getDTypesOfValue(builder, pts);
    // A result that is ⊤ or partial does not short-circuit: the synthetic `matmul.do` param PTS is
    // often context-collapsed to a union of unrelated bare tensors (all `UNKNOWN`), so fall
    // through to the per-context caller-walk, which resolves the actual argument at each call site
    // (wala/ML#570). A partial points-to read stands only when the walk does not resolve either
    // (wala/ML#862).
    return preferResolvedDTypes(
        dtypes, () -> this.getArgumentDTypesViaCallers(builder, paramPos, paramName));
  }

  /**
   * Declares a {@link TypeFeedKind#MATMUL} feed over both operands (wala/ML#877). When an operand's
   * shape lives only in {@code TensorTypeAnalysis} dataflow state that the PTS-based operand walk
   * cannot see (for example an operand bound by tuple-unpacking), {@link #getDefaultShapeResult}
   * seeds an unresolved ⊤ member; the feed replaces it with a synthetic dataflow edge from each
   * operand, composing {@code (..., m, k)} and {@code (..., k, n)} into {@code (..., m, n)} exactly
   * as {@link #getDefaultShapeResult} does for a call that sets no operand flag. The feed composes
   * its operands unswapped: it does not read {@code transpose_a}, {@code transpose_b}, {@code
   * adjoint_a} or {@code adjoint_b}, which {@link #getDefaultShapeResult} reads (wala/ML#1152), and
   * no test in this repository depends on the feed's composition. Its sparse cousin {@code
   * SparseMatrixDot} already declares a feed; this closes the twin-operation gap.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The matmul feed over operands {@code a} and {@code b}, or {@code null} when either
   *     operand cannot be located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    CGNode node = this.getNode();
    if (node == null || node.getIR() == null) return null;
    int aVn = this.getArgumentValueNumber(builder, 0, "a", true);
    int bVn = this.getArgumentValueNumber(builder, 1, "b", true);
    if (aVn <= 0 || bVn <= 0) return null;
    return new TypeFeed(
        TypeFeedKind.MATMUL,
        List.of(
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, aVn),
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, bVn)));
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
}
