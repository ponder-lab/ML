package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Common base for tensor generators that produce a fresh tensor with the same shape as a single
 * input argument, and (for the ops currently using this base) the same dtype. Currently extended by
 * {@link Exp}, {@link Rsqrt}, and {@link LogSoftmax}. {@link Identity} and {@link StopGradient}
 * (added in PR #202) follow the same pattern by hand and should be refactored onto this base once
 * #202 lands. The base is intended for shape-preserving unary ops more broadly — including
 * additional real-input element-wise unary math ({@code tf.math.tanh}) and shape-preserving
 * normalizers ({@code tf.math.l2_normalize}) — but those generators don't yet extend this base. See
 * wala/ML#449 (Tier 1 + Tier 2).
 *
 * <p>Note on dtype: TensorFlow has unary ops that are shape-preserving but <em>not</em>
 * dtype-preserving for some input dtypes — e.g., {@code tf.math.abs} returns a real dtype when
 * given a complex input. Such ops should not extend this base directly; they need their own dtype
 * logic or a more specialized base. The current subclasses are all dtype-preserving for the inputs
 * they accept.
 *
 * <p>Subclasses identify the input argument via {@link #getInputParameterPosition()} and {@link
 * #getInputParameterName()}; everything else is shared. The arg-resolution itself uses the same
 * PTS-first-with-caller-walk-fallback pattern as {@link Sigmoid} (which predates this base and
 * still has its own copy — left untouched here to keep this change reviewable; refactoring {@code
 * Sigmoid}/{@code Identity}/{@code StopGradient} onto this base is a clean follow-up once {@code
 * #202} lands).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class PassThroughUnaryTensorGenerator extends TensorGenerator {

  /**
   * Constructs from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  protected PassThroughUnaryTensorGenerator(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  protected PassThroughUnaryTensorGenerator(CGNode node) {
    super(node);
  }

  /**
   * Positional index of the input argument, <em>excluding</em> the modeled {@code self} receiver.
   * Almost always {@code 0} (the first user-facing positional after {@code self}).
   *
   * <p>Default is {@link #UNDEFINED_PARAMETER_POSITION}, signalling "no input arg" — appropriate
   * <em>only</em> for subclasses that override both {@link #getDefaultShapes} and {@link
   * #getDefaultDTypes} entirely and never reach {@link #shapesOfArg} / {@link #dtypesOfArg}. If the
   * parent's default methods <em>are</em> reached with an undefined position, that's a
   * misconfiguration: throw {@link IllegalStateException} rather than silently producing ⊤.
   *
   * @return The input arg's positional index, or {@link #UNDEFINED_PARAMETER_POSITION} if the
   *     subclass doesn't read from an input argument.
   */
  protected int getInputParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * Keyword name of the input argument as declared in the XML (e.g. {@code "x"}, {@code "input"},
   * {@code "logits"}). Used for keyword-argument resolution when the call site uses kwargs.
   *
   * <p>Default is {@code null}, signalling "no input arg" — see {@link
   * #getInputParameterPosition()} for when this default is appropriate.
   *
   * @return The input arg's keyword name, or {@code null} if the subclass doesn't read from an
   *     input argument.
   */
  protected String getInputParameterName() {
    return null;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, getInputParameterPosition(), getInputParameterName());
  }

  /**
   * Identity record path for the pass-through family (wala/ML#718): the input argument's {@link
   * ShapeResult} passes through unchanged, so a partially resolvable input keeps its members with
   * the remainder marked. A subclass that transforms shapes (an override of {@link
   * #getDefaultShapes}) MUST also override this method — either with the member-wise transform over
   * {@code members()} or with the collapse-safe {@code
   * ShapeResult.fromLegacy(this.getDefaultShapes(builder))} — since this identity default would
   * otherwise bypass the transform for record consumers.
   *
   * @param builder The propagation call graph builder.
   * @return The input argument's resolution result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return shapeResultOfArg(builder, getInputParameterPosition(), getInputParameterName());
  }

  /**
   * Routes the generator's output-shape resolution through {@link #getDefaultShapeResult} (this
   * family has no {@code shape} parameter), so partial results cross the generator boundary
   * (wala/ML#718).
   *
   * @param builder The propagation call graph builder.
   * @return The resolution result.
   */
  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes = dtypesOfArg(builder, getInputParameterPosition(), getInputParameterName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * PTS-first arg-shape resolver with caller-walk fallback. Mirrors {@link Sigmoid#shapesOfArg}.
   *
   * <p>Visible to subclasses that compute their output shape from more than one input argument
   * (e.g. {@link UnsortedSegmentReduction}, which reads both its {@code data} and {@code
   * segment_ids} arguments) and so cannot rely on the single-input {@link #getDefaultShapes}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved shapes, or {@code null} if neither path recovers.
   */
  protected Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    return this.shapeResultOfArg(builder, paramPos, paramName).toLegacy();
  }

  /**
   * Record-carrying core of {@link #shapesOfArg} (wala/ML#718): a partially resolvable input keeps
   * its members with the remainder marked, first over the argument's points-to union and, when that
   * has no members, over the per-context caller walk.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolution result.
   */
  protected ShapeResult shapeResultOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    if (paramPos == UNDEFINED_PARAMETER_POSITION)
      throw new IllegalStateException(
          getClass().getSimpleName()
              + " uses "
              + PassThroughUnaryTensorGenerator.class.getSimpleName()
              + "'s default getDefaultShapes (which reads from the input arg) but did not"
              + " override getInputParameterPosition. Either override the input-arg getters or"
              + " override getDefaultShapes entirely.");
    ShapeResult fromValue = ShapeResult.unknown();
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      // Exact mode (wala/ML#716): the generator asserts an output shape computed from "the"
      // input shape, so a partial union here would overclaim.
      fromValue = this.getShapeResultOfValue(builder, pts, true);
      if (!fromValue.members().isEmpty() && !fromValue.hasUnknown()) return fromValue;
    }
    // An incomplete points-to union commonly reflects context collapse, so the per-context
    // caller walk is preferred; the union's resolvable members are the floor when the walk
    // fails (wala/ML#716, wala/ML#718).
    ShapeResult viaCallers = this.getArgumentShapeResultViaCallers(builder, paramPos, paramName);
    if (!viaCallers.members().isEmpty()) return viaCallers;
    return fromValue.members().isEmpty() ? viaCallers : fromValue;
  }

  /**
   * Dtype counterpart of {@link #shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved dtypes, or {@code null} if neither path recovers.
   */
  private Set<DType> dtypesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    if (paramPos == UNDEFINED_PARAMETER_POSITION)
      throw new IllegalStateException(
          getClass().getSimpleName()
              + " uses "
              + PassThroughUnaryTensorGenerator.class.getSimpleName()
              + "'s default getDefaultDTypes (which reads from the input arg) but did not"
              + " override getInputParameterPosition. Either override the input-arg getters or"
              + " override getDefaultDTypes entirely.");
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, pts);
      if (dtypes != null && !dtypes.isEmpty()) return dtypes;
    }
    return this.getArgumentDTypesViaCallers(builder, paramPos, paramName);
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
