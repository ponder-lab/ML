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
import java.util.Locale;
import java.util.Set;

/**
 * Generator for {@code tf.identity}. Returns a fresh tensor with the same shape and dtype as the
 * {@code input} argument.
 *
 * <p>Pre-fix the {@code identity} XML routed through {@code convert_to_tensor} as a chained
 * synthetic — analyzable but with extra indirection. This dedicated generator reads {@code input}'s
 * shape and dtype directly via the same arg-resolution pattern as {@link Sigmoid} (PTS-first with
 * caller-walk fallback for empty summary-method param slots). See wala/ML#449 (Tier 1).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/identity">tf.identity</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Identity extends TensorGenerator {

  /**
   * Positional parameters of {@code tf.identity.do()}, excluding the modeled {@code self} (the
   * function-object receiver). Modeled as {@code self input name} in the XML — same shape as {@code
   * Sigmoid} — but the enum indices map to argument positions <em>after</em> {@code self}, which
   * {@link TensorGenerator#getArgumentValueNumber(int)} skips automatically for non-static methods.
   */
  private enum Parameters {
    /** The input tensor. */
    INPUT;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs an {@code Identity} from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Identity(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs an {@code Identity} anchored to a manual node. Used by {@link
   * TensorGenerator#createManualGenerator}.
   *
   * @param node The {@link CGNode} for the {@code identity.do()} synthetic method.
   */
  public Identity(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes as a shape-preserving passthrough of the {@code input} argument.
   *
   * @param builder The propagation call graph builder.
   * @return The set of input shapes, or {@code null} if no shape can be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
  }

  /**
   * Resolves the output dtypes as a dtype-preserving passthrough of the {@code input} argument.
   *
   * @param builder The propagation call graph builder.
   * @return The set of input dtypes, or {@code {UNKNOWN\}} if no dtype can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes =
        dtypesOfArg(builder, Parameters.INPUT.getIndex(), Parameters.INPUT.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * PTS-first arg-shape resolver with caller-walk fallback. Mirrors {@link Sigmoid#shapesOfArg}.
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolved shapes, or {@code null} if neither path recovers.
   */
  private Set<List<Dimension<?>>> shapesOfArg(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, pts);
      if (shapes != null && !shapes.isEmpty()) return shapes;
    }
    return this.getArgumentShapesViaCallers(builder, paramPos, paramName);
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
