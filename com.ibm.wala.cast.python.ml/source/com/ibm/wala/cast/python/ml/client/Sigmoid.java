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
 * Generator for {@code tf.nn.sigmoid}. Produces a fresh tensor with the same shape and dtype as the
 * {@code x} input.
 *
 * <p>Shape inference reads the {@code x} argument's shapes unchanged (sigmoid is element-wise).
 * Dtype inference likewise passes through unchanged. When the argument's PTS is empty, falls back
 * to the caller-walk mechanism, mirroring {@link MatMul}'s pattern for summary-method param slots
 * that are empty under synthetic-caller propagation.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid">tf.nn.sigmoid</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Sigmoid extends TensorGenerator {

  /** Positional parameters of {@code tf.nn.sigmoid.do()}: {@code self x name}. */
  private enum Parameters {
    /** The input tensor. */
    X;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs a {@code Sigmoid} from a caller-side {@link PointsToSetVariable} (the return of the
   * {@code tf.nn.sigmoid(...)} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the invoke.
   */
  public Sigmoid(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code Sigmoid} anchored to a manual node. Used when the factory is invoked
   * without a caller-side {@link PointsToSetVariable} (e.g., from {@link
   * TensorGenerator#createManualGenerator}).
   *
   * @param node The {@link CGNode} for the {@code sigmoid.do()} synthetic method.
   */
  public Sigmoid(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes as a shape-preserving passthrough of the {@code x} input.
   *
   * @param builder The propagation call graph builder.
   * @return The set of input shapes, or {@code null} if no shape can be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, Parameters.X.getIndex(), Parameters.X.getName());
  }

  /**
   * Resolves the output dtypes as a dtype-preserving passthrough of the {@code x} input.
   *
   * @param builder The propagation call graph builder.
   * @return The set of input dtypes, or {@code {UNKNOWN\}} if no dtype can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes = dtypesOfArg(builder, Parameters.X.getIndex(), Parameters.X.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * Resolves the shapes of the sigmoid's {@code x} argument. Tries the summary-local PTS first (via
   * {@link #getShapesOfValue}); on empty, falls back to {@link #getArgumentShapesViaCallers}, which
   * walks the caller's invoke site for concrete arg shapes. Mirrors {@link MatMul#shapesOfArg
   * MatMul}.
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
