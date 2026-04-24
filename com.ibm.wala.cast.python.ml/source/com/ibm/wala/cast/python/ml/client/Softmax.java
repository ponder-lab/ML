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
 * Generator for {@code tf.nn.softmax}. Produces a fresh tensor with the same shape and dtype as the
 * {@code logits} input (softmax is element-wise along one axis and shape-preserving).
 *
 * <p>Structurally identical to {@link Sigmoid}; differs only in the parameter name ({@code logits}
 * vs {@code x}).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/nn/softmax">tf.nn.softmax</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Softmax extends TensorGenerator {

  /** Positional parameters of {@code tf.nn.softmax.do()}: {@code self logits axis name}. */
  private enum Parameters {
    /** The input tensor. */
    LOGITS;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  /**
   * Constructs a {@code Softmax} from a caller-side {@link PointsToSetVariable}.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     tf.nn.softmax(...)} invoke.
   */
  public Softmax(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a {@code Softmax} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code softmax.do()} synthetic method.
   */
  public Softmax(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes as a shape-preserving passthrough of the {@code logits} input.
   *
   * @param builder The propagation call graph builder.
   * @return The set of input shapes, or {@code null} if no shape can be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return shapesOfArg(builder, Parameters.LOGITS.getIndex(), Parameters.LOGITS.getName());
  }

  /**
   * Resolves the output dtypes as a dtype-preserving passthrough of the {@code logits} input.
   *
   * @param builder The propagation call graph builder.
   * @return The set of input dtypes, or {@code {UNKNOWN\}} if no dtype can be resolved.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> dtypes =
        dtypesOfArg(builder, Parameters.LOGITS.getIndex(), Parameters.LOGITS.getName());
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * PTS-primary, caller-walk-fallback resolver for the arg at {@code paramPos}. Mirrors {@link
   * Sigmoid#shapesOfArg Sigmoid} / {@link MatMul#shapesOfArg MatMul}.
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
