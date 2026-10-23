package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.ir.ssa.CAstUnaryOp;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for Python's unary operators on a tensor, {@code -t} and {@code ~t}, which TensorFlow
 * evaluates as {@code tf.negative(t)} and {@code tf.bitwise.invert(t)} (or {@code
 * tf.logical_not(t)} on a {@code bool} tensor): the result has the operand's shape and dtype.
 *
 * <p>The operator has no call site, so the operand is read off the {@link SSAUnaryOpInstruction} in
 * the source's own frame rather than through an invocation's arguments. Only source anchoring
 * exists: there is no synthetic summary to anchor a manual generator on. {@link
 * TensorGeneratorFactory} dispatches here only when the operand shows tensor evidence, so a negated
 * Python number is no tensor. Python's {@code not} is a truth test giving a Python {@code bool} and
 * is not dispatched here.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/math/negative">tf.math.negative</a>
 */
public class UnaryOperation extends TensorGenerator {

  /**
   * Constructs from the unary operator's result.
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the operator.
   */
  public UnaryOperation(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Whether the given instruction is a unary operator this generator types: {@code -} or {@code ~}.
   *
   * @param def The instruction, possibly {@code null}.
   * @return {@code true} iff {@code def} is a unary minus or a bitwise invert.
   */
  static boolean isTensorUnaryOperator(SSAInstruction def) {
    if (!(def instanceof SSAUnaryOpInstruction unary)) return false;
    return CAstUnaryOp.MINUS.equals(unary.getOpcode())
        || CAstUnaryOp.BITNOT.equals(unary.getOpcode());
  }

  /**
   * The operand's value number in {@link #getNode()}.
   *
   * @return The operand's value number, or -1 when the source is not a unary operator's result.
   */
  private int getOperandValueNumber() {
    if (this.getSource() == null
        || !(this.getSource().getPointerKey() instanceof LocalPointerKey local)) return -1;
    SSAInstruction def = local.getNode().getDU().getDef(local.getValueNumber());
    return isTensorUnaryOperator(def) ? def.getUse(0) : -1;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * The operand's shapes, read exactly, since the result asserts the operand's shape. An operand
   * that reads as no tensor on this axis is ⊤ here rather than ⊥, so the shape axis cannot disagree
   * with the dtype axis, which is never ⊥.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The operand's shape result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    int operand = this.getOperandValueNumber();
    if (operand <= 0) return ShapeResult.unknown();
    ShapeResult result = this.getShapeResult(builder, this.getNode(), operand, true);
    return result.isBottom() ? ShapeResult.unknown() : result;
  }

  @Override
  protected ShapeResult getShapeResult(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int operand = this.getOperandValueNumber();
    if (operand <= 0) return EnumSet.of(DType.UNKNOWN);
    Set<DType> dtypes = this.getDTypes(builder, this.getNode(), operand);
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * The operand, in the operator's own frame, feeds the result unchanged.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The pass-through feed over the operand, or {@code null} when it cannot be located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    int operand = this.getOperandValueNumber();
    if (operand <= 0) return null;
    return new TypeFeed(
        TypeFeedKind.PASS_THROUGH,
        List.of(
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(this.getNode(), operand)));
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
