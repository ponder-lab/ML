package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.Collections.emptyList;
import static java.util.Collections.singleton;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of an element-wise operation in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/multiply">tf.multiply</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ElementWiseOperation extends ZerosLike {

  private static final Logger LOGGER = getLogger(ElementWiseOperation.class.getName());

  protected enum Parameters {
    X,
    Y,
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ElementWiseOperation(PointsToSetVariable source) {
    super(source);
  }

  protected int getXParameterPosition() {
    return Parameters.X.getIndex();
  }

  protected String getXParameterName() {
    return Parameters.X.getName();
  }

  protected int getXArgumentValueNumber(PropagationCallGraphBuilder builder) {
    if (this.source.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.source.getPointerKey();
      SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
      if (def instanceof SSABinaryOpInstruction) {
        return def.getUse(0);
      }
    }
    return this.getArgumentValueNumber(
        builder, this.getXParameterPosition(), this.getXParameterName(), false);
  }

  protected int getYParameterPosition() {
    return Parameters.Y.getIndex();
  }

  protected String getYParameterName() {
    return Parameters.Y.getName();
  }

  protected int getYArgumentValueNumber(PropagationCallGraphBuilder builder) {
    if (this.source.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.source.getPointerKey();
      SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
      if (def instanceof SSABinaryOpInstruction) {
        return def.getUse(1);
      }
    }
    return this.getArgumentValueNumber(
        builder, this.getYParameterPosition(), this.getYParameterName(), false);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The resulting shape is the broadcasted shape of the shapes of x and y.
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    LOGGER.fine(
        () ->
            "EWO.getDefaultShapes entered with source=" + this.source + ", node=" + this.getNode());

    int xVn = this.getXArgumentValueNumber(builder);
    LOGGER.fine(() -> "EWO.getDefaultShapes xVn: " + xVn);
    if (xVn <= 0) return null;
    Set<List<Dimension<?>>> xShapes = this.getOperandShapes(builder, xVn);
    LOGGER.fine(() -> "EWO.getDefaultShapes xShapes: " + xShapes);
    if (xShapes == null) return null;

    int yVn = this.getYArgumentValueNumber(builder);
    LOGGER.fine(() -> "EWO.getDefaultShapes yVn: " + yVn);
    if (yVn <= 0) return null;
    Set<List<Dimension<?>>> yShapes = this.getOperandShapes(builder, yVn);
    LOGGER.fine(() -> "EWO.getDefaultShapes yShapes: " + yShapes);
    if (yShapes == null) return null;

    for (List<Dimension<?>> xShape : xShapes)
      for (List<Dimension<?>> yShape : yShapes)
        if (areBroadcastable(xShape, yShape)) ret.add(getBroadcastedShapes(xShape, yShape));
        else throw new NonBroadcastableShapesException(this, xShape, yShape);

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    int xVn = this.getXArgumentValueNumber(builder);
    int yVn = this.getYArgumentValueNumber(builder);
    LOGGER.fine("ElementWiseOperation getDefaultDTypes xVn: " + xVn + ", yVn: " + yVn);
    if (xVn <= 0) return EnumSet.of(DType.UNKNOWN);

    // Type promotion for scalar-literal operands: NumPy/TF rules promote `int_tensor op
    // float_literal` to float32 (e.g., `x_train.astype(uint8) / 255.0` → float32). Check for
    // literals BEFORE calling getOperandDTypes to avoid short-circuiting via exception on
    // an implicit-PK operand (see wala/WALA#1889).
    CGNode node = this.getNode();
    if (yVn > 0 && isFloatLiteralVn(node, yVn)) {
      LOGGER.fine(
          "ElementWiseOperation getDefaultDTypes: promoting to FLOAT32 (y is float literal)");
      return EnumSet.of(DType.FLOAT32);
    }
    if (isFloatLiteralVn(node, xVn)) {
      LOGGER.fine(
          "ElementWiseOperation getDefaultDTypes: promoting to FLOAT32 (x is float literal)");
      return EnumSet.of(DType.FLOAT32);
    }

    Set<DType> xDTypes = this.getOperandDTypes(builder, xVn);
    LOGGER.fine("ElementWiseOperation getDefaultDTypes dtypes: " + xDTypes);
    // An element-wise op always produces a tensor. If operand resolution returned ⊥ (empty set)
    // or null, that represents "unable to resolve the operand's dtype," not "not a tensor." Emit
    // ⊤ (UNKNOWN) so `TensorTypeAnalysis` still tracks the result — otherwise chained ops like
    // `layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, ...), ...))` silently lose tensor
    // identification across the chain even though every step is demonstrably a tensor producer.
    if (xDTypes == null || xDTypes.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    return xDTypes;
  }

  /**
   * Resolves the shapes of an operand value number, with a targeted bypass for nested binary ops.
   *
   * <p>The base {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)} path dispatches back
   * through {@link TensorGeneratorFactory#getGenerator} and then refuses to recurse when the
   * resulting generator is of the same class as {@code this} (to avoid infinite recursion with
   * other generator kinds). For nested binop chains like {@code (x - k1) / k2}, that class-skip
   * short-circuits the recursion and we lose the shape.
   *
   * <p>If the operand's def is itself a {@link SSABinaryOpInstruction}, this method constructs a
   * child {@link ElementWiseOperation} for the operand and invokes its {@link
   * #getDefaultShapes(PropagationCallGraphBuilder)} directly, bypassing the class-skip.
   *
   * <p>If the operand is a Python scalar literal (e.g., {@code 127.5} in {@code x - 127.5}), it has
   * an empty points-to set and no defining instruction &mdash; {@link #getShapes} would fall
   * through to a factory lookup that throws. Return {@code {emptyList}} (scalar rank-0) so the
   * tensor-scalar broadcast proceeds correctly. See wala/ML#395.
   *
   * <p>Otherwise falls through to the standard {@link #getShapes(PropagationCallGraphBuilder, int)}
   * path so non-binop, non-scalar-literal operands (invokes, slices, etc.) follow the normal
   * generator dispatch.
   */
  private Set<List<Dimension<?>>> getOperandShapes(PropagationCallGraphBuilder builder, int vn) {
    ElementWiseOperation nested = getNestedForBinop(builder, vn);
    if (nested != null) return nested.getDefaultShapes(builder);
    if (isScalarLiteral(builder, vn)) {
      return singleton(emptyList());
    }
    // PTS-first with SSA-DU fallback — handles operands whose def is a synthetic-method
    // return (implicit PK) by walking the DU chain. See wala/WALA#1889.
    return this.getShapesOrSSAChain(builder, this.getNode(), vn);
  }

  /** Dtype counterpart of {@link #getOperandShapes(PropagationCallGraphBuilder, int)}. */
  private Set<DType> getOperandDTypes(PropagationCallGraphBuilder builder, int vn) {
    ElementWiseOperation nested = getNestedForBinop(builder, vn);
    if (nested != null) return nested.getDefaultDTypes(builder);
    return this.getDTypes(builder, vn);
  }

  /**
   * Identifies Python scalar literals at binop operand slots &mdash; values that have empty
   * points-to sets, no defining instruction, and appear in the IR's symbol table as constants
   * (e.g., {@code 127.5}, {@code 1}, {@code 2}). Used to short-circuit {@link #getOperandShapes}
   * and {@link #getOperandDTypes} so scalar-tensor broadcasts preserve shape/dtype info. See
   * wala/ML#395.
   */
  private boolean isScalarLiteral(PropagationCallGraphBuilder builder, int vn) {
    CGNode node = this.getNode();
    if (node.getDU().getDef(vn) != null) return false;
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    if (!builder.getPointerAnalysis().getPointsToSet(pk).isEmpty()) return false;
    return node.getIR().getSymbolTable().isConstant(vn);
  }

  /**
   * If the given value number is defined by a binary op in {@code this}'s CGNode, returns a new
   * {@code ElementWiseOperation} whose source is the points-to set variable for that value number;
   * otherwise {@code null}.
   */
  private ElementWiseOperation getNestedForBinop(PropagationCallGraphBuilder builder, int vn) {
    CGNode node = this.getNode();
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof SSABinaryOpInstruction)) return null;
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    if (builder.getPropagationSystem().isImplicit(pk)) return null;
    PointsToSetVariable nestedSource = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
    return new ElementWiseOperation(nestedSource);
  }

  /** No explicit dtype argument. Dtype is inferred from 'x'. */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
