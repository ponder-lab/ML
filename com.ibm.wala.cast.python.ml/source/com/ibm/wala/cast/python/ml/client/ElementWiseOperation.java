package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.Collections.emptyList;
import static java.util.Collections.singleton;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAUnaryOpInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of an element-wise operation in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/multiply">tf.multiply</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ElementWiseOperation extends TensorGenerator {

  private static final Logger LOGGER = getLogger(ElementWiseOperation.class.getName());

  protected enum Parameters {
    X,
    Y,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ElementWiseOperation(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs an {@code ElementWiseOperation} anchored to a manual node (e.g. the {@code
   * tf.multiply.do()} synthetic method), for producer delegation when the result's points-to chain
   * is implicit (wala/ML#718).
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public ElementWiseOperation(CGNode node) {
    super(node);
  }

  protected int getXParameterPosition() {
    return Parameters.X.getIndex();
  }

  protected String getXParameterName() {
    return Parameters.X.getName();
  }

  protected int getXArgumentValueNumber(PropagationCallGraphBuilder builder) {
    if (this.source != null && this.source.getPointerKey() instanceof LocalPointerKey) {
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
    if (this.source != null && this.source.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.source.getPointerKey();
      SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
      if (def instanceof SSABinaryOpInstruction) {
        return def.getUse(1);
      }
    }
    return this.getArgumentValueNumber(
        builder, this.getYParameterPosition(), this.getYParameterName(), false);
  }

  /**
   * Record view of the broadcast (wala/ML#718): when one operand resolves to shape members and the
   * other is a statically opaque <em>scalar expression</em> (e.g. {@code 1.0 /
   * math.sqrt(float(...))}, whose value never resolves), the scalar broadcast preserves the
   * resolved side's shapes exactly, so they stand; the legacy path returned ⊥ and erased the tensor
   * entirely. An opaque co-operand that is not provably scalar keeps the legacy result, since
   * broadcast can change the shape when the opaque side's rank dominates.
   *
   * @param builder The propagation call graph builder.
   * @return The broadcast result.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    int xVn = this.getXArgumentValueNumber(builder);
    if (xVn <= 0) return ShapeResult.unknown();
    Set<List<Dimension<?>>> xShapes = this.getOperandShapes(builder, xVn);
    int yVn = this.getYArgumentValueNumber(builder);
    if (yVn <= 0) return ShapeResult.unknown();
    Set<List<Dimension<?>>> yShapes = this.getOperandShapes(builder, yVn);

    boolean xHas = xShapes != null && !xShapes.isEmpty();
    boolean yHas = yShapes != null && !yShapes.isEmpty();
    if (xHas != yHas) {
      Set<List<Dimension<?>>> resolved = xHas ? xShapes : yShapes;
      int opaqueVn = xHas ? yVn : xVn;
      if (resolved.stream().anyMatch(dims -> !dims.isEmpty())
          && isScalarExpression(builder, opaqueVn)) return ShapeResult.of(resolved);
    }
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * Recognizes a statically opaque scalar expression: a scalar literal, a {@code float(...)} or
   * {@code int(...)} builtin call, a {@code math.<fn>(...)} call, a unary minus, or a binary op
   * whose operands are both scalar expressions (wala/ML#718). The value never resolves (the PA does
   * not fold arithmetic), but its scalarness is structural, so a broadcast against it preserves the
   * tensor operand's shape.
   *
   * @param builder The propagation call graph builder.
   * @param vn The operand's value number.
   * @return {@code true} iff the operand is structurally a Python scalar expression.
   */
  private boolean isScalarExpression(PropagationCallGraphBuilder builder, int vn) {
    if (vn <= 0) return false;
    CGNode node = this.getNode();
    if (node.getDU() == null || node.getIR() == null) return false;
    if (isScalarLiteral(builder, vn)) return true;
    if (node.getIR().getSymbolTable().isConstant(vn)) return true;
    SSAInstruction def = node.getDU().getDef(vn);
    if (def instanceof SSABinaryOpInstruction)
      return isScalarExpression(builder, def.getUse(0))
          && isScalarExpression(builder, def.getUse(1));
    if (def instanceof SSAUnaryOpInstruction) return isScalarExpression(builder, def.getUse(0));
    if (def instanceof PythonInvokeInstruction) {
      SSAInstruction funcDef = node.getDU().getDef(def.getUse(0));
      if (funcDef instanceof PythonPropertyRead) {
        int objVn = ((PythonPropertyRead) funcDef).getObjectRef();
        SSAInstruction objDef = node.getDU().getDef(objVn);
        // `math.<fn>(...)`: the receiver is the `math` module read.
        if (objDef == null && node.getIR().getSymbolTable().isConstant(objVn)) return false;
        int memberVn = ((PythonPropertyRead) funcDef).getMemberRef();
        SymbolTable st = node.getIR().getSymbolTable();
        return st.isStringConstant(memberVn) && isMathScalarFunction(st.getStringValue(memberVn));
      }
      // `float(...)` / `int(...)` builtins resolve to builtin classes.
      Set<CGNode> targets =
          builder
              .getCallGraph()
              .getPossibleTargets(node, ((SSAAbstractInvokeInstruction) def).getCallSite());
      if (targets == null || targets.isEmpty()) return false;
      for (CGNode target : targets) {
        String cls = target.getMethod().getDeclaringClass().getReference().getName().toString();
        if (!cls.equals("Lwala/builtin/float") && !cls.equals("Lwala/builtin/int")) return false;
      }
      return true;
    }
    return false;
  }

  /**
   * Whether the given name is a scalar-returning {@code math}-module function.
   *
   * @param name The attribute name read off the receiver.
   * @return {@code true} for the recognized scalar functions.
   */
  private static boolean isMathScalarFunction(String name) {
    switch (name) {
      case "sqrt":
      case "exp":
      case "log":
      case "log2":
      case "log10":
      case "pow":
      case "floor":
      case "ceil":
      case "fabs":
      case "sin":
      case "cos":
      case "tan":
        return true;
      default:
        return false;
    }
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

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The resulting shape is the broadcasted shape of the shapes of x and y.
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    LOGGER.fine(
        () ->
            "EWO.getDefaultShapes entered with source="
                + describe(this.source)
                + ", node="
                + describe(this.getNode()));

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

    // wala/ML#462: when the operands carry per-context shape unions (e.g., `y_true ∈ {[256],
    // [10000]}` for train vs. test), the cartesian product surfaces cross-context pairs (e.g.,
    // `(256, 10000)`) that would never co-occur at runtime under matched contexts. Discard those
    // silently as analysis-level imprecision; only when *every* pair is non-broadcastable does the
    // result shape degrade to ⊤ below (wala/ML#583).
    List<Dimension<?>> sampleNonBroadcastableX = null;
    List<Dimension<?>> sampleNonBroadcastableY = null;
    for (List<Dimension<?>> xShape : xShapes)
      for (List<Dimension<?>> yShape : yShapes)
        if (areBroadcastable(xShape, yShape)) {
          ret.add(getBroadcastedShapes(xShape, yShape));
        } else {
          if (sampleNonBroadcastableX == null) {
            sampleNonBroadcastableX = xShape;
            sampleNonBroadcastableY = yShape;
          }
          LOGGER.fine(
              () ->
                  "EWO.getDefaultShapes: discarding non-broadcastable cross-pair: "
                      + xShape
                      + " vs. "
                      + yShape);
        }

    if (ret.isEmpty() && sampleNonBroadcastableX != null) {
      /*
       * https://github.com/wala/ML/issues/583: every operand-shape pair is non-broadcastable.
       * This is usually a context-collapse artifact (e.g. `accuracy(y_pred, y_true)` analyzed with
       * `[256]` from training and `[10000]` from test, whose per-context shapes get crossed), not a
       * real program error. Degrade the result shape to ⊤ (unknown) so the analysis continues,
       * rather than throwing an exception that aborts the whole run; the dtype is still recovered by
       * getDefaultDTypes. Per-context separation is the precise fix, tracked at
       * https://github.com/wala/ML/issues/530. Log at FINE, not WARNING: element-wise ops are common
       * and the other degrade-to-⊤ paths (Slice, Squeeze) are quiet, so a per-occurrence WARNING
       * would flood normal runs.
       */
      final List<Dimension<?>> x = sampleNonBroadcastableX;
      final List<Dimension<?>> y = sampleNonBroadcastableY;
      LOGGER.fine(
          () ->
              "EWO.getDefaultShapes: all operand-shape pairs are non-broadcastable ("
                  + x
                  + " vs. "
                  + y
                  + "); degrading the result shape to unknown (wala/ML#583).");
      return null;
    }

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

  /** No explicit shape argument; the result shape is the broadcast of the operands' shapes. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }
}
