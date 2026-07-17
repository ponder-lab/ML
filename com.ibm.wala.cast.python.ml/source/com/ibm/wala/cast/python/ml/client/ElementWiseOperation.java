package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.areBroadcastable;
import static com.ibm.wala.cast.python.ml.util.TensorShapeUtil.getBroadcastedShapes;
import static java.util.Collections.emptyList;
import static java.util.Collections.singleton;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
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
import com.ibm.wala.util.intset.OrdinalSet;
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
   * Returns the origin of a binary operator's result by classifying its operands (wala/ML#724).
   * Runtime binary-operator dispatch decides the result's origin: {@code ndarray + ndarray} stays
   * numpy, any TensorFlow operand makes the operation a TensorFlow one and the result a {@code
   * tf.Tensor}, and otherwise a parameter operand makes the traced operator a TensorFlow op whose
   * result is the frame's symbolic value (wala/ML#726). Each non-parameter operand classifies
   * through its producing generator (via {@link TensorGeneratorFactory#getGenerator}), recursing
   * structurally through nested binary operators (whose results have no points-to set to dispatch
   * on); a statically opaque scalar operand is skipped, since it keeps the tensor operand's
   * library. An operand without origin evidence contributes {@link TensorOrigin#TENSORFLOW},
   * preserving the pre-wala/ML#724 reading where the analysis cannot prove numpy provenance.
   *
   * <p>A manual anchor models a TensorFlow API call ({@code tf.multiply.do()} etc.), so the default
   * applies there.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The producing libraries of the operation's result.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    if (this.source == null || !(this.source.getPointerKey() instanceof LocalPointerKey))
      return super.getOrigins(builder);
    LocalPointerKey lpk = (LocalPointerKey) this.source.getPointerKey();
    SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
    if (!(def instanceof SSABinaryOpInstruction)) return super.getOrigins(builder);
    return this.getBinaryOperationOrigins(builder, lpk.getNode(), (SSABinaryOpInstruction) def);
  }

  /**
   * Classifies a binary operator's result by the per-execution dispatch product of its operands'
   * origins: an execution pairing a TensorFlow operand with anything yields a TensorFlow result, a
   * parameter operand paired with anything but TensorFlow yields the hybridization frame's symbolic
   * value, and only a numpy-with-numpy pairing yields an ndarray. The product is more precise than
   * a plain union: {@code {NUMPY} + {TENSORFLOW}} is {@code {TENSORFLOW}}, while {@code {NUMPY} +
   * {NUMPY, TENSORFLOW}} keeps both.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node whose IR defines the operator.
   * @param binop The binary operator instruction.
   * @return The origins of the operator's result.
   */
  private Set<TensorOrigin> getBinaryOperationOrigins(
      PropagationCallGraphBuilder builder, CGNode node, SSABinaryOpInstruction binop) {
    Set<TensorOrigin> x = this.getOperandOrigins(builder, node, binop.getUse(0));
    Set<TensorOrigin> y = this.getOperandOrigins(builder, node, binop.getUse(1));

    // A scalar (or otherwise skipped) operand keeps the other operand's origins.
    if (x == null && y == null) return EnumSet.of(TensorOrigin.TENSORFLOW);
    if (x == null) return y;
    if (y == null) return x;

    Set<TensorOrigin> ret = EnumSet.noneOf(TensorOrigin.class);
    for (TensorOrigin ox : x) for (TensorOrigin oy : y) ret.add(getDispatchedOrigin(ox, oy));
    return ret;
  }

  /**
   * Returns the origin a binary operator's runtime dispatch assigns to one pairing of operand
   * origins, under the dominance order {@link TensorOrigin#TENSORFLOW} &gt; {@link
   * TensorOrigin#PARAMETER} &gt; {@link TensorOrigin#NUMPY}: any TensorFlow operand makes the
   * operation a TensorFlow one and the result a {@code tf.Tensor}; otherwise a parameter operand
   * makes the traced operator a TensorFlow op whose result is the frame's symbolic value
   * (wala/ML#726); only numpy with numpy stays an ndarray.
   *
   * @param x One operand's origin.
   * @param y The other operand's origin.
   * @return The origin of the operator's result for this pairing.
   */
  private static TensorOrigin getDispatchedOrigin(TensorOrigin x, TensorOrigin y) {
    if (x == TensorOrigin.TENSORFLOW || y == TensorOrigin.TENSORFLOW)
      return TensorOrigin.TENSORFLOW;
    if (x == TensorOrigin.PARAMETER || y == TensorOrigin.PARAMETER) return TensorOrigin.PARAMETER;
    return TensorOrigin.NUMPY;
  }

  /**
   * Classifies one binary-operator operand's origins. A parameter of the defining frame carries the
   * hybridization-frame origin (wala/ML#726): under tracing it is a symbolic tensor whatever its
   * eager feeds, so it classifies as {@link TensorOrigin#PARAMETER} rather than through the
   * creators its call sites feed it. A nested binary operator recurses structurally (its result has
   * no points-to set to dispatch on); any other operand classifies through the generator its
   * points-to chain dispatches to.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param node The node whose IR defines the operand.
   * @param vn The operand's value number.
   * @return The operand's origins, or {@code null} when the operand contributes no origin evidence
   *     (a scalar expression, or an unresolvable points-to chain).
   */
  private Set<TensorOrigin> getOperandOrigins(
      PropagationCallGraphBuilder builder, CGNode node, int vn) {
    if (this.isScalarExpression(builder, node, vn)) return null;

    if (node.getIR().getSymbolTable().isParameter(vn)) return EnumSet.of(TensorOrigin.PARAMETER);

    SSAInstruction def = node.getDU().getDef(vn);
    if (def instanceof SSABinaryOpInstruction)
      return this.getBinaryOperationOrigins(builder, node, (SSABinaryOpInstruction) def);

    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    if (builder.getPropagationSystem().isImplicit(pk)) return null;
    PointsToSetVariable operand = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
    if (operand == null) return null;

    TensorGenerator generator;
    try {
      generator = TensorGeneratorFactory.getGenerator(operand, builder);
    } catch (IllegalArgumentException e) {
      return null;
    }
    return generator == null ? null : generator.getOrigins(builder);
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
    if (this.source == null) return this.getManualShapeResult(builder);
    int xVn = this.getXArgumentValueNumber(builder);
    if (xVn <= 0) return ShapeResult.unknown();
    Set<List<Dimension<?>>> xShapes = this.getOperandShapes(builder, xVn);
    int yVn = this.getYArgumentValueNumber(builder);
    if (yVn <= 0) return ShapeResult.unknown();
    Set<List<Dimension<?>>> yShapes = this.getOperandShapes(builder, yVn);

    boolean xHas = xShapes != null && !xShapes.isEmpty();
    boolean yHas = yShapes != null && !yShapes.isEmpty();
    LOGGER.fine(
        () ->
            "EWO record broadcast for source "
                + describe(this.getSource())
                + ": xVn "
                + xVn
                + " shapes "
                + xShapes
                + ", yVn "
                + yVn
                + " shapes "
                + yShapes
                + ".");
    if (xHas != yHas) {
      Set<List<Dimension<?>>> resolved = xHas ? xShapes : yShapes;
      int opaqueVn = xHas ? yVn : xVn;
      if (resolved.stream().anyMatch(dims -> !dims.isEmpty())
          && isScalarExpression(builder, opaqueVn)) return ShapeResult.of(resolved);
    }
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * Broadcast resolution for a manually anchored generator (producer delegation to a synthetic op
   * node, e.g. {@code tf.multiply.do()}): the operands resolve through the caller-aware argument
   * machinery, since the value-number-based operand path pairs caller-frame value numbers with the
   * synthetic node's IR (wala/ML#718). The one-sided scalar rule tests the opaque argument per
   * caller frame.
   *
   * @param builder The propagation call graph builder.
   * @return The broadcast result.
   */
  private ShapeResult getManualShapeResult(PropagationCallGraphBuilder builder) {
    ShapeResult x =
        this.manualArgShapeResult(builder, getXParameterPosition(), getXParameterName());
    ShapeResult y =
        this.manualArgShapeResult(builder, getYParameterPosition(), getYParameterName());

    boolean xHas = !x.members().isEmpty();
    boolean yHas = !y.members().isEmpty();
    if (xHas && yHas) {
      Set<List<Dimension<?>>> ret = HashSetFactory.make();
      for (List<Dimension<?>> xShape : x.members())
        for (List<Dimension<?>> yShape : y.members())
          if (areBroadcastable(xShape, yShape)) ret.add(getBroadcastedShapes(xShape, yShape));
      return ret.isEmpty()
          ? ShapeResult.unknown()
          : new ShapeResult(ret, x.hasUnknown() || y.hasUnknown());
    }
    if (xHas != yHas) {
      Set<List<Dimension<?>>> resolved = xHas ? x.members() : y.members();
      int opaquePos = xHas ? getYParameterPosition() : getXParameterPosition();
      if (resolved.stream().anyMatch(dims -> !dims.isEmpty())
          && this.isScalarArgumentInAllCallers(builder, opaquePos))
        return new ShapeResult(resolved, x.hasUnknown() || y.hasUnknown());
    }
    return ShapeResult.unknown();
  }

  /**
   * Caller-aware argument-shape resolution for the manual anchoring: the argument's points-to union
   * first (exact), then the per-context caller walk, with the union's members as the floor
   * (wala/ML#716, wala/ML#718).
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @param paramName The keyword parameter name.
   * @return The resolution result.
   */
  private ShapeResult manualArgShapeResult(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    ShapeResult fromValue = ShapeResult.unknown();
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, paramPos, paramName);
    if (pts != null && !pts.isEmpty()) {
      fromValue = this.getShapeResultOfValue(builder, pts, true);
      if (!fromValue.members().isEmpty() && !fromValue.hasUnknown()) return fromValue;
    }
    ShapeResult viaCallers = this.getArgumentShapeResultViaCallers(builder, paramPos, paramName);
    if (!viaCallers.members().isEmpty()) return viaCallers;
    return fromValue.members().isEmpty() ? viaCallers : fromValue;
  }

  /**
   * Whether the argument at the given position is structurally a scalar expression in every caller
   * frame (wala/ML#718).
   *
   * @param builder The propagation call graph builder.
   * @param paramPos The positional index of the arg.
   * @return {@code true} iff every resolvable caller passes a scalar expression.
   */
  private boolean isScalarArgumentInAllCallers(PropagationCallGraphBuilder builder, int paramPos) {
    boolean saw = false;
    for (com.ibm.wala.util.collections.Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, this.getNode())) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;
      int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
      if (paramPos >= numPosParams) continue;
      int argVn = pyCall.getUse(paramPos + 1);
      if (argVn <= 0) continue;
      saw = true;
      if (!isScalarExpression(builder, callerInvoke.fst, argVn)) return false;
    }
    return saw;
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
    return isScalarExpression(builder, this.getNode(), vn);
  }

  /**
   * Node-explicit core of {@link #isScalarExpression(PropagationCallGraphBuilder, int)}, so a
   * manually anchored generator can test an argument in its caller's frame (wala/ML#718).
   *
   * @param builder The propagation call graph builder.
   * @param node The {@link CGNode} whose IR defines {@code vn}.
   * @param vn The operand's value number.
   * @return {@code true} iff the operand is structurally a Python scalar expression.
   */
  private boolean isScalarExpression(PropagationCallGraphBuilder builder, CGNode node, int vn) {
    if (vn <= 0) return false;
    if (node.getDU() == null || node.getIR() == null) return false;
    if (node.getIR().getSymbolTable().isConstant(vn)) return true;

    // A value whose every points-to member is a numeric constant is a runtime scalar whatever its
    // syntactic form — e.g. a scalar-defaulted parameter like the vendored LayerNormalization's
    // `epsilon=1e-6`, bound to the default's constant by the call trampoline (wala/ML#739).
    PointerKey scalarPk =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
    OrdinalSet<InstanceKey> scalarPts = builder.getPointerAnalysis().getPointsToSet(scalarPk);
    if (scalarPts != null && !scalarPts.isEmpty()) {
      boolean allNumericConstants = true;
      for (InstanceKey ik : scalarPts)
        if (!(ik instanceof ConstantKey) || !(((ConstantKey<?>) ik).getValue() instanceof Number)) {
          allNumericConstants = false;
          break;
        }
      if (allNumericConstants) return true;
    }

    SSAInstruction def = node.getDU().getDef(vn);
    if (def instanceof SSABinaryOpInstruction)
      return isScalarExpression(builder, node, def.getUse(0))
          && isScalarExpression(builder, node, def.getUse(1));
    if (def instanceof SSAUnaryOpInstruction)
      return isScalarExpression(builder, node, def.getUse(0));
    // A shape-vector element (e.g. `tf.shape(x)[k]`) is a rank-0 int32 tensor: structurally
    // scalar for broadcast purposes even though it is a runtime tensor, so a broadcast against it
    // preserves the tensor operand's shape. Only the singular element qualifies; a shape-vector
    // slice is rank-1 and must not classify as scalar. See wala/ML#723.
    if (def instanceof PythonPropertyRead) {
      Dimension<?> element =
          this.resolveShapeVectorElementDim(builder, node, node.getIR().getSymbolTable(), vn);
      LOGGER.fine(
          () -> "Shape-vector element check for operand vn " + vn + " resolved: " + element + ".");
      if (element != null) return true;
    }
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

    // One-sided broadcast, mirroring the wala/ML#718 record rule at this layer, which the
    // nested-operand resolution routes through: a statically opaque scalar co-operand preserves
    // the resolved side's shapes. Covers a tf.shape-element co-operand, a rank-0 tensor at
    // runtime (wala/ML#723). Manual anchors resolve arguments caller-side and keep their own
    // scalar rule, so this applies to source-anchored instances only.
    if (this.getSource() != null && xShapes.isEmpty() != yShapes.isEmpty()) {
      Set<List<Dimension<?>>> resolved = xShapes.isEmpty() ? yShapes : xShapes;
      int opaqueVn = xShapes.isEmpty() ? xVn : yVn;
      if (resolved.stream().anyMatch(dims -> !dims.isEmpty())
          && this.isScalarExpression(builder, opaqueVn)) return resolved;
    }

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
   * The type-determining operands are both sides of the element-wise operation, resolved in this
   * generator's own frame (wala/ML#736): the result's member types compose as the pairwise
   * broadcast of the operands' converged dataflow members (wala/ML#682). A scalar-constant operand
   * broadcasts as the identity, so the other operand's members pass through unchanged; two scalar
   * constants declare no feed.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The broadcast feed over both operands, the pass-through feed over the tensor operand
   *     when the other is a scalar constant, or {@code null} when neither operand resolves.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    CGNode node = this.getNode();
    int xVn = this.getXArgumentValueNumber(builder);
    int yVn = this.getYArgumentValueNumber(builder);
    if (node.getIR() == null) return null;
    SymbolTable st = node.getIR().getSymbolTable();
    boolean xScalar = xVn <= 0 || st.isConstant(xVn);
    boolean yScalar = yVn <= 0 || st.isConstant(yVn);
    if (xScalar && yScalar) return null;
    if (xScalar || yScalar) {
      int tensorVn = xScalar ? yVn : xVn;
      return new TypeFeed(
          TypeFeedKind.PASS_THROUGH,
          List.of(
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, tensorVn)));
    }
    return new TypeFeed(
        TypeFeedKind.BROADCAST,
        List.of(
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, xVn),
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, yVn)));
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
