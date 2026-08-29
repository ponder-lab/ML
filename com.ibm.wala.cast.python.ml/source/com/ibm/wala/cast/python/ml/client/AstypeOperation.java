package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Modeling of the NumPy astype() operation. This operation preserves the shape of the receiver
 * tensor.
 */
public class AstypeOperation extends TensorGenerator {
  private static final Logger LOGGER = Logger.getLogger(AstypeOperation.class.getName());

  public AstypeOperation(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Manual (node-based) anchor, for producer delegation from a {@code numpy/ndarray} allocation
   * made inside this operation's synthetic body (wala/ML#849).
   *
   * @param node The synthetic {@code do()} node that allocated the value.
   */
  public AstypeOperation(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    int receiverVn = getReceiverVn();
    LOGGER.fine(
        () ->
            "AstypeOperation.getDefaultShapes: source="
                + describe(source)
                + ", receiverVn="
                + receiverVn);
    if (receiverVn > 0) {
      try {
        Set<List<Dimension<?>>> shapes = getShapes(builder, getNode(), receiverVn);
        LOGGER.fine(
            () ->
                "AstypeOperation.getDefaultShapes: shapes from receiverVn="
                    + receiverVn
                    + " -> "
                    + shapes);
        if (shapes != null && !shapes.isEmpty()) {
          return shapes;
        }
      } catch (IllegalArgumentException e) {
        // `getShapes` throws when the receiver's PTS is empty AND its PointerKey is implicit —
        // e.g., a chained `x.astype(int32).astype(float32)` where the inner call's return value
        // is a synthetic-method return (implicit PK, no materialised PTS). The multi-stage
        // helper in `TensorGenerator.getShapes(builder, node, vn)` skips the factory-recursion
        // branch for implicit keys and falls through to IAE. Catch and return `null` (⊤ unknown
        // shape) so dtype inference still proceeds and the result flows downstream as a tensor
        // instead of being dropped entirely. For the non-chained mnist case, the factory
        // recursion fires successfully via `MnistInputData{@code , so the shape is recovered and
        // this
        // catch doesn't fire. See wala/ML#356, wala/WALA#1889.
        LOGGER.log(
            Level.FINE,
            "AstypeOperation.getDefaultShapes: receiver shape lookup failed for receiverVn="
                + receiverVn,
            e);
      }
    }
    // The receiver is never passed as an argument, so a manual anchor has no value number for it
    // at all: `getReceiverVn` reads the property read that def'd the invoke's function object, and
    // a manual anchor has no invoke. Ask the caller-aware walk instead, which finds the `x` of
    // `x.astype(...)` at each call site dispatching to this body. Without this a producer
    // delegation to this operation resolves nothing, and a consumer of the narrowed array — a
    // tensor conversion, say — unions in a ⊤-shaped member beside the exact one and summarizes the
    // whole value to unknown rank (wala/ML#849). It doubles as a fallback for the source anchor,
    // whose caller-frame chain above does not cover every producer.
    OrdinalSet<InstanceKey> receiverPointsToSet =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, null);
    Set<List<Dimension<?>>> preserved = this.getShapesOfValue(builder, receiverPointsToSet);
    LOGGER.fine(
        () ->
            "AstypeOperation.getDefaultShapes: shapes via the receiver walk -> " + preserved + ".");
    // ⊤ (unknown shape) when the receiver does not resolve: the value is still a narrowed array
    // like its receiver, never ⊥.
    return preserved == null || preserved.isEmpty() ? null : preserved;
  }

  private int getReceiverVn() {
    com.ibm.wala.cast.python.ssa.PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int funcVn = call.getUse(0);
      SSAInstruction funcDef = getNode().getDU().getDef(funcVn);
      if (funcDef instanceof PythonPropertyRead) {
        return ((PythonPropertyRead) funcDef).getObjectRef();
      }
    }
    return getArgumentValueNumber(RECEIVER_PARAMETER_POSITION);
  }

  /**
   * The dtype comes from the {@code dtype} argument, read through the declarative hooks below so
   * that the caller-aware machinery resolves it; this method is only the fallback for when that
   * argument does not resolve.
   *
   * <p>It used to read {@code getArgumentValueNumber(0)}, which resolves a value number against
   * <em>this generator's own node</em> rather than against the call's arguments. For the
   * source-anchored path that node is the caller's body, so the read landed on the caller's formal
   * parameters, found nothing, and fell through to a hard-coded {@code FLOAT32}. Since a narrowing
   * to float32 is also the commonest one, the fallback agreed with the truth exactly often enough
   * to look like inference; every other target ({@code astype(np.int32)} and friends) was reported
   * as a confident float32 (wala/ML#849).
   *
   * <p>{@code astype} requires a dtype at runtime, so an unresolved one means the analysis could
   * not read it, not that there is a default to supply. It degrades to ⊤ accordingly.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link DType#UNKNOWN}, singleton.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * The dtype is the only bindable argument of the method form: the array is the receiver, so the
   * dtype sits at position 0 after {@code self}, matching {@code numpy/ndarray/astype}'s {@code
   * self dtype} layout. Declaring both the position and the name lets the base class resolve either
   * spelling, {@code astype(np.int32)} and {@code astype(dtype=np.int32)}.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return 0;
  }

  @Override
  protected String getDTypeParameterName() {
    return "dtype";
  }

  /**
   * Returns the producing library of the modeled value: an }ndarray.astype(...)` call, so the value
   * is an ndarray (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
