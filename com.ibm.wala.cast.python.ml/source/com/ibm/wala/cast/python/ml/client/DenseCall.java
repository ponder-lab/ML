package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for invocations of {@code tf.keras.layers.Dense}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense">tf.keras.layers.Dense</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu>">Raffi Khatchadourian</a>
 */
public class DenseCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(DenseCall.class.getName());

  private static final String DENSE_UNITS_FIELD_NAME = "units";

  /**
   * Per-thread set of {@code Dense.__call__} nodes whose input shape is currently being computed.
   *
   * <p>Breaks unbounded recursion when a layer call's input flows back to itself. The motivating
   * case is a loop over a list of layers ({@code for layer in self.my_layers: x = layer(x)}): under
   * 1-CFA the list elements collapse to a single {@code Dense.__call__} node, so that node's input
   * points-to set includes its own output, and {@link #getInputShapes} would otherwise recurse on
   * the same node forever. On re-entry for an in-progress node we return {@code null}, letting the
   * other phi operand (e.g., the upstream {@code Flatten} output) supply the base shape. See
   * wala/ML#599.
   */
  private static final ThreadLocal<Set<CGNode>> INPUT_SHAPE_NODES_IN_PROGRESS =
      ThreadLocal.withInitial(HashSet::new);

  /**
   * Parameter positions and names for calls to {@code tf.keras.layers.Dense}.
   *
   * @see <a
   *     href="https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/layers/core/dense.py#L149">Dense.call</a>
   */
  protected enum Parameters {
    /** The layer itself. */
    SELF,

    /** The input tensors. */
    INPUTS;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public DenseCall(PointsToSetVariable source) {
    super(source);
  }

  public DenseCall(CGNode node) {
    super(node);
  }

  /**
   * Derives possible values for the 'units' parameter of the `Dense` layer.
   *
   * <p>Derives possible values for the 'units' parameter of the `Dense` layer by analyzing the
   * points-to set of the `self` parameter (the `Dense` layer instance) and extracting the value of
   * the 'units' field from that instance.
   *
   * @param builder The call graph builder used to perform points-to analysis and resolve field
   *     references.
   * @return A set of possible values for the 'units' parameter of the `Dense` layer.
   */
  protected Set<Long> getPossibleUnits(PropagationCallGraphBuilder builder) {
    Set<Long> ret = new HashSet<>();

    // Extract the 'units' value from the Dense layer instance (Parameters.SELF).
    OrdinalSet<InstanceKey> selfPTS =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());
    LOGGER.fine(
        () -> "Found `self` points-to set: " + selfPTS + " for node: " + this.getNode() + ".");

    if (selfPTS != null)
      for (InstanceKey selfIK : selfPTS) {
        LOGGER.finer(
            () -> "Found `self` instance key: " + selfIK + " for node: " + this.getNode() + ".");

        // Extract the 'units' value from the Dense layer instance (Parameters.SELF).
        AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);

        // Create a field reference for the 'units' field of the Dense layer instance.
        FieldReference unitsFieldRef =
            FieldReference.findOrCreate(
                selfASIN.concreteType().getReference(),
                findOrCreateAsciiAtom(DENSE_UNITS_FIELD_NAME),
                Root);

        IField f = builder.getClassHierarchy().resolveField(unitsFieldRef);

        if (f != null) {
          PointerKey fieldPK = builder.getPointerKeyForInstanceField(selfASIN, f);
          LOGGER.finer(
              "Field pointer key: " + fieldPK + " for field reference: " + unitsFieldRef + ".");

          OrdinalSet<InstanceKey> unitsPTS = builder.getPointerAnalysis().getPointsToSet(fieldPK);
          LOGGER.finer("Points-to set: " + unitsPTS + " for field pointer key: " + fieldPK + ".");

          Set<Long> unitValues = getPossibleLongValues(unitsPTS);
          LOGGER.finer(
              "Possible `units` values: " + unitValues + " for points-to set: " + unitsPTS + ".");

          // A null result means `units` is not statically resolvable (wala/ML#669); skip it
          // rather than crashing or claiming a partial value.
          if (unitValues != null) ret.addAll(unitValues);
        }
      }

    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    LOGGER.fine(() -> "Deriving shape for Dense call at: " + this.getNode());

    Set<List<Dimension<?>>> inputShapes = this.getInputShapes(builder);
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    Set<Long> unitValues = this.getPossibleUnits(builder);
    if (unitValues.isEmpty()) return inputShapes; // Preserve input shapes if units unknown.

    Set<List<Dimension<?>>> outputShapes = new HashSet<>();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.isEmpty()) continue;

      // For each possible value of 'units', set the last dimension of the output
      // shape to that value.
      for (Long units : unitValues)
        if (units != null) {
          // Create a new output shape based on the input shape, but with the last
          // dimension set to 'units'.
          List<Dimension<?>> outShape = new ArrayList<>(inputShape);
          outShape.set(outShape.size() - 1, new NumericDim(units.intValue()));
          outputShapes.add(outShape);
        }
    }

    return outputShapes;
  }

  /**
   * Resolves the input tensor shapes for this {@code Dense.__call__} invocation, guarding against
   * cyclic self-recursion before delegating to {@link
   * #computeInputShapes(PropagationCallGraphBuilder)}.
   *
   * <p>The guard ({@link #INPUT_SHAPE_NODES_IN_PROGRESS}) breaks the recursion that arises when a
   * layer call's input flows back to itself — e.g., a loop over a list of layers whose collapsed
   * 1-CFA node feeds its own output back in as input. See wala/ML#599.
   *
   * @param builder The propagation call graph builder used for points-to analysis and factory
   *     dispatch.
   * @return The set of possible input shapes, or {@code null} if neither path resolves a shape or
   *     the node is already being resolved on this thread (cycle).
   */
  private Set<List<Dimension<?>>> getInputShapes(PropagationCallGraphBuilder builder) {
    CGNode self = this.getNode();
    Set<CGNode> inProgress = INPUT_SHAPE_NODES_IN_PROGRESS.get();
    if (!inProgress.add(self)) {
      // Cyclic chain: this `Dense.__call__` node's input flows back to itself (e.g., a layer-list
      // loop whose output feeds in as its own input under 1-CFA node collapse). Break the cycle;
      // the other phi operand supplies the base shape. See wala/ML#599.
      LOGGER.fine(() -> "getInputShapes: cycle detected at node " + self + "; breaking recursion.");
      return null;
    }
    try {
      return this.computeInputShapes(builder);
    } finally {
      inProgress.remove(self);
    }
  }

  /**
   * Computes the input tensor shapes for this {@code Dense.__call__} invocation. Always invoked
   * under the {@link #INPUT_SHAPE_NODES_IN_PROGRESS} cycle guard established by {@link
   * #getInputShapes(PropagationCallGraphBuilder)}.
   *
   * <p>Primary path: walks the {@code inputs} argument's points-to set and dispatches the
   * allocating node through {@link #createManualGenerator(CGNode, PropagationCallGraphBuilder)}.
   * This works when the input is a producer recognized by the manual-dispatch path (e.g., {@code
   * tf.keras.Input}, {@code tf.random.uniform}, {@code tf.ones}).
   *
   * <p>Fallback path: when the primary path yields nothing — typically because the input is the
   * result of another layer call (e.g., {@code Flatten.__call__}, {@code Dense.__call__}, {@code
   * Dropout.__call__}), whose allocating node type is not in {@link #createManualGenerator(CGNode,
   * PropagationCallGraphBuilder)}'s switch — walks to the {@code inputs} value number in this
   * summary method's IR and delegates to {@link #getShapesOrSSAChain(PropagationCallGraphBuilder,
   * CGNode, int)}. That path re-enters the factory via {@link TensorGeneratorFactory#getGenerator}
   * on the upstream call's result, which knows {@code FLATTEN_LAYER_CALL}, {@code DENSE_CALL}, etc.
   * See wala/ML#358.
   *
   * @param builder The propagation call graph builder used for points-to analysis and factory
   *     dispatch.
   * @return The set of possible input shapes, or {@code null} if neither the PTS walk nor the
   *     SSA-chain fallback resolves a shape.
   */
  private Set<List<Dimension<?>>> computeInputShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

    Set<List<Dimension<?>>> ret = new HashSet<>();

    for (InstanceKey inputIK : inputPts) {
      LOGGER.fine(() -> "Found input tensor instance key: " + inputIK);
      AllocationSiteInNode inputASIN = getAllocationSiteInNode(inputIK);
      if (inputASIN == null) continue;

      CGNode node = inputASIN.getNode();
      TensorGenerator generator = createManualGenerator(node, builder);
      LOGGER.fine(
          () ->
              "Found input tensor generator: "
                  + generator
                  + " for instance key: "
                  + inputIK
                  + " at node: "
                  + node
                  + ".");

      if (generator != null) {
        Set<List<Dimension<?>>> generatorShapes = generator.getShapes(builder);
        LOGGER.fine(() -> "Found input shapes: " + generatorShapes + ".");
        if (generatorShapes != null) ret.addAll(generatorShapes);
      } else {
        LOGGER.fine(
            () -> "No generator found for instance key: " + inputIK + " at node: " + node + ".");
      }
    }

    if (!ret.isEmpty()) return ret;

    // Fallback: chained layer calls. The input's allocating node is a layer `__call__` summary
    // whose class isn't recognised by `createManualGenerator` (Flatten, Dense, Dropout, ...).
    // Re-dispatch through `getShapesOrSSAChain` on the `inputs` value number so the factory's
    // caller-side parameter trace-back can pick up the upstream generator. See wala/ML#358.
    int inputsVn =
        this.getArgumentValueNumber(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName(), true);
    if (inputsVn <= 0) return null;
    LOGGER.fine(
        () -> "PTS walk produced no shapes; attempting SSA-chain fallback on vn=" + inputsVn + ".");
    try {
      Set<List<Dimension<?>>> viaSsa = this.getShapesOrSSAChain(builder, this.getNode(), inputsVn);
      LOGGER.fine(() -> "SSA-chain fallback shapes for vn=" + inputsVn + ": " + viaSsa + ".");
      if (viaSsa != null && !viaSsa.isEmpty()) return viaSsa;
    } catch (IllegalArgumentException e) {
      LOGGER.fine(() -> "SSA-chain fallback IAE for vn=" + inputsVn + ": " + e.getMessage() + ".");
    }
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Keras layers like Dense usually output FLOAT32 by default, regardless of input dtype.
    return EnumSet.of(DType.FLOAT32);
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
