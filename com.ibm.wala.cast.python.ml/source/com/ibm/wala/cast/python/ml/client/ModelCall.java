package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * A generator for tensors created by calling a <code>tf.keras.Model</code> instance.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model">tf.keras.Model</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class ModelCall extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(ModelCall.class.getName());

  /** Parameter positions and names for {@code tf.keras.Model.__call__}. */
  protected enum Parameters {
    /** The model itself. */
    SELF,

    /** The input tensors. */
    INPUTS,

    /** The training argument (boolean). */
    TRAINING,

    /** The mask argument (boolean). */
    MASK;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ModelCall(PointsToSetVariable source) {
    super(source);
  }

  public ModelCall(CGNode node) {
    super(node);
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

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> outputShapes =
        this.getOutputGenerators(builder).stream()
            .map(g -> g.getShapes(builder))
            .filter(shapes -> shapes != null)
            .flatMap(Collection::stream)
            .collect(Collectors.toSet());

    // Extract shapes from the inputs passed to __call__.
    OrdinalSet<InstanceKey> inputsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

    // The input shape is only used to refine the batch dimension of a recovered output shape; it is
    // NOT a fallback for the whole output shape. A `tf.keras.Model` generally transforms its input
    // shape (e.g., the GAN discriminator's `Dense(1)` collapses `(batch, 28, 28, 1)` to `(batch,
    // 1)`), so returning the input shape when no output generator resolves emits a load-bearing but
    // unsound result. When the output shape can't be recovered, return ⊤ (null shape) instead and
    // let the dtype axis (see getDefaultDTypes) carry the still-sound dtype. See <a
    // href="https://github.com/wala/ML/issues/537">wala/ML#537</a>.
    if (!outputShapes.isEmpty() && inputsPTS != null && !inputsPTS.isEmpty()) {
      Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputsPTS);

      if (!inputShapes.isEmpty()) {
        Set<List<Dimension<?>>> refinedShapes = HashSetFactory.make();
        for (List<Dimension<?>> outShape : outputShapes) {
          for (List<Dimension<?>> inShape : inputShapes) {
            if (!outShape.isEmpty() && !inShape.isEmpty()) {
              List<Dimension<?>> newShape = new ArrayList<>(outShape);
              // Keras usually preserves the batch dimension at index 0. If the upstream input's
              // batch dim came through as raw `null` (a pre-https://github.com/wala/ML/issues/545
              // unmigrated path), normalize to
              // `DynamicDim` here so the call's output shape doesn't propagate the contract
              // violation downstream.
              Dimension<?> batchDim = inShape.get(0);
              newShape.set(0, batchDim == null ? DynamicDim.INSTANCE : batchDim);
              refinedShapes.add(newShape);
            } else {
              refinedShapes.add(outShape);
            }
          }
        }
        if (!refinedShapes.isEmpty()) return refinedShapes;
      }
    }

    return outputShapes.isEmpty() ? null : outputShapes;
  }

  /**
   * Finds tensor generators for the outputs of this model call.
   *
   * <p>For a functional-API {@code tf.keras.Model(inputs, outputs)}, the output tensor(s) are
   * recovered from the {@code outputs} <em>formal parameter</em> of the synthetic {@code Model.do}
   * node that allocates the model instance (SELF), then mapped to the allocation sites of the
   * output tensors and their manual generators. Reading the model instance's {@code outputs}
   * <em>field</em> instead loses these through synthetic-method PTS-loss (the field's points-to set
   * comes back empty); the formal parameter's points-to set survives. See <a
   * href="https://github.com/wala/ML/issues/537">wala/ML#537</a>.
   *
   * <p>Subclassed models (no {@code outputs} constructor argument; the output flows from the {@code
   * call} method) are not handled here and yield no generators, so {@link #getDefaultShapes}
   * returns ⊤ for them rather than an unsound shape.
   *
   * @param builder The propagation call graph builder.
   * @return A set of tensor generators for the outputs of this model call.
   */
  protected Set<TensorGenerator> getOutputGenerators(PropagationCallGraphBuilder builder) {
    Set<TensorGenerator> ret = HashSetFactory.make();

    // 1. Find the Model instance (SELF).
    OrdinalSet<InstanceKey> selfPts =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());

    // IR index of the `outputs` formal in `Model.do` (paramNames="self inputs outputs name"): the
    // implicit `self` receiver occupies index 0, so the parameter's index is offset by one.
    int outputsParamIRIndex = Model.Parameters.OUTPUTS.getIndex() + 1;

    for (InstanceKey selfIK : selfPts) {
      // 2. Resolve the Model instance's allocation site (the synthetic `Model.do` node).
      AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);
      if (selfASIN == null) continue;

      // Only functional-API models carry an `outputs` constructor argument.
      if (!selfASIN.concreteType().getReference().equals(MODEL.getDeclaringClass())) continue;

      CGNode modelDoNode = selfASIN.getNode();
      IR ir = modelDoNode.getIR();
      if (ir == null) continue;

      // 3. Collect the `outputs` points-to set from two complementary sources, unioned:
      //   (a) the caller-side `Model(...)` invoke's `outputs` argument, resolved by keyword name or
      //       position (covers both `Model(in, out)` and `Model(outputs=out, ...)`); and
      //   (b) the `outputs` formal parameter of the synthetic `Model.do` node (a positional binding
      //       that survives even when the caller invoke can't be located).
      OrdinalSet<InstanceKey> outputsPTS = getOutputsArgumentPTS(builder, modelDoNode);
      if (outputsParamIRIndex < ir.getNumberOfParameters()) {
        int outputsVn = ir.getParameter(outputsParamIRIndex);
        PointerKey outputsPK =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(modelDoNode, outputsVn);
        outputsPTS =
            OrdinalSet.unify(outputsPTS, builder.getPointerAnalysis().getPointsToSet(outputsPK));
      }
      final OrdinalSet<InstanceKey> finalOutputsPTS = outputsPTS;
      LOGGER.finer(
          () ->
              "Found outputs points-to set: "
                  + finalOutputsPTS
                  + " for node: "
                  + modelDoNode
                  + ".");

      for (InstanceKey outputIK : outputsPTS) {
        LOGGER.finest("Found output instance key: " + Loggables.describe(outputIK) + ".");

        AllocationSiteInNode outputASIN = getAllocationSiteInNode(outputIK);
        if (outputASIN != null) {
          LOGGER.finest(
              "Found output allocation site: "
                  + outputASIN
                  + " for instance key: "
                  + outputIK
                  + ".");

          TensorGenerator outputGenerator = createManualGenerator(outputASIN.getNode(), builder);
          if (outputGenerator != null) {
            LOGGER.finest(
                "Created tensor generator: "
                    + outputGenerator
                    + " for output allocation site: "
                    + outputASIN
                    + ".");

            ret.add(outputGenerator);
          }
        }
      }
    }

    return ret;
  }

  /**
   * Resolves the points-to set of the {@code outputs} argument at the caller-side {@code
   * Model(...)} invoke that created the given synthetic {@code Model.do} node. Walks the node's
   * call-graph edges to each calling invoke and reads the {@code outputs} use by keyword name first
   * (covers {@code Model(outputs=...)}) then by position (covers {@code Model(in, out)}). Reading
   * the caller invoke handles keyword construction, which the {@code Model.do} formal-parameter
   * binding alone does not.
   *
   * @param builder The propagation call graph builder.
   * @param modelDoNode The synthetic {@code Model.do} node allocating the model instance.
   * @return The union of the resolved {@code outputs} argument points-to sets, or an empty set if
   *     none could be located.
   */
  private OrdinalSet<InstanceKey> getOutputsArgumentPTS(
      PropagationCallGraphBuilder builder, CGNode modelDoNode) {
    OrdinalSet<InstanceKey> ret = OrdinalSet.empty();
    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, modelDoNode)) {
      CGNode caller = callerInvoke.fst;
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction)) continue;
      PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callerInvoke.snd;

      int argVn = pyCall.getUse(Model.Parameters.OUTPUTS.getName());
      if (argVn == -1) {
        int numPositionalArgs = pyCall.getNumberOfPositionalParameters() - 1;
        int outputsPos = Model.Parameters.OUTPUTS.getIndex();
        if (outputsPos < numPositionalArgs) argVn = pyCall.getUse(outputsPos + 1);
      }
      if (argVn <= 0) continue;

      PointerKey argPK =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, argVn);
      ret = OrdinalSet.unify(ret, builder.getPointerAnalysis().getPointsToSet(argPK));
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> outputDTypes =
        this.getOutputGenerators(builder).stream()
            .flatMap(g -> g.getDTypes(builder).stream())
            .collect(Collectors.toSet());

    if (outputDTypes.isEmpty()) {
      // Extract dtypes from the inputs passed to __call__.
      OrdinalSet<InstanceKey> inputsPTS =
          this.getArgumentPointsToSet(
              builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

      if (inputsPTS != null && !inputsPTS.isEmpty()) {
        Set<DType> inputDTypes = this.getDTypesOfValue(builder, inputsPTS);
        if (!inputDTypes.isEmpty()) return inputDTypes;
      }
    }

    // If we couldn't find any dtypes, default to FLOAT32.
    if (outputDTypes.isEmpty()) outputDTypes.add(FLOAT32);

    return outputDTypes;
  }
}
