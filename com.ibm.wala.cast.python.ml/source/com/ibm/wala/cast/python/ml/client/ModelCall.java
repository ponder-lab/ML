package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
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
      return name().toLowerCase();
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
            .flatMap(g -> g.getShapes(builder).stream())
            .collect(Collectors.toSet());

    // Extract shapes from the inputs passed to __call__.
    OrdinalSet<InstanceKey> inputsPTS =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

    if (inputsPTS != null && !inputsPTS.isEmpty()) {
      Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputsPTS);

      if (!inputShapes.isEmpty()) {
        if (outputShapes.isEmpty()) return inputShapes; // Fallback if no output generators found.

        Set<List<Dimension<?>>> refinedShapes = HashSetFactory.make();
        for (List<Dimension<?>> outShape : outputShapes) {
          for (List<Dimension<?>> inShape : inputShapes) {
            if (!outShape.isEmpty() && !inShape.isEmpty()) {
              List<Dimension<?>> newShape = new ArrayList<>(outShape);
              // Keras usually preserves the batch dimension at index 0.
              newShape.set(0, inShape.get(0));
              refinedShapes.add(newShape);
            } else {
              refinedShapes.add(outShape);
            }
          }
        }
        if (!refinedShapes.isEmpty()) return refinedShapes;
      }
    }

    return outputShapes;
  }

  /**
   * Finds tensor generators for the outputs of this model call
   *
   * <p>Finds tensor generators for the outputs of this model call by traversing the points-to graph
   * from the model instance (SELF) to its 'outputs' field, and then to the allocation sites of the
   * output tensors.
   *
   * @param builder The propagation call graph builder.
   * @return A set of tensor generators for the outputs of this model call.
   */
  protected Set<TensorGenerator> getOutputGenerators(PropagationCallGraphBuilder builder) {
    Set<TensorGenerator> ret = HashSetFactory.make();

    // 1. Find the Model instance (SELF).
    OrdinalSet<InstanceKey> selfPts =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());

    for (InstanceKey selfIK : selfPts) {
      // 2. Resolve the Model generator for this instance by finding its allocation site.
      AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);

      if (selfASIN != null) {
        // create a field reference for the `outputs` field of the `Model` instance.
        FieldReference outputsFieldRef =
            FieldReference.findOrCreate(
                selfASIN.getConcreteType().getReference(),
                findOrCreateAsciiAtom(Model.Fields.OUTPUTS.getName()),
                Root);

        IField field = builder.getClassHierarchy().resolveField(outputsFieldRef);

        if (field != null) {
          PointerKey fieldPK = builder.getPointerKeyForInstanceField(selfASIN, field);
          LOGGER.finer(
              "Found field pointer key: "
                  + fieldPK
                  + " for field reference: "
                  + outputsFieldRef
                  + ".");

          OrdinalSet<InstanceKey> outputsPTS = builder.getPointerAnalysis().getPointsToSet(fieldPK);
          LOGGER.finer(
              () ->
                  "Found points-to set: "
                      + outputsPTS
                      + " for field pointer key: "
                      + fieldPK
                      + ".");

          for (InstanceKey outputIK : outputsPTS) {
            LOGGER.finest("Found output instance key: " + outputIK + ".");

            AllocationSiteInNode outputASIN = getAllocationSiteInNode(outputIK);

            if (outputASIN != null) {
              LOGGER.finest(
                  "Found output allocation site: "
                      + outputASIN
                      + " for instance key: "
                      + outputIK
                      + ".");

              TensorGenerator outputGenerator =
                  createManualGenerator(outputASIN.getNode(), builder);
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
      }
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
