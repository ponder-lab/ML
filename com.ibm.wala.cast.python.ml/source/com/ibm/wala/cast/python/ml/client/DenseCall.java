package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.Collections.emptySet;

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
import java.util.Set;

/**
 * A generator for invocations of {@code tf.keras.layers.Dense}.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense">tf.keras.layers.Dense</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu>">Raffi Khatchadourian</a>
 */
public class DenseCall extends TensorGenerator {

  private static final String DENSE_UNITS_FIELD_NAME = "units";

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
      return name().toLowerCase();
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
                selfASIN.getConcreteType().getReference(),
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

          ret.addAll(unitValues);
        }
      }

    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    LOGGER.info("Deriving shape for Dense call at: " + this.getNode());
    // Derive shape from the input tensor.
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

    if (inputPts.isEmpty()) return emptySet();

    Set<List<Dimension<?>>> outputShapes = new HashSet<>();
    Set<Long> unitValues = this.getPossibleUnits(builder);

    for (InstanceKey inputIK : inputPts) {
      LOGGER.fine("Found input tensor instance key: " + inputIK);
      AllocationSiteInNode inputASIN = getAllocationSiteInNode(inputIK);

      if (inputASIN != null) {
        CGNode node = inputASIN.getNode();
        TensorGenerator generator = createManualGenerator(node, builder);
        LOGGER.fine(
            "Found input tensor generator: "
                + generator
                + " for instance key: "
                + inputIK
                + " at node: "
                + node
                + ".");

        if (generator != null) {
          Set<List<Dimension<?>>> inputShapes = generator.getShapes(builder);
          LOGGER.fine(() -> "Found input shapes: " + inputShapes + ".");

          if (inputShapes != null)
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
        } else
          LOGGER.fine(
              "No generator found for instance key: " + inputIK + " at node: " + node + ".");
      }
    }

    return outputShapes;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Derive dtype from the input tensor.
    OrdinalSet<InstanceKey> inputPts =
        this.getArgumentPointsToSet(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
    if (inputPts.isEmpty()) return EnumSet.noneOf(DType.class);
    return this.getDTypesOfValue(builder, inputPts);
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
