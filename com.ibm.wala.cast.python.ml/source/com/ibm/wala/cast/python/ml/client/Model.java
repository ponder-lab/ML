package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.client.DenseCall.Parameters;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

/**
 * A generator for tensors created by the <code>tf.keras.Model()</code> function in TensorFlow.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model">tf.keras.Model</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class Model extends TensorGenerator {

  /** Parameter positions and names for {@code tf.keras.Model}. */
  protected enum Parameters {
    /** The model itself. */
    SELF,

    /** The input(s) of the model. */
    INPUTS,

    /** The output(s) of the model. */
    OUTPUTS,

    /** Name of the model. */
    NAME;

    public String getName() {
      return name().toLowerCase();
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public Model(PointsToSetVariable source) {
    super(source);
  }

  public Model(CGNode node) {
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
    return getWeightShapes(builder, this.getSource());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return getWeightDTypes(builder, this.getSource());
  }

  /**
   * Infers the shapes of the model's weights by traversing the computation graph backwards from the
   * model's outputs.
   *
   * @param builder The propagation call graph builder.
   * @param modelSource The source points-to set variable representing the model.
   * @return A set of weight shapes.
   */
  public Set<List<Dimension<?>>> getWeightShapes(
      PropagationCallGraphBuilder builder, PointsToSetVariable modelSource) {
    Set<List<Dimension<?>>> weightShapes = HashSetFactory.make();

    OrdinalSet<InstanceKey> outputsPts =
        this.getArgumentPointsToSet(
            builder, Parameters.OUTPUTS.getIndex(), Parameters.OUTPUTS.getName());

    Set<CGNode> visited = HashSetFactory.make();
    Queue<CGNode> queue = new LinkedList<>();

    for (InstanceKey outputIK : outputsPts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(outputIK);
      if (asin != null) {
        queue.add(asin.getNode());
      }
    }

    while (!queue.isEmpty()) {
      CGNode node = queue.poll();
      if (!visited.add(node)) continue;

      TensorGenerator generator = createManualGenerator(node, builder);
      if (generator instanceof DenseCall denseCall) {
        int valNum =
            denseCall.getArgumentValueNumber(
                builder,
                DenseCall.Parameters.INPUTS.getIndex(),
                DenseCall.Parameters.INPUTS.getName(),
                false);
        Set<List<Dimension<?>>> inputShapes = denseCall.getShapes(builder, valNum);
        Set<Long> unitsValues = denseCall.getPossibleUnits(builder);

        for (List<Dimension<?>> inputShape : inputShapes) {
          if (inputShape.isEmpty()) continue;
          Dimension<?> lastDim = inputShape.get(inputShape.size() - 1);
          for (Long units : unitsValues) {
            // Kernel shape: (input_dim, units)
            List<Dimension<?>> kernelShape = new ArrayList<>();
            kernelShape.add(lastDim);
            kernelShape.add(new NumericDim(units.intValue()));
            weightShapes.add(kernelShape);

            // Bias shape: (units,)
            List<Dimension<?>> biasShape = new ArrayList<>();
            biasShape.add(new NumericDim(units.intValue()));
            weightShapes.add(biasShape);
          }
        }
      }

      // Trace back to preceding layers
      if (generator != null) {
        for (int i = 0; i < 10; i++) { // Check up to 10 positional arguments
          OrdinalSet<InstanceKey> inputs = generator.getArgumentPointsToSet(builder, i, null);
          for (InstanceKey inputIK : inputs) {
            AllocationSiteInNode asin = getAllocationSiteInNode(inputIK);
            if (asin != null) {
              queue.add(asin.getNode());
            }
          }
        }
      }
    }

    return weightShapes;
  }

  /**
   * Infers the dtypes of the model's weights.
   *
   * @param builder The propagation call graph builder.
   * @param modelSource The source points-to set variable representing the model.
   * @return A set of weight dtypes.
   */
  public Set<DType> getWeightDTypes(
      PropagationCallGraphBuilder builder, PointsToSetVariable modelSource) {
    // For now, we assume float32 for most Keras weights.
    return EnumSet.of(DType.FLOAT32);
  }
}
