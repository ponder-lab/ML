package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.client.DenseCall.Parameters;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.types.PythonTypes;
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
import java.util.Locale;
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
    /** The input(s) of the model. */
    INPUTS,

    /** The output(s) of the model. */
    OUTPUTS,

    /** Name of the model. */
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
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
    return this.getWeightShapes(builder, this.getSource());
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.getWeightDTypes(builder, this.getSource());
  }

  /** The prefix of the types of the Keras layers' summaries. */
  static final String KERAS_LAYERS_PREFIX = "Ltensorflow/keras/layers/";

  /**
   * The Keras layers that hold no weights, so a model's weights need nothing from them
   * (wala/ML#1123).
   */
  static final Set<String> WEIGHTLESS_LAYERS =
      Set.of(
          "Activation",
          "Add",
          "Average",
          "AveragePooling1D",
          "AveragePooling2D",
          "AveragePooling3D",
          "Concatenate",
          "Cropping1D",
          "Cropping2D",
          "Cropping3D",
          "Dropout",
          "Flatten",
          "GaussianDropout",
          "GaussianNoise",
          "GlobalAveragePooling1D",
          "GlobalAveragePooling2D",
          "GlobalAveragePooling3D",
          "GlobalMaxPooling1D",
          "GlobalMaxPooling2D",
          "GlobalMaxPooling3D",
          "InputLayer",
          "LeakyReLU",
          "Masking",
          "MaxPooling1D",
          "MaxPooling2D",
          "MaxPooling3D",
          "Maximum",
          "Minimum",
          "Multiply",
          "Permute",
          "ReLU",
          "RepeatVector",
          "Reshape",
          "Softmax",
          "SpatialDropout1D",
          "SpatialDropout2D",
          "SpatialDropout3D",
          "Subtract",
          "UpSampling1D",
          "UpSampling2D",
          "UpSampling3D",
          "ZeroPadding1D",
          "ZeroPadding2D",
          "ZeroPadding3D");

  /**
   * The Keras layer class whose call a node is, as {@code Dense} for the node of {@code
   * Ltensorflow/keras/layers/Dense/__call__} (wala/ML#1123).
   *
   * @param node The node.
   * @return The layer's class name, or {@code null} when the node is no Keras layer's call.
   */
  private static String kerasLayerName(CGNode node) {
    String type = node.getMethod().getDeclaringClass().getName().toString();
    if (!type.startsWith(KERAS_LAYERS_PREFIX)) return null;
    String rest = type.substring(KERAS_LAYERS_PREFIX.length());
    int slash = rest.indexOf('/');
    if (slash < 0) return null;
    // A call node only, `Dense/__call__` or `Dense/call`: the walk also meets constructors
    // (`class/Dense`) and inputs, which hold no weights of their own.
    String method = rest.substring(slash + 1);
    return method.equals(PythonTypes.CALLABLE_METHOD_NAME)
            || method.equals(PythonTypes.CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)
        ? rest.substring(0, slash)
        : null;
  }

  /**
   * Infers the shapes of the model's weights by traversing the computation graph backwards from the
   * model's outputs, or, for a {@code Sequential} model, from its layer list (wala/ML#832).
   *
   * @param builder The propagation call graph builder.
   * @param modelSource The source points-to set variable representing the model.
   * @return A set of weight shapes, or {@code null} when a layer the walk reaches holds weights it
   *     does not model or a {@code Sequential} model's layer list cannot give them.
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
      // Only a `Dense` layer's weights are modeled: the convolution and recurrent calls extend
      // `DenseCall`, but their `units` are no dense kernel's width (wala/ML#1123). A layer known to
      // hold no weights contributes none, and any other Keras layer holds weights this does not
      // know, so the model's weights are then tensors of unknown shape.
      String layer = kerasLayerName(node);
      if (layer != null
          && (generator == null || generator.getClass() != DenseCall.class)
          && !WEIGHTLESS_LAYERS.contains(layer)) return null;
      if (generator != null && generator.getClass() == DenseCall.class) {
        DenseCall denseCall = (DenseCall) generator;
        int valNum =
            denseCall.getArgumentValueNumber(
                builder,
                DenseCall.Parameters.INPUTS.getIndex(),
                DenseCall.Parameters.INPUTS.getName(),
                false);
        Set<List<Dimension<?>>> inputShapes = denseCall.getShapes(builder, valNum);
        // An unresolvable input skips only this node's kernel/bias contribution; the trace-back
        // below still runs, so one unmodeled layer input does not stop the upstream walk
        // (wala/ML#670).
        if (inputShapes != null) {
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

    // A `Sequential` model has no `outputs` to walk back from: its weights are those of the layers
    // in its list (wala/ML#832).
    if (this.getSource() != null)
      for (InstanceKey modelIK :
          builder.getPointerAnalysis().getPointsToSet(this.getSource().getPointerKey())) {
        AllocationSiteInNode model = getAllocationSiteInNode(modelIK);
        if (model == null
            || !model
                .concreteType()
                .getReference()
                .equals(TensorFlowTypes.SEQUENTIAL.getDeclaringClass())) continue;
        Set<List<Dimension<?>>> layerWeights = SequentialCall.getWeightShapes(builder, model);
        // Weights the list cannot give are still tensors, of shapes this does not know.
        if (layerWeights == null) return null;
        weightShapes.addAll(layerWeights);
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
