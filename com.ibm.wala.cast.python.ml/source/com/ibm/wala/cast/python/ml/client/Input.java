package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static java.util.Collections.emptySet;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for tensors created by the `Input()` function in TensorFlow/Keras.
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Input">TensorFlow Input() API</a>.
 */
public class Input extends TensorGenerator {

  private static final Logger LOGGER = getLogger(Input.class.getName());

  private static final String FUNCTION_NAME = "tf.keras.Input()";

  private static final int SHAPE_PARAMETER_POSITION = -1;

  private static final int DTYPE_PARAMETER_POSITION = 2;

  public Input(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    LOGGER.info(
        "No dtype specified for source: " + source + ". Using default dtype of: " + FLOAT32 + " .");

    // Use the default dtype of float32.
    return EnumSet.of(FLOAT32);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return emptySet();
  }

  @Override
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = super.getShapes(builder);
    Set<List<Dimension<?>>> newShapes = HashSetFactory.make();
    for (List<Dimension<?>> shape : shapes) {
      List<Dimension<?>> newShape = new ArrayList<>();
      // Prepend unknown dimension (None) for batch size.
      newShape.add(null);
      newShape.addAll(shape);
      newShapes.add(newShape);
    }
    return newShapes;
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }

  @Override
  protected int getShapeArgumentValueNumber() {
    return this.getNode().getIR().getParameter(0);
  }

  @Override
  protected int getDTypeArgumentValueNumber() {
    return this.getNode().getIR().getParameter(3);
  }

  @Override
  protected int getShapeParameterPosition() {
    return SHAPE_PARAMETER_POSITION;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }
}
