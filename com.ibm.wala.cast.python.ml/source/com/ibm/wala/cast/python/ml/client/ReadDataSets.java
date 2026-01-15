package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/** A representation of the `read_data_sets()` function in TensorFlow. */
public class ReadDataSets extends TensorGenerator {

  protected enum Parameters {
    TRAIN_DIR,
    FAKE_DATA,
    ONE_HOT,
    DTYPE,
    RESHAPE,
    VALIDATION_SIZE,
    SEED,
    SOURCE_URL
  }

  public ReadDataSets(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.ordinal();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // read_data_sets returns a container object, not a single tensor.
    // The analysis of its fields (train.images, etc.) would need to be handled separately.
    return Collections.emptySet();
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Default dtype for MNIST images is float32.
    return EnumSet.of(FLOAT32);
  }
}
