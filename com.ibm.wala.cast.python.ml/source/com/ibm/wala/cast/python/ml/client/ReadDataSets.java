package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.ReadDataSets.Parameters.DTYPE;
import static com.ibm.wala.cast.python.ml.client.ReadDataSets.Parameters.RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorType.mnistInput;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/** A representation of the `read_data_sets()` function in TensorFlow. */
public class ReadDataSets extends Ones {

  private static final TensorType MNIST_INPUT = mnistInput();

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
    return RESHAPE.ordinal();
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE.ordinal();
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    List<Dimension<?>> dims = MNIST_INPUT.getDims();
    return Collections.singleton(dims);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    String cellType = MNIST_INPUT.getCellType().toUpperCase();
    DType dType = DType.valueOf(cellType);
    return EnumSet.of(dType);
  }
}
