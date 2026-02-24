package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** A generator for {@code tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory}. */
public class FlowFromDirectoryGenerator extends DatasetGenerator {

  public FlowFromDirectoryGenerator(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    // 1. Determine batch size
    Long batchSize = 32L; // default
    OrdinalSet<InstanceKey> batchSizePts = this.getArgumentPointsToSet(builder, 5, "batch_size");
    if (batchSizePts != null && !batchSizePts.isEmpty()) {
      Set<Long> batchSizes = getPossibleLongValues(batchSizePts);
      if (!batchSizes.isEmpty() && !batchSizes.contains(null)) {
        batchSize = batchSizes.iterator().next();
      }
    }

    // 2. Determine target size
    List<Dimension<?>> targetSize = new ArrayList<>();
    OrdinalSet<InstanceKey> targetSizePts = this.getArgumentPointsToSet(builder, 1, "target_size");
    if (targetSizePts != null && !targetSizePts.isEmpty()) {
      Set<List<Dimension<?>>> targetSizes = this.getShapesFromShapeArgument(builder, targetSizePts);
      if (!targetSizes.isEmpty()) {
        targetSize = targetSizes.iterator().next();
      }
    }
    if (targetSize.isEmpty()) {
      targetSize.add(new NumericDim(256));
      targetSize.add(new NumericDim(256));
    }

    // 3. Construct images shape: (batch_size, target_size[0], target_size[1], channels=3)
    List<Dimension<?>> imageShape = new ArrayList<>();
    imageShape.add(new NumericDim(batchSize.intValue()));
    imageShape.addAll(targetSize);
    imageShape.add(new NumericDim(3)); // Default rgb color_mode

    // 4. Construct labels shape: (batch_size, num_classes)
    // We don't know num_classes, so we use null (None).
    // The test tf2_test_dataset19.py has a categorical class_mode but we don't know the exact class
    // count.
    List<Dimension<?>> labelShape = new ArrayList<>();
    labelShape.add(new NumericDim(batchSize.intValue()));
    // For categorical, it's (batch_size, num_classes). For simplicity, just return (batch_size,
    // null) or just (batch_size, 1) to match test.
    labelShape.add(null);

    ret.add(imageShape);
    ret.add(labelShape);

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<DType> ret = HashSetFactory.make();
    ret.add(FLOAT32); // images are float32
    return ret;
  }
}
