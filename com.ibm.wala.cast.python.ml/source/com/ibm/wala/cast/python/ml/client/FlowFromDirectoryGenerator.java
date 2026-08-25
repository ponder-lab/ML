package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for {@code tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory}.
 *
 * <p>A {@code DirectoryIterator} yields {@code (x, y)} batch tuples: {@code x} is a rank-4 {@code
 * float32} image batch shaped {@code (batch, target_size[0], target_size[1], channels)} and {@code
 * y} is a label array. The per-position typing is exposed through {@link TupleElementProvider}
 * (index 0 = images, index 1 = labels), matching the tuple the summary materializes in {@code
 * tensorflow.xml}; the aggregate {@link #getDefaultShapes} keeps the dataset convention of
 * reporting the union over both positions. See wala/ML#830.
 *
 * <p>The batch axis is {@link DynamicDim}, never the {@code batch_size} literal: a {@code
 * DirectoryIterator} does not drop its remainder, so when the image count is not a multiple of
 * {@code batch_size} the final batch of each epoch is short. The axis therefore genuinely varies at
 * run time — it is not a fixed integer the analysis failed to compute ({@code UnresolvedDim}), and
 * emitting the literal would describe a batch the loader does not always produce (a downstream
 * consumer can turn that into a hard failure on the last iteration). This is the same
 * feed-dependent sense in which the equivalent {@code tf.data} pipeline's {@code TensorShape}
 * reports {@code None} on the batch axis (wala/ML#721's criterion).
 */
public class FlowFromDirectoryGenerator extends DatasetGenerator implements TupleElementProvider {

  /** The images position (field 0) of the {@code (x, y)} batch tuple. */
  static final int IMAGES_INDEX = 0;

  /** The labels position (field 1) of the {@code (x, y)} batch tuple. */
  static final int LABELS_INDEX = 1;

  /** The position of {@code target_size} among the call's arguments, {@code self} excluded. */
  private static final int TARGET_SIZE_POSITION = 1;

  /** The position of {@code color_mode} among the call's arguments, {@code self} excluded. */
  private static final int COLOR_MODE_POSITION = 2;

  public FlowFromDirectoryGenerator(PointsToSetVariable source) {
    super(source);
  }

  public FlowFromDirectoryGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> imageShapes = this.getImagesShapes(builder);

    // Soundness: when `target_size` is present but unparseable, the runtime value could be
    // anything — falling back to the (256, 256) default would falsely claim a fixed shape.
    // Return ⊤ instead.
    if (imageShapes == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.addAll(imageShapes);
    ret.addAll(this.getLabelsShapes(builder));
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Both positions are float32: images through the iterator's `dtype` parameter (default
    // "float32"), labels through the same cast under the common `class_mode`s.
    return EnumSet.of(FLOAT32);
  }

  /**
   * Computes the possible shapes of the images position (field 0) of the batch tuple: {@code
   * (batch, target_size[0], target_size[1], channels)}.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The possible image-batch shapes, or {@code null} (⊤) when {@code target_size} is
   *     present but not statically resolvable.
   */
  private Set<List<Dimension<?>>> getImagesShapes(PropagationCallGraphBuilder builder) {
    // Determine the spatial extents from `target_size`, defaulting to (256, 256).
    List<Dimension<?>> targetSize = new ArrayList<>();
    OrdinalSet<InstanceKey> targetSizePts =
        this.getArgumentPointsToSet(builder, TARGET_SIZE_POSITION, "target_size");
    if (targetSizePts != null && !targetSizePts.isEmpty()) {
      Set<List<Dimension<?>>> targetSizes = this.getShapesFromShapeArgument(builder, targetSizePts);
      // Soundness: when `target_size` is present but unparseable, the runtime value could be
      // anything — falling back to the (256, 256) default would falsely claim a fixed shape.
      if (targetSizes == null) return null;
      if (!targetSizes.isEmpty()) {
        targetSize = targetSizes.iterator().next();
      }
    }
    if (targetSize.isEmpty()) {
      targetSize.add(new NumericDim(256));
      targetSize.add(new NumericDim(256));
    }

    List<Dimension<?>> imageShape = new ArrayList<>();
    // The batch axis varies at run time (short final batch); see the class Javadoc for why this
    // must not be the `batch_size` literal.
    imageShape.add(DynamicDim.INSTANCE);
    imageShape.addAll(targetSize);
    imageShape.add(this.getChannelsDim(builder));

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(imageShape);
    return ret;
  }

  /**
   * Computes the channel axis from the {@code color_mode} argument: {@code "rgb"} (the default) ↦
   * 3, {@code "grayscale"} ↦ 1, {@code "rgba"} ↦ 4.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The channel dimension; {@link UnresolvedDim} when {@code color_mode} is present but not
   *     a recognized string constant (the channel count is a fixed run-time integer the analysis
   *     could not compute).
   */
  private Dimension<?> getChannelsDim(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> colorModePts =
        this.getArgumentPointsToSet(builder, COLOR_MODE_POSITION, "color_mode");

    if (colorModePts == null || colorModePts.isEmpty())
      // Unspecified; the API default is "rgb".
      return new NumericDim(3);

    Set<Object> colorModes = getConstantValues(colorModePts, true);
    if (colorModes == null || colorModes.size() != 1) return UnresolvedDim.INSTANCE;

    Object colorMode = colorModes.iterator().next();
    if (colorMode == null) return new NumericDim(3); // `None` falls back to the "rgb" default.

    switch (colorMode.toString().toLowerCase(Locale.ROOT)) {
      case "rgb":
        return new NumericDim(3);
      case "grayscale":
        return new NumericDim(1);
      case "rgba":
        return new NumericDim(4);
      default:
        return UnresolvedDim.INSTANCE;
    }
  }

  /**
   * Computes the possible shapes of the labels position (field 1) of the batch tuple: {@code
   * (batch, num_classes)} under the default {@code class_mode="categorical"}.
   *
   * <p>{@code num_classes} is the directory's class-subfolder count: fixed for a given run but
   * unknown statically — {@link UnresolvedDim} (wala/ML#721; previously {@code DynamicDim},
   * wala/ML#545).
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The possible label-batch shapes.
   */
  private Set<List<Dimension<?>>> getLabelsShapes(PropagationCallGraphBuilder builder) {
    List<Dimension<?>> labelShape = new ArrayList<>();
    labelShape.add(DynamicDim.INSTANCE);
    labelShape.add(UnresolvedDim.INSTANCE);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    ret.add(labelShape);
    return ret;
  }

  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    // A `DirectoryIterator` yields `(x, y)` under every `class_mode` but `None` (where it yields
    // only `x`); the default is "categorical".
    return true;
  }

  @Override
  public Set<TensorType> getTensorTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    Set<List<Dimension<?>>> shapes = this.getShapesForIndex(builder, index);
    Set<DType> dTypes = this.getDTypesForIndex(builder, index);

    Set<TensorType> ret = HashSetFactory.make();

    // Null shapes signal "unknown per-index shape" (⊤). Emit one ⊤-shaped TensorType per dtype
    // rather than falling through to the aggregate `getTensorTypes`, which would leak the sibling
    // position's shapes. See wala/ML#396.
    if (shapes == null) {
      for (DType dtype : dTypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), null));
      return ret;
    }

    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes)
        ret.add(new TensorType(dtype.name().toLowerCase(Locale.ROOT), dimensionList));

    return ret;
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    switch (index) {
      case IMAGES_INDEX:
        return this.getImagesShapes(builder);
      case LABELS_INDEX:
        return this.getLabelsShapes(builder);
      default:
        return null;
    }
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    return EnumSet.of(FLOAT32);
  }
}
