package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
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
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * A generator for {@code tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory}.
 *
 * <p>A {@code DirectoryIterator} yields {@code (x, y)} batch tuples: {@code x} is a rank-4 {@code
 * float32} image batch shaped {@code (batch, target_size[0], target_size[1], channels)} and {@code
 * y} is a label array whose rank follows {@code class_mode}. The per-position typing is exposed
 * through {@link TupleElementProvider} (index 0 = images, index 1 = labels), matching the tuple the
 * summary materializes in {@code tensorflow.xml}; the aggregate {@link #getDefaultShapeResult}
 * keeps the dataset convention of reporting the union over both positions, expressing a
 * partially-resolved union (one position ⊤, the other known) as a partial rather than collapsing
 * the whole batch to ⊤ (wala/ML#718). See wala/ML#830.
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
public class FlowFromDirectoryGenerator extends DatasetGenerator {

  /** The images position (field 0) of the {@code (x, y)} batch tuple. */
  static final int IMAGES_INDEX = 0;

  /** The labels position (field 1) of the {@code (x, y)} batch tuple. */
  static final int LABELS_INDEX = 1;

  /** The position of {@code target_size} among the call's arguments, {@code self} excluded. */
  private static final int TARGET_SIZE_POSITION = 1;

  /** The position of {@code color_mode} among the call's arguments, {@code self} excluded. */
  private static final int COLOR_MODE_POSITION = 2;

  /** The position of {@code class_mode} among the call's arguments, {@code self} excluded. */
  private static final int CLASS_MODE_POSITION = 4;

  public FlowFromDirectoryGenerator(PointsToSetVariable source) {
    super(source);
  }

  public FlowFromDirectoryGenerator(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * {@inheritDoc}
   *
   * @implNote The union over both tuple positions, kept partial when only one position resolves: an
   *     unparseable {@code target_size} makes the images position ⊤ without discarding the labels
   *     shapes (and vice versa), per the wala/ML#718 pairing rule.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> imagesShapes = this.getImagesShapes(builder);
    Set<List<Dimension<?>>> labelsShapes = this.getLabelsShapes(builder);

    if (imagesShapes == null && labelsShapes == null) return ShapeResult.unknown();

    Set<List<Dimension<?>>> members = HashSetFactory.make();
    if (imagesShapes != null) members.addAll(imagesShapes);
    if (labelsShapes != null) members.addAll(labelsShapes);

    return new ShapeResult(members, imagesShapes == null || labelsShapes == null);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Both positions are float32: images through the iterator's `dtype` parameter (default
    // "float32"), labels through the same cast under the common `class_mode`s.
    return EnumSet.of(FLOAT32);
  }

  /**
   * Computes the possible shapes of the images position (field 0) of the batch tuple: {@code
   * (batch, target_size[0], target_size[1], channels)}, unioned over every resolved {@code
   * target_size} and {@code color_mode} candidate.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The possible image-batch shapes, or {@code null} (⊤) when {@code target_size} is
   *     present but not statically resolvable.
   */
  private Set<List<Dimension<?>>> getImagesShapes(PropagationCallGraphBuilder builder) {
    // Determine the spatial extents from `target_size`, defaulting to (256, 256). Every resolved
    // candidate contributes a member; collapsing a plural set to one arbitrary member would drop
    // the rest by hash order.
    Set<List<Dimension<?>>> targetSizes = null;
    OrdinalSet<InstanceKey> targetSizePts =
        this.getArgumentPointsToSet(builder, TARGET_SIZE_POSITION, "target_size");
    if (targetSizePts != null && !targetSizePts.isEmpty()) {
      targetSizes = this.getShapesFromShapeArgument(builder, targetSizePts);
      // Soundness: when `target_size` is present but unparseable, the runtime value could be
      // anything — falling back to the (256, 256) default would falsely claim a fixed shape.
      if (targetSizes == null) return null;
    }
    if (targetSizes == null || targetSizes.isEmpty())
      targetSizes = Collections.singleton(List.of(new NumericDim(256), new NumericDim(256)));

    Set<Dimension<?>> channels = this.getChannelsDims(builder);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> targetSize : targetSizes)
      for (Dimension<?> channel : channels) {
        List<Dimension<?>> imageShape = new ArrayList<>();
        // The batch axis varies at run time (short final batch); see the class Javadoc for why
        // this must not be the `batch_size` literal.
        imageShape.add(DynamicDim.INSTANCE);
        imageShape.addAll(targetSize);
        imageShape.add(channel);
        ret.add(imageShape);
      }
    return ret;
  }

  /**
   * Computes the channel-axis candidates from the {@code color_mode} argument: {@code "rgb"} (the
   * default, also taken for {@code None}) ↦ 3, {@code "grayscale"} ↦ 1, {@code "rgba"} ↦ 4. Every
   * resolved constant contributes a candidate.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The channel-dimension candidates; a candidate is {@link UnresolvedDim} when a {@code
   *     color_mode} value is not a recognized constant (the channel count is a fixed run-time
   *     integer the analysis could not compute).
   */
  private Set<Dimension<?>> getChannelsDims(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> colorModePts =
        this.getArgumentPointsToSet(builder, COLOR_MODE_POSITION, "color_mode");

    if (colorModePts == null || colorModePts.isEmpty())
      // Unspecified; the API default is "rgb".
      return Collections.singleton(new NumericDim(3));

    Set<Object> colorModes = getConstantValues(colorModePts, true);
    if (colorModes == null || colorModes.isEmpty())
      return Collections.singleton(UnresolvedDim.INSTANCE);

    Set<Dimension<?>> ret = HashSetFactory.make();
    for (Object colorMode : colorModes) {
      if (colorMode == null) {
        ret.add(new NumericDim(3)); // `None` falls back to the "rgb" default.
        continue;
      }
      switch (colorMode.toString().toLowerCase(Locale.ROOT)) {
        case "rgb":
          ret.add(new NumericDim(3));
          break;
        case "grayscale":
          ret.add(new NumericDim(1));
          break;
        case "rgba":
          ret.add(new NumericDim(4));
          break;
        default:
          ret.add(UnresolvedDim.INSTANCE);
      }
    }
    return ret;
  }

  /**
   * Computes the possible shapes of the labels position (field 1) of the batch tuple, following
   * {@code class_mode}: {@code "categorical"} (the default) yields rank-2 {@code (batch,
   * num_classes)}, {@code "sparse"} and {@code "binary"} yield rank-1 {@code (batch,)}, {@code
   * "input"} yields the images shapes, and {@code None} yields no label position at all (⊥ here;
   * {@link #yieldsTuple} answers {@code false} there).
   *
   * <p>{@code num_classes} is the directory's class-subfolder count: fixed for a given run but
   * unknown statically — {@link UnresolvedDim} (wala/ML#721; previously {@code DynamicDim},
   * wala/ML#545).
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The possible label-batch shapes; {@code null} (⊤) when {@code class_mode} is present
   *     but not statically resolvable, since the rank itself then depends on the unresolved mode.
   */
  private Set<List<Dimension<?>>> getLabelsShapes(PropagationCallGraphBuilder builder) {
    Set<Object> classModes = this.getClassModes(builder);
    if (classModes == null) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Object classMode : classModes) {
      if (classMode == null) continue; // `None`: the iterator yields bare images, no labels.
      switch (classMode.toString().toLowerCase(Locale.ROOT)) {
        case "categorical":
          ret.add(List.of(DynamicDim.INSTANCE, UnresolvedDim.INSTANCE));
          break;
        case "sparse":
        case "binary":
          ret.add(List.of(DynamicDim.INSTANCE));
          break;
        case "input":
          Set<List<Dimension<?>>> imagesShapes = this.getImagesShapes(builder);
          if (imagesShapes == null) return null;
          ret.addAll(imagesShapes);
          break;
        default:
          // An unrecognized mode leaves the label rank itself unknown.
          return null;
      }
    }
    return ret;
  }

  /**
   * Resolves the {@code class_mode} argument's constant values, with the API default.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The constant values ({@code null} elements encode Python {@code None}); the singleton
   *     {@code "categorical"} when the argument is absent; {@code null} when the argument is
   *     present but not statically resolvable.
   */
  private Set<Object> getClassModes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> classModePts =
        this.getArgumentPointsToSet(builder, CLASS_MODE_POSITION, "class_mode");

    if (classModePts == null || classModePts.isEmpty())
      // Unspecified; the API default is "categorical".
      return Collections.singleton("categorical");

    return getConstantValues(classModePts, true);
  }

  /**
   * {@inheritDoc}
   *
   * @implNote A {@code DirectoryIterator} yields {@code (x, y)} under every {@code class_mode} but
   *     {@code None}, where it yields bare {@code x}. An unresolvable {@code class_mode} keeps the
   *     tuple claim: every string mode yields the tuple, so the claim is wrong only where the
   *     unresolved value is {@code None} itself.
   */
  @Override
  public boolean yieldsTuple(PropagationCallGraphBuilder builder) {
    Set<Object> classModes = this.getClassModes(builder);
    if (classModes == null || classModes.isEmpty()) return true;
    return !classModes.stream().allMatch(m -> m == null);
  }

  @Override
  public Set<List<Dimension<?>>> getShapesForIndex(PropagationCallGraphBuilder builder, int index) {
    switch (index) {
      case IMAGES_INDEX:
        return this.getImagesShapes(builder);
      case LABELS_INDEX:
        return this.getLabelsShapes(builder);
      default:
        // The tuple has exactly two positions; any other index is a runtime error, not a tensor
        // (⊥ — paired with the empty dtype set below per the lattice rule).
        return Collections.emptySet();
    }
  }

  @Override
  public Set<DType> getDTypesForIndex(PropagationCallGraphBuilder builder, int index) {
    if (index == IMAGES_INDEX || index == LABELS_INDEX) return this.getDefaultDTypes(builder);
    return EnumSet.noneOf(DType.class);
  }
}
