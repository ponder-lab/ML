package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.logging.Logger;

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
          ret.add(List.of(DynamicDim.INSTANCE, this.categoricalClassAxis(builder)));
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

  private static final Logger LOGGER = Logger.getLogger(FlowFromDirectoryGenerator.class.getName());

  /**
   * The consumer-constrained class axis per anchor node, per builder (wala/ML#920): the recognizer
   * scans the whole call graph, so it runs once per generator anchor; a sentinel is entered before
   * computing so a re-entry through the shape machinery reads "no constraint" instead of recursing
   * (the wala/ML#923 hazard).
   */
  private static final Map<PropagationCallGraphBuilder, Map<CGNode, Optional<Dimension<?>>>>
      CLASS_AXIS_CACHE = Collections.synchronizedMap(new WeakHashMap<>());

  /**
   * The class axis of {@code class_mode="categorical"} labels (wala/ML#920). The class count is the
   * directory's subfolder count, which no forward chase reads, so the axis is {@link UnresolvedDim}
   * unless a shape-constrained consumer fixes it: {@code tf.keras.losses.CategoricalCrossentropy}
   * requires {@code y_true} and {@code y_pred} to have the same shape ({@code Loss.__call__} raises
   * on a mismatch), so labels reaching such a call whose predictions resolve to rank 2 with one
   * concrete last axis have that axis. The recognizer names the class, not the family: the sparse
   * variant takes rank-1 integer labels and constrains no class axis. Membership is the identity of
   * THIS generator's labels allocation in the call's {@code y_true} points-to set, so a second
   * categorical generator in the same program inherits nothing. Calls that disagree, a prediction
   * whose width does not resolve, or no reaching call at all leave the axis unresolved.
   *
   * <p>An extent recovered this way is derived from the program being well-formed rather than
   * computed forward, and it is not independent evidence for the prediction width it was derived
   * from; the two rows agree by construction afterwards.
   *
   * <p><b>Why a prediction that depends on these labels cannot derive a width from them.</b> Two
   * facts together: the width is read from {@code y_pred}'s LAST axis only, and this generator's
   * own class axis reads {@link UnresolvedDim} while it is being computed, because the memo below
   * enters its sentinel before computing (this path; the rank-predicate path of wala/ML#923 has no
   * such sentinel and recurses). So a {@code y_pred} whose last axis came from these labels
   * supplies no width and the recognizer declines, and a {@code y_pred} that depends on these
   * labels only through the batch (a {@code Sequential} over the images) takes its width from its
   * own layers, which is correct whether or not the cycle resolves. Reading the width from any
   * other position, or from a fold over more than the last axis, would reopen the question, and no
   * test would go red for it.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @return The consumer-fixed class extent, or {@link UnresolvedDim#INSTANCE}.
   */
  private Dimension<?> categoricalClassAxis(PropagationCallGraphBuilder builder) {
    CGNode anchor = this.getNode();
    if (anchor == null) return UnresolvedDim.INSTANCE;
    Map<CGNode, Optional<Dimension<?>>> cache =
        CLASS_AXIS_CACHE.computeIfAbsent(
            builder, b -> Collections.synchronizedMap(new HashMap<>()));
    Optional<Dimension<?>> memo = cache.get(anchor);
    if (memo == null) {
      cache.put(anchor, Optional.empty());
      Dimension<?> computed = this.consumerConstrainedClassAxis(builder, anchor);
      memo = Optional.ofNullable(computed);
      cache.put(anchor, memo);
    }
    return memo.orElse(UnresolvedDim.INSTANCE);
  }

  /**
   * Finds the {@code CategoricalCrossentropy} calls this generator's labels reach and reads the
   * class extent off their predictions.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param anchor The {@code flow_from_directory} frame this generator is anchored at.
   * @return The agreed concrete class extent, or {@code null} when none is established.
   */
  private Dimension<?> consumerConstrainedClassAxis(
      PropagationCallGraphBuilder builder, CGNode anchor) {
    OrdinalSet<InstanceKey> labelsKeys = this.labelsAllocation(builder, anchor);
    LOGGER.fine(
        () -> "Categorical labels allocation at " + anchor + ": " + labelsKeys + " (wala/ML#920).");
    if (labelsKeys == null || labelsKeys.isEmpty()) return null;
    // The allocation identifies ONE generator only if the frame that made it is reached by a single
    // chain of call sites. A `flow_from_directory` inside a wrapper function called from several
    // places gives the labels helper one context for every caller (the call string keeps the two
    // innermost sites, and both are inside the wrapper), so one allocation serves several
    // generators, possibly of different class counts, and a width fixed through one caller's loss
    // would be read for every caller. Measured: two wrapped generators of five and ten classes, one
    // loss, and the five-class labels read ten. The walk starts at the ALLOCATING node, since that
    // is where the sharing happens, and passes through this frame on the way to the root. The call
    // graph is complete when a generator runs (generators are seeded after `makeCallGraph` returns,
    // the same stability the shape memo relies on), so a predecessor count read here is final.
    for (InstanceKey labelsKey : labelsKeys) {
      if (!(labelsKey instanceof AllocationSiteInNode)) continue;
      CGNode allocating = ((AllocationSiteInNode) labelsKey).getNode();
      if (!reachedThroughOneSite(builder, allocating)) {
        LOGGER.fine(
            () ->
                "Labels allocation in "
                    + allocating
                    + " is reached through more than one call site; it does not identify one"
                    + " generator, declining (wala/ML#920).");
        return null;
      }
    }

    Integer agreed = null;
    boolean anyCall = false;
    for (CGNode caller : builder.getCallGraph()) {
      if (caller.getIR() == null || caller.getDU() == null) continue;
      if (!(caller.getMethod() instanceof AstMethod)) continue;
      for (Iterator<SSAInstruction> it = caller.getIR().iterateAllInstructions(); it.hasNext(); ) {
        SSAInstruction inst = it.next();
        if (!(inst instanceof PythonInvokeInstruction)) continue;
        PythonInvokeInstruction call = (PythonInvokeInstruction) inst;
        if (call.getNumberOfUses() < 3
            || !this.isCategoricalCrossentropyCall(builder, caller, call)) continue;
        int yTrueVn = call.getUse("y_true");
        if (yTrueVn <= 0 && call.getNumberOfPositionalParameters() >= 2) yTrueVn = call.getUse(1);
        int yPredVn = call.getUse("y_pred");
        if (yPredVn <= 0 && call.getNumberOfPositionalParameters() >= 3) yPredVn = call.getUse(2);
        if (yTrueVn <= 0 || yPredVn <= 0) continue;
        PointerKey yTrueKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, yTrueVn);
        OrdinalSet<InstanceKey> yTruePts = builder.getPointerAnalysis().getPointsToSet(yTrueKey);
        boolean reaches = false;
        if (yTruePts != null)
          for (InstanceKey ik : yTruePts) if (labelsKeys.contains(ik)) reaches = true;
        final int fYTrue = yTrueVn;
        final int fYPred = yPredVn;
        final boolean fReaches = reaches;
        LOGGER.fine(
            () ->
                "CategoricalCrossentropy call "
                    + call
                    + " in "
                    + caller
                    + ": y_true vn "
                    + fYTrue
                    + " points to "
                    + yTruePts
                    + ", reaches these labels: "
                    + fReaches
                    + ", y_pred vn "
                    + fYPred
                    + " (wala/ML#920).");
        if (!reaches) continue;
        anyCall = true;
        Set<List<Dimension<?>>> predShapes = this.getShapes(builder, caller, yPredVn);
        LOGGER.fine(
            () -> "Predictions beside the labels resolve to " + predShapes + " (wala/ML#920).");
        Integer width = classWidth(predShapes);
        if (width == null) return null; // Unreadable or ambiguous predictions: no constraint.
        if (agreed != null && !agreed.equals(width)) {
          final Integer a = agreed;
          LOGGER.fine(
              () -> "Two loss calls fix different widths (" + a + ", " + width + "); declining.");
          return null;
        }
        agreed = width;
      }
    }
    if (!anyCall) LOGGER.fine(() -> "No CategoricalCrossentropy call reached by these labels.");
    return agreed == null ? null : new NumericDim(agreed);
  }

  /** Bound on the caller walk in {@link #reachedThroughOneSite}; a real chain is a few frames. */
  private static final int SITE_CHAIN_DEPTH_CAP = 32;

  /**
   * Whether the given frame is reached from the program's roots through exactly one call site at
   * every level, so that an allocation made in it belongs to one program construct rather than to
   * every caller of some wrapper (wala/ML#920).
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param node The frame to test.
   * @return {@code true} iff each frame on the way up to a root has exactly one incoming call site.
   */
  private static boolean reachedThroughOneSite(PropagationCallGraphBuilder builder, CGNode node) {
    CGNode current = node;
    for (int depth = 0; depth < SITE_CHAIN_DEPTH_CAP; depth++) {
      CGNode onlyCaller = null;
      int sites = 0;
      for (Iterator<CGNode> preds = builder.getCallGraph().getPredNodes(current);
          preds.hasNext(); ) {
        CGNode pred = preds.next();
        for (Iterator<?> it = builder.getCallGraph().getPossibleSites(pred, current);
            it.hasNext(); ) {
          it.next();
          sites++;
        }
        onlyCaller = pred;
      }
      if (sites == 0) return builder.getCallGraph().getFakeRootNode().equals(current);
      if (sites > 1) return false;
      current = onlyCaller;
    }
    return false;
  }

  /**
   * The single concrete last axis shared by every rank-2 prediction shape, or {@code null}.
   *
   * @param shapes The predictions' shapes.
   * @return The agreed width, or {@code null} when a member is not rank 2 with a numeric last axis
   *     or members disagree.
   */
  private static Integer classWidth(Set<List<Dimension<?>>> shapes) {
    if (shapes == null || shapes.isEmpty()) return null;
    Integer width = null;
    for (List<Dimension<?>> shape : shapes) {
      if (shape == null || shape.size() != 2) return null;
      Dimension<?> last = shape.get(1);
      if (!(last instanceof NumericDim)) return null;
      Integer value = ((NumericDim) last).value();
      if (value == null || (width != null && !width.equals(value))) return null;
      width = value;
    }
    return width;
  }

  /**
   * Whether the call's callee is an instance of {@code tf.keras.losses.CategoricalCrossentropy}.
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param caller The node containing the call.
   * @param call The call.
   * @return {@code true} iff the callee's points-to set holds such an instance.
   */
  private boolean isCategoricalCrossentropyCall(
      PropagationCallGraphBuilder builder, CGNode caller, PythonInvokeInstruction call) {
    PointerKey calleeKey =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, call.getUse(0));
    OrdinalSet<InstanceKey> calleePts = builder.getPointerAnalysis().getPointsToSet(calleeKey);
    if (calleePts == null) return false;
    for (InstanceKey ik : calleePts) {
      // A callee may also be a builtin function (a ConcreteTypeKey the allocation-site extractor
      // rejects); only an allocated instance can be the loss object.
      if (!(ik instanceof AllocationSiteInNode)) continue;
      AllocationSiteInNode asin = (AllocationSiteInNode) ik;
      if (asin.concreteType()
          .getReference()
          .equals(TensorFlowTypes.KERAS_CATEGORICAL_CROSSENTROPY_TYPE)) return true;
    }
    return false;
  }

  /**
   * The points-to set of this generator's labels position: the result of the summary-internal
   * helper call in the anchor frame that allocates the labels marker (wala/ML#834).
   *
   * @param builder The propagation call graph builder used for the analysis.
   * @param anchor The {@code flow_from_directory} frame.
   * @return The labels value's points-to set, or {@code null} when the helper call is not found.
   */
  private OrdinalSet<InstanceKey> labelsAllocation(
      PropagationCallGraphBuilder builder, CGNode anchor) {
    if (anchor.getIR() == null) return null;
    for (Iterator<SSAInstruction> it = anchor.getIR().iterateAllInstructions(); it.hasNext(); ) {
      SSAInstruction inst = it.next();
      if (!(inst instanceof SSAAbstractInvokeInstruction)) continue;
      SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) inst;
      if (!call.getDeclaredTarget()
          .getDeclaringClass()
          .equals(TensorFlowTypes.DIRECTORY_ITERATOR_LABELS_TYPE)) continue;
      if (!call.hasDef()) continue;
      PointerKey key =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(anchor, call.getDef());
      return builder.getPointerAnalysis().getPointsToSet(key);
    }
    return null;
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
