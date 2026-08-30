package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

/**
 * A generator for the result of calling a {@code tf.keras.Sequential} instance, which composes the
 * model's forward chain by folding its layer list (<a
 * href="https://github.com/wala/ML/issues/832">wala/ML#832</a>).
 *
 * <p>A {@code Sequential} keeps in its layer LIST what a functional model keeps in its {@code
 * inputs} and {@code outputs} arguments, so the machinery that walks a functional model backwards
 * from its outputs has nothing to anchor on here and the call's shape floored at ⊤. The fold below
 * supplies the missing piece: each layer's own transform, applied in list order to the running
 * shape, starting from the shapes of the value the model was called with.
 *
 * <p>The layers are constructed but never called, so no layer has a {@code __call__} node of its
 * own to dispatch on. The fold names each layer's call type from its instance type and anchors the
 * generator at this model call, supplying the receiver and the input directly ({@link
 * TensorGenerator.ComposedArguments}). That keeps every layer's transform in the one place it
 * already lives rather than restating it here.
 *
 * <p><b>The fold refuses rather than guesses.</b> A layer the dispatch does not know, a list whose
 * indices are not the contiguous run a literal list produces, or any layer whose transform declines
 * abandons the whole composition and leaves the call at ⊤. A partial fold would apply SOME of a
 * model's layers and report the result as the model's output shape, which is a confidently wrong
 * shape rather than a missing one — the worse of the two, and the exact hazard of a layer list the
 * analysis can only partly see.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SequentialCall extends ModelCall {

  private static final Logger LOGGER = Logger.getLogger(SequentialCall.class.getName());

  /**
   * The IR index of the {@code layers} formal in {@code Sequential.do} ({@code paramNames="self
   * layers name"}): the callable occupies index 0, so the first real argument is index 1.
   */
  private static final int LAYERS_PARAMETER_IR_INDEX = 1;

  public SequentialCall(PointsToSetVariable source) {
    super(source);
  }

  public SequentialCall(CGNode node) {
    super(node);
  }

  /**
   * Composes the forward chain through the model's layer list, falling back to the inherited
   * functional-model treatment when the fold declines.
   *
   * @param builder The propagation call graph builder.
   * @return The composed output shapes, or the inherited result when no fold is available.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Composition composed = this.fold(builder);
    if (composed != null && composed.shapes() != null) return composed.shapes();

    LOGGER.fine(
        () -> "Sequential fold declined at " + describe(this.getNode()) + "; deferring to Model.");

    return super.getDefaultShapes(builder);
  }

  /**
   * The two axes a fold produces together.
   *
   * @param shapes The composed shapes, or {@code null} when the shape axis declined.
   * @param dTypes The composed dtypes, or {@code null} when the dtype axis declined.
   */
  private record Composition(Set<List<Dimension<?>>> shapes, Set<DType> dTypes) {}

  /**
   * The dtype twin of {@link #getDefaultShapes(PropagationCallGraphBuilder)}. A Keras layer's dtype
   * is its own business — {@code Dense} declares {@code float32} whatever it is fed — so the fold
   * carries dtypes through the same chain rather than assuming the input's survive it.
   *
   * @param builder The propagation call graph builder.
   * @return The composed output dtypes, or the inherited result when the fold declines.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Composition composed = this.fold(builder);

    return composed != null && composed.dTypes() != null && !composed.dTypes().isEmpty()
        ? composed.dTypes()
        : super.getDefaultDTypes(builder);
  }

  /**
   * Folds the model's layer list over the called value, on one axis.
   *
   * <p>One traversal serves both axes because the ordering, the dispatch, and the refusal
   * conditions are identical between them; only the accumulator's element type differs. Splitting
   * it would leave two copies of the refusal logic to drift apart, and a fold that refused on one
   * axis but not the other would emit a shape belonging to a chain the dtype axis had already
   * judged uncomposable.
   *
   * @param builder The propagation call graph builder.
   * @return The composed result, or {@code null} when the chain itself is uncomposable. An axis
   *     that declines on its own is {@code null} within a non-null result.
   */
  private Composition fold(PropagationCallGraphBuilder builder) {
    List<InstanceKey> layers = this.getLayers(builder);
    if (layers == null || layers.isEmpty()) return null;

    Set<List<Dimension<?>>> runningShapes =
        this.getArgumentShapesWithFallback(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
    Set<DType> runningDTypes =
        this.getArgumentDTypesWithFallback(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

    // The chain has to start somewhere: a model called with a value whose shape does not resolve
    // composes nothing on the shape axis, since every layer's output is a function of its input.
    if (runningShapes != null && runningShapes.isEmpty()) runningShapes = null;
    if (runningDTypes != null && runningDTypes.isEmpty()) runningDTypes = null;

    for (InstanceKey layer : layers) {
      TensorGenerator generator = this.dispatchLayer(builder, layer);
      if (generator == null) {
        LOGGER.fine(() -> "No generator for layer " + describe(layer) + "; declining the fold.");
        return null;
      }

      generator.composeWith(
          new ComposedArguments(
              OrdinalSet.toOrdinalSet(
                  List.of(layer), builder.getPointerAnalysis().getInstanceKeyMapping()),
              runningShapes,
              runningDTypes));

      // A layer that cannot type its own output ends that axis: everything after it would be
      // folded over a value the analysis does not have. The axes end independently, since a layer
      // declaring its own dtype (`Dense` is `float32` whatever it is fed) can carry the dtype
      // chain past the point where the shape chain has stopped.
      Set<List<Dimension<?>>> layerShapes =
          runningShapes == null ? null : generator.getDefaultShapes(builder);
      Set<DType> layerDTypes = generator.getDefaultDTypes(builder);

      runningShapes = layerShapes == null || layerShapes.isEmpty() ? null : layerShapes;
      runningDTypes = layerDTypes == null || layerDTypes.isEmpty() ? null : layerDTypes;
    }

    return new Composition(runningShapes, runningDTypes);
  }

  /**
   * Dispatches the generator carrying a layer instance's shape transform.
   *
   * <p>The layer's instance type names its call type by construction ({@code
   * Ltensorflow/keras/layers/Dense} against {@code Ltensorflow/keras/layers/Dense/__call__}), so
   * the existing dispatch table answers this without a second table keyed on layer classes.
   *
   * @param builder The propagation call graph builder.
   * @param layer The layer instance.
   * @return The generator, or {@code null} when the layer's class is unmodeled.
   */
  private TensorGenerator dispatchLayer(PropagationCallGraphBuilder builder, InstanceKey layer) {
    AllocationSiteInNode allocation = getAllocationSiteInNode(layer);
    if (allocation == null) return null;

    TypeReference layerType = allocation.concreteType().getReference();
    TypeReference callType =
        TypeReference.findOrCreate(
            layerType.getClassLoader(),
            TypeName.string2TypeName(layerType.getName().toString() + "/" + CALLABLE_METHOD_NAME));

    return createManualGenerator(this.getNode(), callType, builder);
  }

  /**
   * Resolves the model's layers, in list order.
   *
   * @param builder The propagation call graph builder.
   * @return The layer instances in order, or {@code null} when the list is not one the fold can
   *     read end to end.
   */
  private List<InstanceKey> getLayers(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPTS =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());

    List<InstanceKey> ret = null;

    for (InstanceKey selfIK : selfPTS) {
      AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);
      if (selfASIN == null) continue;
      if (!selfASIN
          .concreteType()
          .getReference()
          .equals(TensorFlowTypes.SEQUENTIAL.getDeclaringClass())) continue;

      IR ir = selfASIN.getNode().getIR();
      if (ir == null || LAYERS_PARAMETER_IR_INDEX >= ir.getNumberOfParameters()) continue;

      PointerKey layersPK =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(
                  selfASIN.getNode(), ir.getParameter(LAYERS_PARAMETER_IR_INDEX));

      List<InstanceKey> ordered =
          this.readContainer(builder, builder.getPointerAnalysis().getPointsToSet(layersPK));

      // Two models reaching one call site is imprecision the fold cannot resolve: their layer
      // lists are different chains, and folding either one would report it as THE output shape.
      if (ordered == null) return null;
      if (ret != null && !ret.equals(ordered)) return null;

      ret = ordered;
    }

    return ret;
  }

  /**
   * Reads a list or tuple's elements in index order.
   *
   * @param builder The propagation call graph builder.
   * @param containerPTS The container's points-to set.
   * @return The elements in index order, or {@code null} unless exactly one container resolves and
   *     its indices are the contiguous run from zero that a literal list produces.
   */
  private List<InstanceKey> readContainer(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> containerPTS) {
    AllocationSiteInNode container = null;

    for (InstanceKey ik : containerPTS) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      if (asin == null) continue;
      TypeReference reference = asin.concreteType().getReference();
      if (!reference.equals(list) && !reference.equals(tuple)) continue;
      // A second container is a second layer chain; see the caller.
      if (container != null) return null;
      container = asin;
    }

    if (container == null) return null;

    OrdinalSet<InstanceKey> catalogPTS =
        builder
            .getPointerAnalysis()
            .getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(container));

    TreeMap<Integer, InstanceKey> byIndex = new TreeMap<>();

    for (InstanceKey catalogIK : catalogPTS) {
      if (!(catalogIK instanceof ConstantKey)) continue;
      Integer index = getFieldIndex((ConstantKey<?>) catalogIK);
      if (index == null) continue;

      IField field =
          builder
              .getClassHierarchy()
              .resolveField(
                  FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(index.toString()), Root));
      if (field == null) return null;

      OrdinalSet<InstanceKey> elementPTS =
          builder
              .getPointerAnalysis()
              .getPointsToSet(builder.getPointerKeyForInstanceField(container, field));

      // One layer per position. A smeared position — the shape a loop-built list takes when its
      // elements collapse — is exactly the case where folding what is visible would compose a
      // chain the program does not have.
      if (elementPTS == null || elementPTS.size() != 1) return null;

      byIndex.put(index, elementPTS.iterator().next());
    }

    if (byIndex.isEmpty()) return null;

    // A run that does not start at zero or that skips a position is a list the analysis is seeing
    // only part of, and the missing layers are missing transforms.
    if (byIndex.firstKey() != 0 || byIndex.lastKey() != byIndex.size() - 1) return null;

    return new ArrayList<>(byIndex.values());
  }
}
