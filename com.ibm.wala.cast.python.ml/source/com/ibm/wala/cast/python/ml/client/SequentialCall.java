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
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSACFG;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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

  /** The keyword name of that same argument. */
  private static final String LAYERS_PARAMETER_NAME = "layers";

  /** The method a model built one layer at a time is grown with. */
  private static final String ADD_METHOD_NAME = "add";

  /**
   * The model members the walk will tolerate beside {@code add}, being the ones that cannot change
   * what is in the layer list.
   *
   * <p>A list rather than a rejection of the mutating ones, because the safe direction is deny by
   * default: {@code pop} exists to shrink the chain, and any allowance framed as "a member whose
   * name is a constant" admits it. {@code layers} is deliberately absent, since reading it yields
   * the list itself, which a program can then append to.
   *
   * <p>Each entry is a claim about the Keras API, and a wrong one is caught by a composing fixture
   * rather than by reading this list: a member that grew or shrank the chain would make the fold
   * disagree with the runtime. {@code build}, {@code compile}, {@code summary}, {@code
   * count_params} and {@code get_weights} carry that witness. The rest are unwitnessed, and an
   * entry added here should arrive with a fixture that composes through it.
   */
  private static final Set<String> NON_MUTATING_MODEL_MEMBERS =
      Set.of(
          "build",
          "call",
          "compile",
          "count_params",
          "evaluate",
          "fit",
          "get_weights",
          "load_weights",
          "predict",
          "save",
          "save_weights",
          "set_weights",
          "summary",
          "trainable_variables",
          "trainable_weights",
          "weights");

  /** How far a returned model is followed through its callers before the walk gives up. */
  private static final int ESCAPE_FOLLOWING_DEPTH = 4;

  /**
   * The most chains a conditionally-built model may have before the walk declines. A builder with a
   * handful of optional stages is the idiom; a body whose branching multiplies past this is one
   * whose paths the walk is no longer really enumerating.
   */
  private static final int MAXIMUM_CHAINS = 8;

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
    List<List<InstanceKey>> chains = this.getLayerChains(builder);
    if (chains == null || chains.isEmpty()) return null;

    Set<List<Dimension<?>>> inputShapes =
        this.getArgumentShapesWithFallback(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());
    Set<DType> inputDTypes =
        this.getArgumentDTypesWithFallback(
            builder, Parameters.INPUTS.getIndex(), Parameters.INPUTS.getName());

    Set<List<Dimension<?>>> shapes = HashSetFactory.make();
    Set<DType> dTypes = HashSetFactory.make();

    // A model whose layers are added under a condition has more than one chain, and the result is
    // the union over them: each is a chain the program can actually run. One uncomposable chain
    // sinks the whole union, since a union missing one of its members understates the value.
    for (List<InstanceKey> chain : chains) {
      Composition composed = this.foldChain(builder, chain, inputShapes, inputDTypes);
      if (composed == null) return null;
      if (composed.shapes() == null) shapes = null;
      else if (shapes != null) shapes.addAll(composed.shapes());
      if (composed.dTypes() == null) dTypes = null;
      else if (dTypes != null) dTypes.addAll(composed.dTypes());
    }

    return new Composition(
        shapes == null || shapes.isEmpty() ? null : shapes,
        dTypes == null || dTypes.isEmpty() ? null : dTypes);
  }

  /**
   * Folds one chain of layers over the called value.
   *
   * @param builder The propagation call graph builder.
   * @param layers The chain, in application order.
   * @param inputShapes The shapes of the value the model was called with.
   * @param inputDTypes The dtypes of that value.
   * @return The composed result, or {@code null} when a layer in the chain has no transform.
   */
  private Composition foldChain(
      PropagationCallGraphBuilder builder,
      List<InstanceKey> layers,
      Set<List<Dimension<?>>> inputShapes,
      Set<DType> inputDTypes) {
    Set<List<Dimension<?>>> runningShapes = inputShapes;
    Set<DType> runningDTypes = inputDTypes;

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

    TensorGenerator generator = createManualGenerator(this.getNode(), callType, builder);
    if (generator != null) return generator;

    // A USER class in the list has no table arm; its transform is its own `call` body, folded by
    // the same discipline one level down (wala/ML#832).
    return UserLayerCall.forLayerInstance(
        builder, this.getNode(), layer, UserLayerCall.BODY_FOLD_DEPTH_CAP);
  }

  /**
   * Resolves the chains of layers the model can apply.
   *
   * <p>Usually one, from the constructor's list. A model built by {@code .add()} calls has one
   * chain per path through the building code, since a layer added under a condition is in some runs
   * and not others.
   *
   * @param builder The propagation call graph builder.
   * @return The chains, or {@code null} when the model's layers are not ones the fold can read end
   *     to end.
   */
  private List<List<InstanceKey>> getLayerChains(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPTS =
        this.getArgumentPointsToSet(builder, Parameters.SELF.getIndex(), Parameters.SELF.getName());

    List<List<InstanceKey>> ret = null;

    for (InstanceKey selfIK : selfPTS) {
      AllocationSiteInNode selfASIN = getAllocationSiteInNode(selfIK);
      if (selfASIN == null) continue;
      if (!selfASIN
          .concreteType()
          .getReference()
          .equals(TensorFlowTypes.SEQUENTIAL.getDeclaringClass())) continue;

      IR ir = selfASIN.getNode().getIR();
      if (ir == null || LAYERS_PARAMETER_IR_INDEX >= ir.getNumberOfParameters()) continue;

      List<List<InstanceKey>> chains = this.getChainsOfModel(builder, selfASIN.getNode(), ir);

      // Two models reaching one call site is imprecision the fold cannot resolve: their layer
      // lists are different chains, and folding either one would report it as THE output shape.
      if (chains == null) return null;
      if (ret != null && !ret.equals(chains)) return null;

      ret = chains;
    }

    return ret;
  }

  /**
   * Resolves one model instance's chains, from whichever of the two spellings built it.
   *
   * @param builder The propagation call graph builder.
   * @param constructorNode The synthetic {@code Sequential.do} node allocating the model.
   * @param ir That node's IR.
   * @return The chains, or {@code null} when the model's layers cannot be read whole.
   */
  private List<List<InstanceKey>> getChainsOfModel(
      PropagationCallGraphBuilder builder, CGNode constructorNode, IR ir) {
    // The points-to set alone cannot tell a list the caller WROTE from one it derived: a slice
    // application's result aliases its receiver, so `Sequential(base[:2])` hands the fold the FULL
    // list's allocation and every refusal below sees a perfectly well-formed catalog. The
    // constructor's own argument has to be the literal (wala/ML#832).
    if (this.constructorArgumentIsLiteralContainer(builder, constructorNode)) {
      PointerKey layersPK =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(constructorNode, ir.getParameter(LAYERS_PARAMETER_IR_INDEX));

      List<InstanceKey> ordered =
          this.readContainer(builder, builder.getPointerAnalysis().getPointsToSet(layersPK));

      return ordered == null ? null : List.of(ordered);
    }

    return this.getChainsFromAddCalls(builder, constructorNode);
  }

  /**
   * Resolves the chains of a model built by {@code .add()} calls rather than by a constructor list
   * (the remaining half of <a href="https://github.com/wala/ML/issues/854">wala/ML#854</a>).
   *
   * <p>This spelling keeps the same information in a different place, and neither the model nor the
   * points-to substrate holds it: the layers arrive one call at a time, and their ORDER, which the
   * whole composition depends on, exists only as the order of the calls in the building code. So
   * the recovery is a walk of that code rather than a read of a container.
   *
   * <p>A layer added under a condition is the reason this returns chains rather than a chain. The
   * canonical builder is exactly that shape, a stage that adds a normalization only when it is
   * asked for, and the two paths are two different models. Enumerating the paths gives each its own
   * chain, and the fold unions their results, which is what a value with two possible producers
   * means. Composing either alone, or pretending the conditional layer is always there, would be a
   * shape for a program that was not run.
   *
   * @param builder The propagation call graph builder.
   * @param constructorNode The synthetic {@code Sequential.do} node allocating the model.
   * @return The chains, or {@code null} when the building code cannot be read whole.
   */
  private List<List<InstanceKey>> getChainsFromAddCalls(
      PropagationCallGraphBuilder builder, CGNode constructorNode) {
    List<List<InstanceKey>> ret = null;
    boolean sawCallSite = false;

    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, constructorNode)) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction construction)) continue;

      sawCallSite = true;

      CGNode caller = callerInvoke.fst;
      if (caller.getIR() == null || caller.getDU() == null) return null;

      List<List<InstanceKey>> chains =
          chainsBuiltInFrame(builder, caller, construction, construction.getDef());

      if (chains == null) return null;
      if (ret != null && !ret.equals(chains)) return null;

      ret = chains;
    }

    return sawCallSite ? ret : null;
  }

  /**
   * Walks one frame's building code for the chains it can add to a model.
   *
   * @param builder The propagation call graph builder.
   * @param frame The node whose body constructs the model.
   * @param construction The {@code Sequential()} invoke.
   * @param modelVn The value number the construction defines.
   * @return The chains, or {@code null} on anything the walk cannot account for.
   */
  private static List<List<InstanceKey>> chainsBuiltInFrame(
      PropagationCallGraphBuilder builder,
      CGNode frame,
      PythonInvokeInstruction construction,
      int modelVn) {
    if (modelVn <= 0) return null;

    IR ir = frame.getIR();
    Map<Integer, InstanceKey> addedAt = new HashMap<>();

    // Every use of the model has to be one that cannot change what is in it. The walk sees one
    // frame, and any operation elsewhere that grows or shrinks the layer list makes the chain it
    // composed the wrong length.
    if (!modelIsUnmodifiedIn(frame, modelVn, construction, true)) return null;

    // Returning the model hands it to callers that can add to it after this frame is done, which
    // the walk cannot see from here and which produces a chain SHORTER than the program's. The
    // idiom is common enough (a builder function returning its stage) to be worth following rather
    // than declining outright, so every caller of this frame is held to the same discipline.
    if (modelIsReturnedFrom(frame, modelVn)
        && !returnedModelIsUnmodified(builder, frame, ESCAPE_FOLLOWING_DEPTH)) return null;

    for (SSAInstruction instruction : ir.getInstructions()) {
      if (!(instruction instanceof PythonInvokeInstruction call)) continue;
      if (call.getNumberOfPositionalParameters() < 2) continue;

      SSAInstruction calleeDef = frame.getDU().getDef(call.getUse(0));
      if (!(calleeDef instanceof PythonPropertyRead read) || read.getObjectRef() != modelVn)
        continue;

      SymbolTable symbolTable = ir.getSymbolTable();
      if (!symbolTable.isStringConstant(read.getMemberRef())) return null;
      if (!ADD_METHOD_NAME.equals(symbolTable.getStringValue(read.getMemberRef()))) continue;

      OrdinalSet<InstanceKey> layerPTS =
          builder
              .getPointerAnalysis()
              .getPointsToSet(
                  builder
                      .getPointerAnalysis()
                      .getHeapModel()
                      .getPointerKeyForLocal(frame, call.getUse(1)));

      // One layer per call, as the source has it. A smeared argument is a call the walk cannot
      // turn into a position in the chain.
      if (layerPTS == null || layerPTS.size() != 1) return null;

      addedAt.put(call.iIndex(), layerPTS.iterator().next());
    }

    if (addedAt.isEmpty()) return null;

    List<List<InstanceKey>> chains = new ArrayList<>();
    SSACFG cfg = ir.getControlFlowGraph();

    return collectChains(
            cfg,
            cfg.getBlockForInstruction(construction.iIndex()),
            addedAt,
            new ArrayList<>(),
            new HashSet<>(),
            chains)
        ? chains
        : null;
  }

  /**
   * Whether every use of the model in one frame leaves its layer list alone.
   *
   * <p>The allowance is a LIST of members known not to change the list, not a rejection of the ones
   * known to. Allowing anything whose name is merely a constant admits {@code pop()}, whose whole
   * purpose is to shrink the chain, and the walk would then compose a model longer than the one the
   * program runs. Deny by default is the only direction that is safe here, and the cost is that a
   * program calling some other model method declines rather than composing.
   *
   * @param frame The node whose body holds the uses.
   * @param modelVn The model's value number in that frame.
   * @param construction The invoke that defines it, whose own use of the value is its definition.
   * @param addAllowed Whether {@code add} is a safe use here, which it is only in the frame whose
   *     adds the walk collected.
   * @return {@code true} iff every use is safe.
   */
  private static boolean modelIsUnmodifiedIn(
      CGNode frame, int modelVn, SSAInstruction construction, boolean addAllowed) {
    SymbolTable symbolTable = frame.getIR().getSymbolTable();

    for (Iterator<SSAInstruction> uses = frame.getDU().getUses(modelVn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();

      // Returning is handled by the caller, which follows the value instead of stopping here.
      if (use instanceof SSAReturnInstruction) continue;

      if (use instanceof PythonPropertyRead read && read.getObjectRef() == modelVn) {
        if (!symbolTable.isStringConstant(read.getMemberRef())) return false;
        String member = symbolTable.getStringValue(read.getMemberRef());
        // `add` is the builder, and only in the frame whose adds the walk collected. The SAME call
        // in a caller of that frame is the escape: those layers land at the end of a chain this
        // walk has already finished reading.
        boolean growsHere = ADD_METHOD_NAME.equals(member) && addAllowed;
        if (!growsHere && !NON_MUTATING_MODEL_MEMBERS.contains(member)) return false;
        continue;
      }

      // The construction itself uses the value only as its definition.
      if (use.equals(construction)) continue;

      // Calling the model does not change what is in it, so a use as the callee is fine. A use as
      // an ARGUMENT is an escape: the callee could add to it, and the chain this frame sees would
      // be a prefix of the real one.
      if (use instanceof PythonInvokeInstruction call && call.getUse(0) == modelVn) {
        boolean asArgument = false;
        for (int i = 1; i < call.getNumberOfUses(); i++)
          if (call.getUse(i) == modelVn) asArgument = true;
        if (!asArgument) continue;
      }

      return false;
    }

    return true;
  }

  /**
   * Whether the model leaves its frame as a return value.
   *
   * @param frame The node whose body constructs it.
   * @param modelVn The model's value number.
   * @return {@code true} iff some return instruction returns it.
   */
  private static boolean modelIsReturnedFrom(CGNode frame, int modelVn) {
    for (Iterator<SSAInstruction> uses = frame.getDU().getUses(modelVn); uses.hasNext(); )
      if (uses.next() instanceof SSAReturnInstruction) return true;

    return false;
  }

  /**
   * Whether every caller of a frame that returns the model leaves the model's layer list alone.
   *
   * <p>A builder function that returns its stage is the ordinary way this spelling is written, so
   * declining on the return itself would give up the case worth having. Following it costs one
   * check per caller: the returned value is held to the same use discipline, and a caller that
   * returns it onward is followed in turn.
   *
   * <p>A caller that DOES add is declined rather than composed. Its layers would belong at the end
   * of the chain, which is recoverable in principle, but a chain assembled across frames is a
   * different walk than this one and would need its own witnesses.
   *
   * @param builder The propagation call graph builder.
   * @param frame The node returning the model.
   * @param depth The remaining budget for following returns onward.
   * @return {@code true} iff no caller can modify the model.
   */
  private static boolean returnedModelIsUnmodified(
      PropagationCallGraphBuilder builder, CGNode frame, int depth) {
    if (depth <= 0) return false;

    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke : getCallerInvokes(builder, frame))
      if (callerInvoke.snd instanceof PythonInvokeInstruction call) {
        CGNode caller = callerInvoke.fst;
        if (caller.getIR() == null || caller.getDU() == null) return false;

        int returnedVn = call.getDef();
        // A result nobody names cannot be modified.
        if (returnedVn <= 0) continue;

        if (!modelIsUnmodifiedIn(caller, returnedVn, call, false)) return false;
        if (modelIsReturnedFrom(caller, returnedVn)
            && !returnedModelIsUnmodified(builder, caller, depth - 1)) return false;
      }

    return true;
  }

  /**
   * Enumerates the add sequences along every path from the construction to the frame's exit.
   *
   * @param cfg The frame's control-flow graph.
   * @param block The block to continue from.
   * @param addedAt The layer added at each {@code add} call's instruction index.
   * @param prefix The layers collected on the path so far.
   * @param onPath The blocks already on this path, breaking cycles.
   * @param chains The distinct chains found so far.
   * @return {@code false} when the walk cannot enumerate the paths: a loop reaches an {@code add}
   *     call, whose trip count decides the chain's LENGTH, or the branching exceeds the cap.
   */
  private static boolean collectChains(
      SSACFG cfg,
      ISSABasicBlock block,
      Map<Integer, InstanceKey> addedAt,
      List<InstanceKey> prefix,
      Set<ISSABasicBlock> onPath,
      List<List<InstanceKey>> chains) {
    if (block == null || chains.size() > MAXIMUM_CHAINS) return false;

    if (!onPath.add(block)) {
      // A back edge. It is only fatal when it can repeat an `add`, since a loop over unrelated
      // code leaves the chain alone; a loop that adds layers makes the chain's length a trip
      // count, which is not a static list at all.
      for (int i = block.getFirstInstructionIndex(); i <= block.getLastInstructionIndex(); i++)
        if (addedAt.containsKey(i)) return false;
      return true;
    }

    try {
      List<InstanceKey> extended = new ArrayList<>(prefix);
      for (int i = block.getFirstInstructionIndex(); i <= block.getLastInstructionIndex(); i++) {
        InstanceKey added = addedAt.get(i);
        if (added != null) extended.add(added);
      }

      boolean terminal = true;
      for (ISSABasicBlock successor : cfg.getNormalSuccessors(block)) {
        if (successor.isExitBlock()) continue;
        terminal = false;
        if (!collectChains(cfg, successor, addedAt, extended, onPath, chains)) return false;
      }

      if (terminal && !extended.isEmpty() && !chains.contains(extended)) chains.add(extended);

      return chains.size() <= MAXIMUM_CHAINS;
    } finally {
      onPath.remove(block);
    }
  }

  /**
   * Whether every {@code Sequential(...)} call site that built this model passed a container the
   * caller allocated on the spot.
   *
   * <p>This is the refusal for derived lists. A slice application's result aliases its receiver, so
   * {@code Sequential(base[:2])} reaches the fold as the FULL list's allocation site, with a
   * catalog that is contiguous, single-valued at every position, and simply the wrong list. No
   * property of the container can distinguish that case, because the container IS the receiver;
   * only the argument's own definition can, and a literal is the one definition that guarantees the
   * value the fold reads is the value the constructor received.
   *
   * <p>The cost is precision on a list built elsewhere and passed in, which now declines rather
   * than composing. That is the correct direction: such a list's contents are whatever reached it
   * along every path, and folding them would report one path's chain as the model's.
   *
   * @param builder The propagation call graph builder.
   * @param constructorNode The synthetic {@code Sequential.do} node allocating the model.
   * @return {@code true} iff at least one call site was found and all of them pass a literal.
   */
  private boolean constructorArgumentIsLiteralContainer(
      PropagationCallGraphBuilder builder, CGNode constructorNode) {
    boolean sawCallSite = false;

    for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
        getCallerInvokes(builder, constructorNode)) {
      if (!(callerInvoke.snd instanceof PythonInvokeInstruction call)) continue;

      sawCallSite = true;

      int argVn = call.getUse(LAYERS_PARAMETER_NAME);
      if (argVn == -1 && call.getNumberOfPositionalParameters() > LAYERS_PARAMETER_IR_INDEX)
        argVn = call.getUse(LAYERS_PARAMETER_IR_INDEX);
      if (argVn <= 0) return false;

      // A parameter has no defining instruction, which is itself a decline: the list was built
      // somewhere this call site cannot see.
      SSAInstruction def = callerInvoke.fst.getDU().getDef(argVn);
      if (!(def instanceof SSANewInstruction allocation)) return false;

      TypeReference allocated = allocation.getConcreteType();
      if (!allocated.equals(list) && !allocated.equals(tuple)) return false;
    }

    return sawCallSite;
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

    // A list the program appended to keeps those elements under the synthetic append-contents
    // field rather than at integer indices, so the catalog below shows only what the literal held
    // and reads as a complete, contiguous run. Appended contents are positive evidence that the
    // catalog UNDERCOUNTS, which no property of the catalog itself can supply (wala/ML#832).
    if (getAppendedContentsPts(builder, container) != null) {
      LOGGER.fine(
          () -> "Layer list has appended contents; the catalog undercounts. Declining the fold.");
      return null;
    }

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
