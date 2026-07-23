package com.ibm.wala.cast.python.ml.client;

import static com.google.common.collect.Sets.newHashSet;
import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory.getGenerator;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATA_PACKAGE_PREFIX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR_FUNCTIONS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_FUNCTIONS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_TYPE;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.lsp.AnalysisError;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.TrampolineReceiverContextSelector;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.PythonInterpreter;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.AnalysisOptions;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.ContextSelector;
import com.ibm.wala.ipa.callgraph.IAnalysisCacheView;
import com.ibm.wala.ipa.callgraph.impl.ContextInsensitiveSelector;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.HeapModel;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.PropagationSystem;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector;
import com.ibm.wala.ipa.callgraph.propagation.cfa.nCFAContextSelector;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.ISSABasicBlock;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPhiInstruction;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.NullProgressMonitor;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.graph.Graph;
import com.ibm.wala.util.graph.impl.SlowSparseNumberedGraph;
import com.ibm.wala.util.intset.IntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public class PythonTensorAnalysisEngine extends PythonAnalysisEngine<TensorTypeAnalysis> {

  public static final String TENSORFLOW = TensorFlowTypes.TENSORFLOW;

  /**
   * The default k-CFA depth applied to framework API methods (and user model subclasses). The
   * migration from {@code read_data} synthetic methods to inline {@code do()} allocations required
   * 2-CFA to keep framework-API precision; deeper context is occasionally needed when allocations
   * merge across call sites (wala/ML#379, wala/ML#530).
   */
  public static final int DEFAULT_TARGETED_CFA_DEPTH = 2;

  /**
   * The recommended targeted k-CFA depth for consumers analyzing the <em>model-forward
   * archetype</em>: {@code tf.keras.Model} subclasses whose {@code call}/{@code predict} chain
   * layer invocations ({@code x = self.layer1(x); x = self.layer2(x)}). Such a model invoked from
   * multiple call sites (e.g. train vs. test) needs enough call-string context to keep its
   * per-call-site layer-output allocations distinct; at the back-compat {@link
   * #DEFAULT_TARGETED_CFA_DEPTH} they merge and the per-context shapes collapse to &#8868;
   * (unknown). Depth 4 separates the typical two-to-four-layer chain &mdash; validated by {@code
   * testNeuralNetwork} through {@code testNeuralNetwork4}, which recover e.g. {@code accuracy}'s
   * {@code y_pred} as the per-context union {@code {(256, 10) float32, ? float32}} (wala/ML#379,
   * wala/ML#530).
   *
   * <p>This is a recommended default for the archetype, not a universal constant: the depth a model
   * needs scales with its layer-chain length and call-site nesting. The targeted context selector
   * applies deep context only to framework-API methods and {@code tf.keras.Model} forward methods
   * (all other code stays context-insensitive), so the cost stays bounded to that subgraph rather
   * than whole-program k-CFA; within it, context count still grows with depth. Deepen past this
   * only when per-context layer outputs still merge (the symptom: a shape that should be a
   * per-context union stays &#8868; or single-valued), and measure on the deepest model before
   * raising it.
   *
   * <p>Usage: {@code new PythonTensorAnalysisEngine(TENSORFLOW, MODEL_FORWARD_CFA_DEPTH)}.
   */
  public static final int MODEL_FORWARD_CFA_DEPTH = 4;

  private final String targetFramework;

  /**
   * The k-CFA depth applied to the targeted context selector. See {@link
   * #DEFAULT_TARGETED_CFA_DEPTH}.
   */
  private final int targetedCfaDepth;

  public PythonTensorAnalysisEngine() {
    this(TENSORFLOW);
  }

  public PythonTensorAnalysisEngine(List<File> pythonPath) {
    this(pythonPath, TENSORFLOW);
  }

  public PythonTensorAnalysisEngine(String targetFramework) {
    this(targetFramework, DEFAULT_TARGETED_CFA_DEPTH);
  }

  /**
   * Validates the requested k-CFA depth for the targeted context selector. The depth is the
   * call-string length passed to {@link nCFAContextSelector}, so it must be non-negative (0 is
   * context-insensitive; higher values are deeper).
   *
   * @param targetedCfaDepth The requested depth.
   * @return {@code targetedCfaDepth} unchanged when valid.
   * @throws IllegalArgumentException if {@code targetedCfaDepth} is negative.
   */
  private static int checkTargetedCfaDepth(int targetedCfaDepth) {
    if (targetedCfaDepth < 0)
      throw new IllegalArgumentException(
          "The targeted k-CFA depth must be non-negative: " + targetedCfaDepth + ".");
    return targetedCfaDepth;
  }

  /**
   * Constructs an engine with a configurable k-CFA depth for the targeted context selector.
   *
   * @param targetFramework The framework name prefix whose API methods receive deep context.
   * @param targetedCfaDepth The k-CFA depth for the targeted context selector; must be
   *     non-negative. Use {@link #DEFAULT_TARGETED_CFA_DEPTH} for framework-API precision, or
   *     {@link #MODEL_FORWARD_CFA_DEPTH} when analyzing chained-layer {@code tf.keras.Model}
   *     forward passes. Higher values increase precision at the cost of analysis time.
   * @throws IllegalArgumentException if {@code targetedCfaDepth} is negative.
   */
  public PythonTensorAnalysisEngine(String targetFramework, int targetedCfaDepth) {
    this.targetFramework = targetFramework;
    this.targetedCfaDepth = checkTargetedCfaDepth(targetedCfaDepth);
  }

  public PythonTensorAnalysisEngine(List<File> pythonPath, String targetFramework) {
    this(pythonPath, targetFramework, DEFAULT_TARGETED_CFA_DEPTH);
  }

  /**
   * Constructs an engine with an explicit Python path and a configurable k-CFA depth.
   *
   * @param pythonPath The additional Python path entries for module resolution.
   * @param targetFramework The framework name prefix whose API methods receive deep context.
   * @param targetedCfaDepth The k-CFA depth for the targeted context selector; must be
   *     non-negative. Use {@link #DEFAULT_TARGETED_CFA_DEPTH} for framework-API precision, or
   *     {@link #MODEL_FORWARD_CFA_DEPTH} when analyzing chained-layer {@code tf.keras.Model}
   *     forward passes. Higher values increase precision at the cost of analysis time.
   * @throws IllegalArgumentException if {@code targetedCfaDepth} is negative.
   */
  public PythonTensorAnalysisEngine(
      List<File> pythonPath, String targetFramework, int targetedCfaDepth) {
    super(pythonPath);
    this.targetFramework = targetFramework;
    this.targetedCfaDepth = checkTargetedCfaDepth(targetedCfaDepth);
  }

  @Override
  protected PythonSSAPropagationCallGraphBuilder getCallGraphBuilder(
      IClassHierarchy cha, AnalysisOptions options, IAnalysisCacheView cache2) {
    PythonSSAPropagationCallGraphBuilder builder = super.getCallGraphBuilder(cha, options, cache2);

    final ContextSelector base = builder.getContextSelector();
    final ContextSelector targetedCFA =
        new nCFAContextSelector(this.targetedCfaDepth, new ContextInsensitiveSelector());
    final IClass modelClass = cha.lookupClass(TensorFlowTypes.MODEL.getDeclaringClass());

    // The trampoline-receiver rules must stay outermost: the targeted-CFA routing below matches
    // trampoline classes too (a `$Foo/call` trampoline ends in a forward-method segment), and
    // call strings re-collapse the per-receiver distinction those rules establish (wala/ML#679).
    builder.setContextSelector(
        new TrampolineReceiverContextSelector(
            new ContextSelector() {
              @Override
              public Context getCalleeTarget(
                  CGNode caller,
                  CallSiteReference site,
                  IMethod callee,
                  InstanceKey[] actualParameters) {
                // Apply k-CFA for any methods in the target framework, which includes internal
                // helpers, as well as methods declared on user-defined `tf.keras.Model` subclasses
                // (e.g. `NeuralNet.call`). Without the latter, a user model called from multiple
                // sites (train vs. test) merges into one context-insensitive node, collapsing its
                // layer-output allocations across callers and losing per-context shape
                // (wala/ML#530).
                if (receivesTargetedContext(callee, modelClass)) {
                  return targetedCFA.getCalleeTarget(caller, site, callee, actualParameters);
                }
                return base.getCalleeTarget(caller, site, callee, actualParameters);
              }

              @Override
              public IntSet getRelevantParameters(CGNode caller, CallSiteReference site) {
                return base.getRelevantParameters(caller, site);
              }
            }));

    return builder;
  }

  /**
   * Determines whether {@code callee} is declared on a user-defined subclass of {@code
   * tf.keras.Model}. Walks the declaring class's superclass chain looking for {@code modelClass}.
   * Methods on such classes (the forward-pass {@code call}/{@code predict}) receive deep context so
   * a model invoked from multiple call sites does not merge its layer-output allocations
   * (wala/ML#530).
   *
   * @param callee The method being dispatched.
   * @param modelClass The resolved {@code tf.keras.Model} class, or {@code null} if it is absent
   *     from the class hierarchy (in which case no method is treated as a model-subclass method).
   * @return {@code true} if {@code callee}'s declaring class transitively extends {@code
   *     modelClass}; {@code false} otherwise.
   */
  private static boolean isModelSubclassMethod(IMethod callee, IClass modelClass) {
    if (modelClass == null) return false;
    for (IClass c = callee.getDeclaringClass(); c != null; c = c.getSuperclass())
      if (c.equals(modelClass)) return true;
    return false;
  }

  /**
   * Determines whether {@code method} receives deep ({@code targetedCfaDepth}) k-CFA context rather
   * than the context-insensitive base. This is the routing decision the call-graph builder's
   * context selector applies (framework-API methods and {@code tf.keras.Model} forward methods); it
   * is also the eligibility gate for the depth-too-short signal (wala/ML#601), since only these
   * methods' contexts are governed by {@code targetedCfaDepth} &mdash; a ⊤ elsewhere cannot be
   * resolved by deepening it.
   *
   * @param method The method whose context routing is being decided.
   * @param modelClass The resolved {@code tf.keras.Model} class, or {@code null} when absent.
   * @return {@code true} if {@code method} is routed through the targeted k-CFA selector.
   */
  private boolean receivesTargetedContext(IMethod method, IClass modelClass) {
    String declaringClass = method.getDeclaringClass().getName().toString();
    return declaringClass.contains(targetFramework)
        || isModelSubclassMethod(method, modelClass)
        || isUserModelForwardMethod(declaringClass);
  }

  /**
   * Heuristic fallback for {@link #isModelSubclassMethod}: the front-end does not yet record a user
   * model class's framework base class, so {@code class NeuralNet(Model)} has superclass {@code
   * Lobject} rather than {@code tf.keras.Model} and the CHA-subtype check above never matches (see
   * <a href="https://github.com/wala/ML/issues/571">wala/ML#571</a>). Until then, recognize a user
   * model's forward method structurally by its declaring class name: a script-defined class whose
   * final segment is a Keras forward-pass method ({@code call}, {@code __call__}, or {@code
   * predict}). Once wala/ML#571 lands, this heuristic can be removed in favor of {@link
   * #isModelSubclassMethod}.
   *
   * @param calleeClass The declaring class name of the dispatched method.
   * @return {@code true} if {@code calleeClass} names a user-defined model forward method.
   */
  private static boolean isUserModelForwardMethod(String calleeClass) {
    return calleeClass.contains("script ")
        && (calleeClass.endsWith("/" + PythonTypes.CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)
            || calleeClass.endsWith("/" + PythonTypes.CALLABLE_METHOD_NAME)
            || calleeClass.endsWith("/predict"));
  }

  /** A "fake" function name in the summaries that indicates that an API produces a new tensor. */
  public static final String TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME = "read_data";

  /**
   * System property naming the wala/ML#756 order-perturbation seed: a {@code long} that seeds a
   * deterministic shuffle of the demand-root order and the worklist engine's cycle-internal orders.
   * The {@link #SHUFFLE_CYCLES_VARIABLE} environment variable is its fallback.
   */
  public static final String SHUFFLE_CYCLES_PROPERTY = "ariadne.typeResolution.shuffleCycles";

  /** Environment-variable fallback of {@link #SHUFFLE_CYCLES_PROPERTY}. */
  public static final String SHUFFLE_CYCLES_VARIABLE = "ARIADNE_SHUFFLE_CYCLES";

  /**
   * System property naming the wala/ML#753 post-settlement replay filter: semicolon-separated
   * substrings selecting the settled shape queries whose transfers re-run read-only against the
   * settled state, logging each aggregation's per-member results. Plain segments match pure-⊤
   * queries by key string; {@code value:}-prefixed segments match any shape query by value string.
   * The {@link #REPLAY_FILTER_VARIABLE} environment variable is its fallback.
   */
  public static final String REPLAY_FILTER_PROPERTY = "ariadne.typeResolution.replayFilter";

  /** Environment-variable fallback of {@link #REPLAY_FILTER_PROPERTY}. */
  public static final String REPLAY_FILTER_VARIABLE = "ARIADNE_REPLAY_FILTER";

  private static final Logger LOGGER = Logger.getLogger(PythonTensorAnalysisEngine.class.getName());

  private static final MethodReference conv2d =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/conv2d")),
          AstMethodReference.fnSelector);

  private static final MethodReference conv3d =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/conv3d")),
          AstMethodReference.fnSelector);

  private static final MethodReference reshape =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/reshape")),
          AstMethodReference.fnSelector);

  private static final MethodReference placeholder =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/placeholder")),
          AstMethodReference.fnSelector);

  /** The Python attribute name for {@code tf.Tensor.set_shape}, used by the IR-syntactic scan. */
  private static final String SET_SHAPE_ATTRIBUTE = "set_shape";

  /**
   * WALA allocation types whose Python counterparts expose a public {@code set_shape} method
   * ({@code tf.Tensor} and {@code tf.SparseTensor}). Used as a receiver-type whitelist by {@link
   * #getSetShapeCallsSyntactic} to filter out non-tensor receivers that happen to invoke {@code
   * set_shape}. {@code tf.RaggedTensor} and {@code tf.IndexedSlices} aren't included pending
   * verified test fixtures.
   */
  private static final Set<TypeReference> SET_SHAPE_RECEIVER_TYPES =
      Set.of(TENSOR_FUNCTIONS_TYPE, TENSOR_TYPE, SPARSE_TENSOR_FUNCTIONS_TYPE, SPARSE_TENSOR_TYPE);

  private static final MethodReference convert_to_tensor =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/convert_to_tensor")),
          AstMethodReference.fnSelector);

  private static final MethodReference ENUMERATE =
      MethodReference.findOrCreate(PythonTypes.ENUMERATE_BUILTIN, AstMethodReference.fnSelector);

  private static final MethodReference NEXT =
      MethodReference.findOrCreate(PythonTypes.NEXT_BUILTIN, AstMethodReference.fnSelector);

  private final Map<PointerKey, AnalysisError> errorLog = HashMapFactory.make();

  /**
   * A tensor value that resolved to ⊤ (unknown) at a call-string context saturated to the
   * configured {@code targetedCfaDepth} (wala/ML#601). Because the value's node is routed through
   * the targeted k-CFA selector (see {@link #receivesTargetedContext}) and its call string was
   * truncated at the depth budget, a deeper {@code targetedCfaDepth} might separate the merged
   * caller contexts and resolve the value; a ⊤ at an unsaturated context (its call string reached
   * the call-graph root within the budget) cannot be helped by more depth, so it is not recorded.
   * The signal over-approximates: saturation is necessary but not sufficient for a depth-induced ⊤,
   * matching the warning (not error) semantics of the Java 8 Stream Refactoring precedent (see <a
   * href="https://github.com/wala/ML/issues/601">wala/ML#601</a>).
   *
   * @param pointerKey The local pointer key of the ⊤ value.
   * @param node The call-graph node holding the value.
   * @param valueNumber The value number within {@code node}.
   * @param callStringLength The (saturated) call-string length of {@code node}'s context, equal to
   *     the configured {@code targetedCfaDepth}.
   */
  public record DepthLimitedResult(
      LocalPointerKey pointerKey, CGNode node, int valueNumber, int callStringLength) {}

  private final List<DepthLimitedResult> depthLimitedResults = new ArrayList<>();

  /**
   * An allocator whose shape argument did not resolve, with the classified reason (wala/ML#735).
   * The reason triages the wala/ML#370 annotation worklist: {@link
   * TensorGenerator.ShapeUnresolutionCause#CONTENT_DEPENDENT} is a genuine annotation candidate
   * (the shape depends on runtime content), while {@link
   * TensorGenerator.ShapeUnresolutionCause#RECOVERABLE_GAP} is an analyzer precision gap that a
   * #370 annotation would wrongly paper over.
   *
   * @param pointerKey The pointer key of the allocator's ⊤-shape result (a synthetic {@code do()}
   *     {@code ReturnValueKey} for a manually anchored allocator, or a local for a caller-site
   *     one).
   * @param cause The classified reason the shape argument did not resolve.
   */
  public record ShapeAnnotationCandidate(
      PointerKey pointerKey, TensorGenerator.ShapeUnresolutionCause cause) {}

  private final List<ShapeAnnotationCandidate> shapeAnnotationCandidates = new ArrayList<>();

  /**
   * Identifies the dataflow sources for tensor analysis.
   *
   * @param builder The {@link PropagationCallGraphBuilder} containing analysis information.
   * @param dataflow The graph of {@link PointsToSetVariable}s representing the pointer analysis
   *     system's constraint graph, where nodes are variables (points-to sets) and edges represent
   *     data flow.
   * @return A {@link Set} of {@link PointsToSetVariable}s that are considered tensor dataflow
   *     sources.
   */
  private static Set<PointsToSetVariable> getDataflowSources(
      PropagationCallGraphBuilder builder, Graph<PointsToSetVariable> dataflow) {
    Set<PointsToSetVariable> sources = HashSetFactory.make();
    CallGraph callGraph = builder.getCallGraph();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (PointsToSetVariable src : dataflow) {
      PointerKey k = src.getPointerKey();

      if (k instanceof LocalPointerKey) {
        LocalPointerKey kk = (LocalPointerKey) k;
        int vn = kk.getValueNumber();
        CGNode localPointerKeyNode = kk.getNode();
        DefUse du = localPointerKeyNode.getDU();
        SSAInstruction inst = du.getDef(vn);

        if (inst instanceof SSAAbstractInvokeInstruction) {
          // We potentially have a function call that generates a tensor.
          SSAAbstractInvokeInstruction ni = (SSAAbstractInvokeInstruction) inst;
          processInstruction(
              builder, ni, du, localPointerKeyNode, src, vn, sources, pointerAnalysis);
        } else if (inst instanceof SSABinaryOpInstruction) {
          // Binary operations (e.g. +, *) on tensors are also sources.
          sources.add(src);
          LOGGER.fine("Added dataflow source from binary op: " + describe(src) + ".");
        } else if (inst instanceof EachElementGetInstruction) {
          // We are potentially pulling a tensor out of a tensor iterable.
          EachElementGetInstruction eachElementGetInstruction = (EachElementGetInstruction) inst;

          // Don't add the source if the container has elements in it. In that case, we want to add
          // the individual elements themselves as sources instead.
          if (definitionIsNonScalar(eachElementGetInstruction, du))
            LOGGER.fine(
                "Definition of instruction: "
                    + eachElementGetInstruction
                    + " is non-scalar. Skipping...");
          else {
            LOGGER.fine(
                "Definition of instruction: "
                    + eachElementGetInstruction
                    + " is scalar. Processing...");

            // Find the potential tensor iterable definition.
            processInstruction(
                eachElementGetInstruction,
                du,
                localPointerKeyNode,
                src,
                sources,
                callGraph,
                pointerAnalysis);
          }
        } else if (inst instanceof PythonPropertyRead) {
          // We are potentially pulling a tensor out of a non-scalar tensor iterable.
          PythonPropertyRead propertyRead = (PythonPropertyRead) inst;

          // The content read inside the `iter` builtin's summary (see
          // `BuiltinFunctions.iterSummary`; wala/ML#698) must not seed: the summary node is shared
          // across its call site's calling contexts (1-CFA keys on the call site), so the
          // argument's points-to set unions every caller's dataset, and a seed here injects that
          // union into the shared iterator's content field, poisoning every context's `next`
          // result. The element type is instead recovered context-sensitively at each `next`
          // result, by chasing `iter`'s argument in the caller's frame.
          if (localPointerKeyNode
              .getMethod()
              .getReference()
              .getDeclaringClass()
              .equals(PythonTypes.ITER_BUILTIN)) {
            LOGGER.fine(
                () ->
                    "Skipping seeding of content read: "
                        + describe(src)
                        + " in the shared iter summary.");
            continue;
          }

          // Find the potential tensor iterable definition.
          int objectRef = propertyRead.getObjectRef();
          SSAInstruction def = du.getDef(objectRef);

          if (def == null)
            // definition is unavailable from the local DefUse. Use interprocedural analysis using
            // the PA.
            processInstructionInterprocedurally(
                propertyRead, objectRef, localPointerKeyNode, src, sources, pointerAnalysis);
          else if (def instanceof EachElementGetInstruction
              || def instanceof PythonPropertyRead
              || def instanceof PythonInvokeInstruction) {
            boolean added = false;

            // When the iterated object itself came from a container read (its def is a property
            // read, e.g. `d = [ds, other][0]`), it may point directly to a dataset. Check its
            // points-to set before the def-chain walk below, which would otherwise climb to the
            // container and miss the dataset. Gated to dataset instances inside, and only on this
            // container-read shape, so attribute reads on a direct dataset are unaffected.
            // wala/ML#648.
            if (def instanceof PythonPropertyRead) {
              added =
                  processInstructionInterprocedurally(
                      propertyRead, objectRef, localPointerKeyNode, src, sources, pointerAnalysis);
            }

            // we may be invoking `next()` on a dataset.
            if (!added
                && def instanceof SSAAbstractInvokeInstruction
                && def.getNumberOfUses() > 1) {
              SSAAbstractInvokeInstruction invokeInstruction = (SSAAbstractInvokeInstruction) def;
              added =
                  processInstruction(
                      builder,
                      invokeInstruction,
                      du,
                      localPointerKeyNode,
                      src,
                      vn,
                      sources,
                      pointerAnalysis);
            }

            if (!added)
              processInstruction(
                  def, du, localPointerKeyNode, src, sources, callGraph, pointerAnalysis);
          }
        }
      }
    }
    return sources;
  }

  /**
   * Processes the given {@link SSAAbstractInvokeInstruction}, adding the given {@link PointsToSetVariable} to the given {@link Set} of {@link PointsToSetVariable}s as a dataflow source if the given {@link SSAAbstractInvokeInstruction} results in a tensor value.
   *
   * @param instruction The {@link SSAAbstractInvokeInstruction} to consider.
   * @param du The {@link DefUse} for the given {@link SSAAbstractInvokeInstruction}.
   * @param node The {@link CGNode} containing the given {@link SSAAbstractInvokeInstruction}.
   * @param src The {@link PointsToSetVariable} to add to the given {@link Set} of {@link PointsToSetVariable}s if there a tensor flows from the given {@link SSAAbstractInvokeInstruction.
   * @param vn The value number in the given {@link CGNode} corresponding to the given {@link PointsToSetVariable}.
   * @param sources The {@link Set} of {@link PointsToSetVariable}s representing tensor dataflow sources.
   * @param pointerAnalysis The {@link PointerAnalysis} for the given {@link CGNode}.
   * @return True iff given the source was added to the set.
   */
  private static boolean processInstruction(
      PropagationCallGraphBuilder builder,
      SSAAbstractInvokeInstruction instruction,
      DefUse du,
      CGNode node,
      PointsToSetVariable src,
      int vn,
      Set<PointsToSetVariable> sources,
      PointerAnalysis<InstanceKey> pointerAnalysis) {
    boolean ret = false;

    // don't consider exceptions as a data source.
    if (instruction.getException() != vn) {
      String methodName = instruction.getCallSite().getDeclaredTarget().getName().toString();
      if (methodName.equals(TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME)
          || methodName.equals(DO_METHOD_NAME)) {
        try {
          TensorGenerator generator = getGenerator(src, builder);
          LOGGER.fine(
              () -> "Found tensor generator: " + generator + " for source: " + describe(src) + ".");
          sources.add(src);
          LOGGER.fine("Added dataflow source from tensor generator: " + describe(src) + ".");
          ret = true;
        } catch (IllegalArgumentException e) {
          // not a tensor source.
          LOGGER.log(Level.FINE, "Not a tensor source: " + methodName, e);
        }
      } else if (instruction.getNumberOfUses() > 1) {
        // Get the invoked function from the PA.
        int target = instruction.getUse(0);
        PointerKey targetKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, target);

        for (InstanceKey ik : pointerAnalysis.getPointsToSet(targetKey)) {
          if (ik instanceof ConcreteTypeKey) {
            ConcreteTypeKey ctk = (ConcreteTypeKey) ik;
            IClass type = ctk.type();
            TypeReference reference = type.getReference();

            if (reference.equals(NEXT.getDeclaringClass())) {
              // it's a call to `next()`. Look up the iterator definition.
              int iterator = instruction.getUse(1);
              SSAInstruction iteratorDef = du.getDef(iterator);

              // Let's see if the iterator is over a tensor dataset. First, check the iterator
              // for a dataset source. `iter()` returns a fresh iterator that does not carry a
              // dataset's element flow (see `iterSummary`; wala/ML#698), so a dataset source is
              // recovered by chasing `iter()`'s argument below.
              if (iteratorDef != null && iteratorDef.getNumberOfUses() > 1) {
                boolean added =
                    processInstructionInterprocedurally(
                        iteratorDef, iteratorDef.getDef(), node, src, sources, pointerAnalysis);

                ret |= added;

                if (!added && iteratorDef instanceof SSAAbstractInvokeInstruction) {
                  // It may be a call to `iter()`. Get the argument.
                  int iterArg = iteratorDef.getUse(1);
                  ret |=
                      processInstructionInterprocedurally(
                          iteratorDef, iterArg, node, src, sources, pointerAnalysis);
                }

                // When the iterator definition is not a direct call (e.g., a property read on
                // a user-defined class field like `c.some_iter`), chase the PA to find iterator
                // allocations and check their creator's iter() argument for a dataset.
                if (!added && !ret) {
                  PointerKey iterPK =
                      pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, iterator);
                  for (InstanceKey iterIK : pointerAnalysis.getPointsToSet(iterPK)) {
                    if (ret) break;
                    AllocationSiteInNode asin;
                    try {
                      asin = getAllocationSiteInNode(iterIK);
                    } catch (IllegalArgumentException e) {
                      continue;
                    }
                    if (asin != null) {
                      CGNode creatorNode = asin.getNode();
                      if (creatorNode
                          .getMethod()
                          .getReference()
                          .getDeclaringClass()
                          .equals(PythonTypes.ITER_BUILTIN)) {
                        int iterArgVn = 2;
                        ret |=
                            processInstructionInterprocedurally(
                                iteratorDef, iterArgVn, creatorNode, src, sources, pointerAnalysis);
                      }
                    }
                  }
                }
              } else
                // Use the original instruction to chase the iterator directly for a dataset
                // source (see `iterSummary`; wala/ML#698).
                ret |=
                    processInstructionInterprocedurally(
                        instruction, iterator, node, src, sources, pointerAnalysis);
            }
          }
        }
      }
    }

    return ret;
  }

  /**
   * Processes the given {@link SSAInstruction} to decide if the given {@link PointsToSetVariable}
   * is added to the given {@link Set} of {@link PointsToSetVariable}s as tensor dataflow sources.
   *
   * @param instruction The {@link SSAInstruction} to process.
   * @param du The {@link DefUse} corresponding to the given {@link SSAInstruction}.
   * @param node The {@link CGNode} containing the given {@link SSAInstruction}.
   * @param src The {@link PointsToSetVariable} under question as to whether it should be considered
   *     a tensor dataflow source.
   * @param sources The {@link Set} of tensor dataflow sources.
   * @param callGraph The {@link CallGraph} containing the given {@link SSAInstruction}.
   * @param pointerAnalysis The {@link PointerAnalysis} corresponding to the given {@link
   *     CallGraph}.
   * @return True iff the given {@link PointsToSetVariable} was added to the given {@link Set} of
   *     {@link PointsToSetVariable} dataflow sources.
   */
  private static boolean processInstruction(
      SSAInstruction instruction,
      DefUse du,
      CGNode node,
      PointsToSetVariable src,
      Set<PointsToSetVariable> sources,
      CallGraph callGraph,
      PointerAnalysis<InstanceKey> pointerAnalysis) {
    return processInstruction(
        instruction, du, node, src, sources, callGraph, pointerAnalysis, newHashSet());
  }

  /**
   * Processes the given {@link SSAInstruction} to decide if the given {@link PointsToSetVariable}
   * is added to the given {@link Set} of {@link PointsToSetVariable}s as tensor dataflow sources.
   *
   * @param instruction The {@link SSAInstruction} to process.
   * @param du The {@link DefUse} corresponding to the given {@link SSAInstruction}.
   * @param node The {@link CGNode} containing the given {@link SSAInstruction}.
   * @param src The {@link PointsToSetVariable} under question as to whether it should be considered
   *     a tensor dataflow source.
   * @param sources The {@link Set} of tensor dataflow sources.
   * @param callGraph The {@link CallGraph} containing the given {@link SSAInstruction}.
   * @param pointerAnalysis The {@link PointerAnalysis} corresponding to the given {@link
   *     CallGraph}.
   * @param seen A {@link Set} of previously processed {@link SSAInstruction}.
   * @return True iff the given {@link PointsToSetVariable} was added to the given {@link Set} of
   *     {@link PointsToSetVariable} dataflow sources.
   */
  private static boolean processInstruction(
      SSAInstruction instruction,
      DefUse du,
      CGNode node,
      PointsToSetVariable src,
      Set<PointsToSetVariable> sources,
      CallGraph callGraph,
      PointerAnalysis<InstanceKey> pointerAnalysis,
      Set<SSAInstruction> seen) {
    if (seen.contains(instruction))
      LOGGER.fine(() -> "Skipping instruction: " + instruction + ". We've seen it before.");
    else {
      LOGGER.fine(() -> "Processing instruction: " + instruction + ".");
      seen.add(instruction);

      if (instruction != null && instruction.getNumberOfUses() > 0) {
        int use = instruction.getUse(0);
        SSAInstruction def = du.getDef(use);

        // First try intraprocedural analysis.
        if (definesTensorIterable(def, node, callGraph, pointerAnalysis)) {
          sources.add(src);
          LOGGER.fine("Added dataflow source from tensor iterable: " + describe(src) + ".");
          return true;
        } else {
          // Use interprocedural analysis using the PA.
          boolean added =
              processInstructionInterprocedurally(
                  instruction, use, node, src, sources, pointerAnalysis);

          if (added) return true;
          else
            // keep going up.
            return processInstruction(
                def, du, node, src, sources, callGraph, pointerAnalysis, seen);
        }
      }
    }

    return false;
  }

  /**
   * Similar to processInstruction but does so using the given {@link PointerAnalysis}.
   *
   * @param instruction The {@link SSAInstruction} to be processed.
   * @param use The use in the {@link Instruction} to analyze.
   * @param node The {@link CGNode} containing the given {@link SSAInstruction}.
   * @param src The {@link PointsToSetVariable} being decided upon whether it should be considered
   *     as a tensor dataflow source.
   * @param sources The {@link Set} of all tensor dataflow sources, i.e., {@link
   *     PointsToSetVariable}s.
   * @param pointerAnalysis The {@link PointerAnalysis} built from the given {@link CGNode}'s {@link
   *     CallGraph}.
   * @return True iff the given {@link PointsToSetVariable} was added to the given set of tensor
   *     dataflow sources, i.e., the given {@link Set} of {@link PointsToSetVariable}s.
   */
  private static boolean processInstructionInterprocedurally(
      SSAInstruction instruction,
      int use,
      CGNode node,
      PointsToSetVariable src,
      Set<PointsToSetVariable> sources,
      PointerAnalysis<InstanceKey> pointerAnalysis) {
    LOGGER.fine(
        () ->
            "Using interprocedural analysis to find potential tensor definition for use: "
                + use
                + " of instruction: "
                + instruction
                + ".");

    // Look up the use in the pointer analysis to see if it points to a dataset.
    PointerKey usePointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, use);

    for (InstanceKey ik : pointerAnalysis.getPointsToSet(usePointerKey)) {
      if (ik instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = (AllocationSiteInNode) ik;
        IClass concreteType = asin.concreteType();
        TypeReference reference = concreteType.getReference();

        if ((reference.equals(DATASET)
                || reference.getName().toString().startsWith(DATA_PACKAGE_PREFIX))
            && isDatasetTensorElement(src, use, pointerAnalysis)) {
          sources.add(src);
          LOGGER.fine("Added dataflow source from tensor dataset: " + describe(src) + ".");
          return true;
        }
      }
    }

    return false;
  }

  /**
   * Returns true iff the given {@link PointsToSetVariable} refers to a tensor dataset element of
   * the dataset defined by the given value number in the its associated {@link CGNode}.
   *
   * @param variable The {@link PointsToSetVariable} to consider.
   * @param val The value in the given {@link CGNode} representing the tensor dataset.
   * @param pointerAnalysis The {@link PointerAnalysis} that includes points-to information for the
   *     given {@link CGNode}.
   * @return True iff src refers to a tensor dataset element defined by the dataset represented by
   *     val in the node associated with src.
   */
  private static boolean isDatasetTensorElement(
      PointsToSetVariable variable, int val, PointerAnalysis<InstanceKey> pointerAnalysis) {
    if (variable.getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey localPointerKey = (LocalPointerKey) variable.getPointerKey();
      CGNode node = localPointerKey.getNode();
      SSAInstruction def = node.getDU().getDef(val);

      if (def instanceof PythonInvokeInstruction) {
        PythonInvokeInstruction invokeInstruction = (PythonInvokeInstruction) def;

        // Check whether we are calling enumerate(), as that returns a tuple.
        // Get the invoked function.
        int invocationUse = invokeInstruction.getUse(0);

        PointerKey invocationUsePointerKey =
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, invocationUse);

        for (InstanceKey functionInstance :
            pointerAnalysis.getPointsToSet(invocationUsePointerKey)) {
          if (functionInstance instanceof ConcreteTypeKey) {
            ConcreteTypeKey typeKey = (ConcreteTypeKey) functionInstance;
            IClass type = typeKey.type();
            TypeReference typeReference = type.getReference();

            if (typeReference.equals(ENUMERATE.getDeclaringClass())) {
              // it's a call to enumerate(), where the returned value is an iterator over
              // tuples. Each tuple consists of the enumeration number and the dataset
              // element. Check that we are not looking at the enumeration number.

              PythonPropertyRead srcDef =
                  (PythonPropertyRead)
                      node.getDU()
                          .getDef(((LocalPointerKey) variable.getPointerKey()).getValueNumber());

              // What does the member reference point to?
              PointerKey memberRefPointerKey =
                  pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, srcDef.getMemberRef());

              for (InstanceKey memberInstance :
                  pointerAnalysis.getPointsToSet(memberRefPointerKey)) {
                ConstantKey<?> constant = (ConstantKey<?>) memberInstance;
                Object value = constant.getValue();

                // if it's the first tuple element.
                if (value.equals(0)) {
                  // Now that we know it's the first tuple element, we now need to know whether it's
                  // the first tuple, i.e., the one returned by enumerate.
                  // To do that, we examine the object being referenced on the RHS.

                  SSAInstruction objRefDef = node.getDU().getDef(srcDef.getObjectRef());

                  LOGGER.finest(
                      () ->
                          "objRefDef is: "
                              + objRefDef.getClass().getName()
                              + " with use 0: "
                              + (objRefDef.getNumberOfUses() > 0 ? objRefDef.getUse(0) : "N/A")
                              + " and val: "
                              + val);

                  // If the object being read is that of the dataset, we know that this is the first
                  // tuple read of the result of enumerate() on the dataset.
                  if (objRefDef instanceof PythonPropertyRead
                      && ((PythonPropertyRead) objRefDef).getObjectRef() == val) return false;

                  // In Python iteration, the object being read may be an EachElementGetInstruction.
                  if (objRefDef instanceof EachElementGetInstruction && objRefDef.getUse(0) == val)
                    return false;
                }
              }
            }
          }
        }
      }
    }

    return true;
  }

  /**
   * True iff the given {@link SSAInstruction} constitutes individual elements.
   *
   * @param instruction The {@link SSAInstruction} in question.
   * @param du The {@link DefUse} for the containing {@link CGNode}.
   * @return True iff the definition of the given {@link EachElementGetInstruction} is non-scalar.
   */
  private static boolean definitionIsNonScalar(SSAInstruction instruction, DefUse du) {
    int def = instruction.getDef();
    LOGGER.fine("Processing definition: " + def + " of instruction: " + instruction + ".");

    int numberOfUses = du.getNumberOfUses(def);
    LOGGER.fine(
        "Definition: "
            + def
            + " of instruction: "
            + instruction
            + " has "
            + numberOfUses
            + " uses.");

    for (Iterator<SSAInstruction> uses = du.getUses(def); uses.hasNext(); ) {
      SSAInstruction useInstruction = uses.next();
      LOGGER.fine("Processing use: " + useInstruction + ".");

      if (useInstruction instanceof PythonPropertyRead) {
        PythonPropertyRead read = (PythonPropertyRead) useInstruction;
        LOGGER.fine("Found property read use: " + read + ".");

        // if the definition appears on the LHS of the read.
        if (read.getObjectRef() == def) return true;
      }
    }
    return false;
  }

  /**
   * Returns true iff the given {@link SSAInstruction} defines an iterable of tensors.
   *
   * @param instruction The {@link SSAInstruction} in question.
   * @param node The {@link CGNode} of the function containing the given {@link SSAInstruction}.
   * @param callGraph The {@link CallGraph} that includes a node corresponding to the given {@link
   *     SSAInstruction}.
   * @param pointerAnalysis The {@link PointerAnalysis} built from the given {@link CallGraph}.
   * @return True iff the given {@link SSAInstruction} defines an iterable over tensors.
   */
  private static boolean definesTensorIterable(
      SSAInstruction instruction,
      CGNode node,
      CallGraph callGraph,
      PointerAnalysis<InstanceKey> pointerAnalysis) {
    if (instruction instanceof SSAAbstractInvokeInstruction) {
      SSAAbstractInvokeInstruction invocationInstruction =
          (SSAAbstractInvokeInstruction) instruction;

      LOGGER.fine(() -> "definesTensorIterable checking instruction: " + invocationInstruction);

      int defVn = invocationInstruction.getDef();

      if (defVn >= 0) {
        PointerKey defPointerKey =
            pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, defVn);
        OrdinalSet<InstanceKey> defPointsToSet = pointerAnalysis.getPointsToSet(defPointerKey);

        for (InstanceKey ik : defPointsToSet) {
          AllocationSiteInNode asin = getAllocationSiteInNode(ik);

          if (asin != null) {
            TypeReference reference = asin.concreteType().getReference();

            if (reference.getName().toString().startsWith(DATA_PACKAGE_PREFIX)) {
              LOGGER.fine(
                  () ->
                      "Instruction: "
                          + instruction
                          + " defines a tensor iterable of type: "
                          + reference
                          + ".");
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  @FunctionalInterface
  interface SourceCallHandler {
    void handleCall(CGNode src, SSAAbstractInvokeInstruction call);
  }

  private void getSourceCalls(
      MethodReference op, PropagationCallGraphBuilder builder, SourceCallHandler handler) {
    for (CGNode n : builder.getCallGraph()) {
      if (n.getMethod().getReference().equals(op)) {
        for (Iterator<CGNode> srcs = builder.getCallGraph().getPredNodes(n); srcs.hasNext(); ) {
          CGNode src = srcs.next();
          for (Iterator<CallSiteReference> sites = builder.getCallGraph().getPossibleSites(src, n);
              sites.hasNext(); ) {
            CallSiteReference site = sites.next();
            for (SSAAbstractInvokeInstruction call : src.getIR().getCalls(site)) {
              handler.handleCall(src, call);
            }
          }
        }
      }
    }
  }

  /**
   * Pointwise merge of the two literal-shape pin candidates (wala/ML#713): the generator-side type
   * and the {@code TensorType.shapeArg} type approximate the same shape, so each dimension takes
   * the more useful of the two. Numeric wins, and disagreeing numeric dimensions degrade to dynamic
   * rather than trusting either. Between the unresolved kinds, symbolic wins over dynamic only when
   * the generator resolved no numeric dimension at all: {@code SetShapeOp}'s reshape transfer later
   * fills a lone symbolic dimension from the input's element count (the {@code np.prod(...,
   * axis=0)} pin owes its precision to that), whereas when the generator did resolve structure, its
   * seeded member already carries it and a symbolic-flavored pin would only add a duplicate union
   * member. An unknown-rank generator result yields the {@code shapeArg} type, a rank mismatch
   * yields the generator type, and the cell type prefers the generator's unless it is unknown.
   *
   * @param generatorType The generator-side pin candidate.
   * @param shapeArgType The {@code shapeArg} pin candidate.
   * @return The merged pin type.
   */
  private static TensorType mergePinnedTypes(TensorType generatorType, TensorType shapeArgType) {
    LOGGER.fine(
        () -> "mergePinnedTypes: generator=" + generatorType + ", shapeArg=" + shapeArgType);
    List<Dimension<?>> generatorDims = generatorType.getDims();
    List<Dimension<?>> shapeArgDims = shapeArgType.getDims();
    if (generatorDims == null) return shapeArgType;
    if (shapeArgDims == null || shapeArgDims.size() != generatorDims.size()) return generatorType;

    boolean generatorHasNumeric = generatorDims.stream().anyMatch(d -> d instanceof NumericDim);
    List<Dimension<?>> merged = new ArrayList<>(generatorDims.size());
    for (int i = 0; i < generatorDims.size(); i++) {
      Dimension<?> fromGenerator = generatorDims.get(i);
      Dimension<?> fromShapeArg = shapeArgDims.get(i);
      if (fromGenerator instanceof NumericDim && fromShapeArg instanceof NumericDim)
        // Two concrete computations disagreeing means the size is fixed but the analysis cannot
        // tell which value holds — an unresolved size, not a runtime-varying one (wala/ML#721).
        merged.add(fromGenerator.equals(fromShapeArg) ? fromGenerator : UnresolvedDim.INSTANCE);
      else if (fromGenerator instanceof NumericDim) merged.add(fromGenerator);
      else if (fromShapeArg instanceof NumericDim) merged.add(fromShapeArg);
      else if (!generatorHasNumeric && fromShapeArg instanceof SymbolicDim)
        merged.add(fromShapeArg);
      else merged.add(fromGenerator);
    }

    String cellType =
        generatorType.getCellType() == null
                || TensorFlowTypes.DType.UNKNOWN
                    .name()
                    .toLowerCase(java.util.Locale.ROOT)
                    .equals(generatorType.getCellType())
            ? shapeArgType.getCellType()
            : generatorType.getCellType();
    return new TensorType(cellType, merged);
  }

  private Map<PointsToSetVariable, Set<TensorType>> getShapeSourceCalls(
      MethodReference op, PropagationCallGraphBuilder builder, int param) {
    Map<PointsToSetVariable, Set<TensorType>> targets = HashMapFactory.make();
    getSourceCalls(
        op,
        builder,
        (CGNode src, SSAAbstractInvokeInstruction call) -> {
          if (call.getNumberOfUses() > param) {
            // `TensorType.shapeArg` reads the shape operand's element writes, so it is only
            // meaningful for a literal container construction (a `new` list/tuple). A shape
            // vector (`t.shape.as_list()[-2:]` and friends) is resolved precisely by the
            // generator-side provenance walk, so pinning would only pollute its result — skip it
            // (wala/ML#703). Any other opaque operand — a `tf.shape(y)` tensor, a computed list,
            // or a parameter — has no element writes, and `shapeArg` would fabricate a scalar
            // type for it; pin a tensor of unknown rank instead, which keeps the result
            // tensor-classified without asserting a wrong shape.
            int shapeVn = call.getUse(param);
            SSAInstruction shapeDef = src.getDU().getDef(shapeVn);
            boolean literalContainer =
                shapeDef instanceof SSANewInstruction
                    && (((SSANewInstruction) shapeDef)
                            .getNewSite()
                            .getDeclaredType()
                            .equals(PythonTypes.list)
                        || ((SSANewInstruction) shapeDef)
                            .getNewSite()
                            .getDeclaredType()
                            .equals(PythonTypes.tuple));
            PointerKey defKey =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(src, call.getDef());
            // Materializing an implicitly-represented key would make WALA dump the entire call
            // graph's IR (an unconditional debug print), so skip it; implicit keys carry no
            // explicit dataflow variable to pin (wala/ML#573).
            if (builder.getPropagationSystem().isImplicit(defKey)) return;
            PointsToSetVariable defVariable =
                builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
            // A chain-classified operand is the provenance walk's territory only when the
            // generator side actually types the result: any nonempty generator result already
            // tensor-classifies the destination (a precise resolution the pin would only pollute,
            // wala/ML#703, wala/ML#706; a dtype-only member says what the pin would say). The pin's
            // unique value is the chain the generator resolves to nothing at all, so only that
            // falls through to the unknown-rank pin below; the bare structural skip withdrew the
            // pin there too and starved consumers whose only tensor evidence it was (wala/ML#765).
            if (TensorGenerator.isShapeVectorChain(builder, src, shapeVn)) {
              Set<TensorType> generatorTypes = this.getTensorTypes(defVariable, builder);
              if (generatorTypes != null && !generatorTypes.isEmpty()) return;
            }
            Set<TensorType> pinned;
            if (literalContainer) {
              // Merge the generator-side computation into the pinned type so the pin agrees with
              // the seeded result instead of contributing a weaker parallel member (wala/ML#713).
              // Neither path subsumes the other: `shapeArg` folds literal source text through the
              // embedded interpreter (e.g. an `np.prod(..., axis=0)` the provenance walk refuses),
              // while the generator side also resolves field reads, shape-vector subscripts, and
              // `build`-computed attributes; the pointwise merge keeps the more precise dimension
              // from each. A multi-member generator result merges per member, so the pin carries
              // one member per resolved shape possibility instead of discarding the generator's
              // precision for the weaker interpreter type (wala/ML#748).
              TensorType shapeArgType = TensorType.shapeArg(src, shapeVn, builder);
              Set<TensorType> generatorTypes = this.getTensorTypes(defVariable, builder);
              if (generatorTypes != null && !generatorTypes.isEmpty()) {
                pinned = HashSetFactory.make();
                for (TensorType generatorType : generatorTypes)
                  pinned.add(mergePinnedTypes(generatorType, shapeArgType));
              } else pinned = Collections.singleton(shapeArgType);
            } else
              pinned =
                  Collections.singleton(
                      new TensorType(
                          TensorFlowTypes.DType.FLOAT32.name().toLowerCase(java.util.Locale.ROOT),
                          null));
            targets.put(defVariable, pinned);
          }
        });
    return targets;
  }

  /**
   * Finds {@code x.set_shape(shape)} call sites by scanning the IR directly for {@link
   * PythonPropertyRead}s of the {@code set_shape} attribute followed by an invoke on the
   * property-read result. Returns a map keyed by the *receiver's* points-to-set variable (the
   * {@code x}) to the shape-arg {@link TensorType}.
   *
   * <p>The legacy {@link #getShapeSourceCalls} path resolves the {@code set_shape} method via
   * WALA's call-graph dispatch on the {@code Ltensorflow/functions/set_shape} synthetic class. That
   * dispatch only fires when the receiver has the {@code set_shape} attribute attached via the
   * {@code FixedLenFeature.do} {@code <putfield>} chain — broken when {@code tf.cast}'s {@code
   * pass_through} alias is removed (the freshly-allocated {@code Tensor} has no {@code set_shape}
   * field). See wala/ML#509.
   *
   * <p>Syntactic recognition decouples the {@code set_shape} pin from the attribute-attachment
   * chain. False positives are possible—any Python class can define a {@code set_shape} method, and
   * a user-defined class that happens to do so would otherwise tip into the tensor analysis. The
   * {@link #SET_SHAPE_RECEIVER_TYPES} whitelist is a required safeguard: receivers whose PA
   * allocation type isn't in the whitelist are skipped, with the empty-PTS allowance documented
   * inline at the containment-check site. {@code testSetShapeNonTensorReceiver} is the regression
   * guard for this filtering.
   *
   * @param builder The propagation call graph builder.
   * @return Map from receiver {@link PointsToSetVariable} to the asserted {@link TensorType}.
   */
  private Map<PointsToSetVariable, TensorType> getSetShapeCallsSyntactic(
      PropagationCallGraphBuilder builder) {
    Map<PointsToSetVariable, TensorType> targets = HashMapFactory.make();
    for (CGNode caller : builder.getCallGraph()) {
      IR ir = caller.getIR();
      if (ir == null) continue;
      SymbolTable st = ir.getSymbolTable();
      DefUse du = caller.getDU();
      // Walk every SSA instruction in the caller's IR. We are looking for the two-instruction
      // pattern that a Python attribute-call lowers into: a `PythonPropertyRead` defining the
      // call target, followed by a `PythonInvokeInstruction` consuming that target.
      for (Iterator<SSAInstruction> it = ir.iterateAllInstructions(); it.hasNext(); ) {
        SSAInstruction inst = it.next();
        if (!(inst instanceof PythonInvokeInstruction)) continue;
        PythonInvokeInstruction call = (PythonInvokeInstruction) inst;
        // Python invoke uses are `[callTarget, arg1, arg2, ...]`; there is no explicit receiver
        // slot in the invoke. `use(0)` is the call target (the `PythonPropertyRead` result), and
        // `use(1)` is the first positional argument — for `x.set_shape(shape)`, that's the shape.
        // The receiver `x` is recovered separately below from the property-read's `objectRef`.
        // We need at least the call target plus one argument to identify an `x.set_shape(shape)`
        // site, so skip invokes with fewer than 2 uses.
        if (call.getNumberOfUses() < 2) continue;
        // Resolve the call target's defining instruction. If the target wasn't produced by a
        // property read, this isn't an attribute call (could be a function-typed local, a
        // closure, etc.) and we ignore it.
        SSAInstruction targetDef = du.getDef(call.getUse(0));
        if (!(targetDef instanceof PythonPropertyRead)) continue;
        PythonPropertyRead prop = (PythonPropertyRead) targetDef;
        // The property-read's `memberRef` value-number identifies the attribute name. We only
        // pin on the literal string `"set_shape"`; dynamic attribute names (rare in TF source)
        // can't be matched syntactically and are out of scope.
        int memberVn = prop.getMemberRef();
        if (!st.isStringConstant(memberVn)) continue;
        if (!SET_SHAPE_ATTRIBUTE.equals(st.getStringValue(memberVn))) continue;
        // The property-read's `objectRef` value-number is the receiver — the `x` in
        // `x.set_shape(shape)`. That's what we pin to the asserted shape.
        int receiverVn = prop.getObjectRef();
        PointerKey receiverKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, receiverVn);
        // Containment check: only pin when the receiver's PA allocation type is one that
        // supports `set_shape` in real TF. See `SET_SHAPE_RECEIVER_TYPES` for the whitelist.
        //
        // Receivers with empty PTS are still pinned. Empty PTS occurs when a TF API in the call
        // chain upstream of the `set_shape` site has no XML summary (or the summary doesn't
        // model its return as a fresh allocation) — see CLAUDE.md, "PTS vs `TensorTypeAnalysis`".
        // The syntactic `set_shape` pattern itself is strong evidence in the TF source files
        // this analyzer targets, so pinning on empty PTS is correct-on-balance.
        OrdinalSet<InstanceKey> recvPts = builder.getPointerAnalysis().getPointsToSet(receiverKey);
        boolean receiverEligible = recvPts == null || recvPts.isEmpty();
        if (!receiverEligible) {
          for (InstanceKey ik : recvPts) {
            // Use the `getAllocationSiteInNode` helper to unwrap `ScopeMappingInstanceKey` /
            // `ConstantKey` wrappers — mirrors the pattern used at lines 386 and 714 of this
            // file and avoids missing tensor receivers that flow through closures or constants.
            AllocationSiteInNode asin = getAllocationSiteInNode(ik);
            if (asin != null
                && SET_SHAPE_RECEIVER_TYPES.contains(asin.concreteType().getReference())) {
              receiverEligible = true;
              break;
            }
          }
        }
        if (!receiverEligible) continue;
        // Skip implicitly-represented receivers: materializing one makes WALA dump the entire
        // call graph's IR via an unconditional debug print (wala/ML#573), and an implicit key has
        // no explicit dataflow variable to pin.
        if (builder.getPropagationSystem().isImplicit(receiverKey)) continue;
        targets.put(
            builder.getPropagationSystem().findOrCreatePointsToSet(receiverKey),
            TensorType.shapeArg(caller, call.getUse(1), builder));
      }
    }
    return targets;
  }

  private Set<PointsToSetVariable> getKeysDefinedByCall(
      MethodReference op, PropagationCallGraphBuilder builder) {
    Set<PointsToSetVariable> lvals = HashSetFactory.make();
    getSourceCalls(
        op,
        builder,
        (CGNode src, SSAAbstractInvokeInstruction call) -> {
          PointerKey defKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(src, call.getDef());
          // Skip implicitly-represented keys: materializing one makes WALA dump the entire call
          // graph's IR via an unconditional debug print (wala/ML#573), and an implicit key has no
          // explicit dataflow variable to track.
          if (!builder.getPropagationSystem().isImplicit(defKey))
            lvals.add(builder.getPropagationSystem().findOrCreatePointsToSet(defKey));
        });
    return lvals;
  }

  /**
   * Returns whether the given invoke is a slice <em>constructor</em>: it resolves to the {@code
   * slice} builtin and every argument is a compile-time constant (the {@code slice(None, None,
   * None)} form that subscripts like {@code x[:, 0]} compile to). The subscript-application form
   * never matches, since its first argument is the sliced tensor. See wala/ML#732.
   *
   * @param node The {@link CGNode} containing the invoke.
   * @param invoke The invoke instruction to test.
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the targets.
   * @return {@code true} iff the invoke constructs a slice object from constants.
   */
  private static boolean isConstantSliceConstructor(
      CGNode node, SSAAbstractInvokeInstruction invoke, PropagationCallGraphBuilder builder) {
    SymbolTable st = node.getIR().getSymbolTable();
    for (int i = 1; i < invoke.getNumberOfUses(); i++)
      if (!st.isConstant(invoke.getUse(i))) return false;
    for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite()))
      if (callee.getMethod().getReference().getDeclaringClass().equals(PythonTypes.SLICE_BUILTIN))
        return true;
    return false;
  }

  /**
   * Returns whether the given invoke resolves to the {@code enumerate} builtin. The declared target
   * is a generic trampoline ({@code LCodeBody}), so resolution goes through {@code
   * getPossibleTargets}, matching {@code isEnumerateFirstFieldRead}.
   *
   * @param node The {@link CGNode} containing the invoke.
   * @param invoke The invoke instruction to test.
   * @param builder The {@link PropagationCallGraphBuilder} whose call graph resolves the targets.
   * @return {@code true} iff a resolved callee is the {@code enumerate} builtin.
   */
  private static boolean isEnumerateCall(
      CGNode node, SSAAbstractInvokeInstruction invoke, PropagationCallGraphBuilder builder) {
    for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite()))
      if (callee
          .getMethod()
          .getReference()
          .getDeclaringClass()
          .equals(PythonTypes.ENUMERATE_BUILTIN)) return true;
    return false;
  }

  /**
   * Reports whether {@code v}'s defining instruction is the first-field read of the tuple yielded
   * by Python's {@code enumerate} builtin &mdash; i.e., the {@code step} slot in {@code for step, x
   * in enumerate(iterable)}. Such variables are integer indices, not tensors, even though the
   * underlying PA graph aliases their field-0 pointer key with the iterable's element type. See
   * wala/ML#409.
   *
   * <p>Structural detection, matching the dispatch pattern in {@link
   * TensorGeneratorFactory#getGenerator}:
   *
   * <ol>
   *   <li>{@code v}'s def is a {@link PythonPropertyRead} whose {@code memberRef} PTS contains the
   *       integer constant {@code 0}.
   *   <li>Its {@code objectRef}'s def is itself a {@link PythonPropertyRead} (the iterator-element
   *       fetch).
   *   <li>That inner read's {@code objectRef}'s def is an invoke of {@link
   *       PythonTypes#ENUMERATE_BUILTIN}.
   * </ol>
   *
   * @param v The candidate points-to-set variable.
   * @param builder The propagation call graph builder.
   * @return {@code true} iff {@code v} is the enumerate-first-field read.
   */
  private static boolean isEnumerateFirstFieldRead(
      PointsToSetVariable v, PropagationCallGraphBuilder builder) {
    if (!(v.getPointerKey() instanceof LocalPointerKey)) return false;
    LocalPointerKey lpk = (LocalPointerKey) v.getPointerKey();
    CGNode node = lpk.getNode();
    SSAInstruction def = node.getDU().getDef(lpk.getValueNumber());
    if (!(def instanceof PythonPropertyRead)) return false;
    PythonPropertyRead outer = (PythonPropertyRead) def;

    // Member ref must be constant 0.
    PointerKey memberKey =
        builder
            .getPointerAnalysis()
            .getHeapModel()
            .getPointerKeyForLocal(node, outer.getMemberRef());
    boolean isFirstElement = false;
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(memberKey)) {
      if (ik instanceof ConstantKey
          && Integer.valueOf(0).equals(((ConstantKey<?>) ik).getValue())) {
        isFirstElement = true;
        break;
      }
    }
    if (!isFirstElement) return false;

    // Object ref must be another PropertyRead (the iterator-element fetch).
    SSAInstruction objDef = node.getDU().getDef(outer.getObjectRef());
    if (!(objDef instanceof PythonPropertyRead)) return false;
    PythonPropertyRead inner = (PythonPropertyRead) objDef;

    // Inner's object ref must be an invoke of `enumerate`. The declared target is a generic
    // trampoline (`LCodeBody`), so resolve via `getPossibleTargets` — matching the factory's
    // dispatch pattern in `TensorGeneratorFactory.getGenerator`.
    SSAInstruction innerObjDef = node.getDU().getDef(inner.getObjectRef());
    if (!(innerObjDef instanceof PythonInvokeInstruction)) return false;
    PythonInvokeInstruction invoke = (PythonInvokeInstruction) innerObjDef;
    for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, invoke.getCallSite())) {
      if (callee
          .getMethod()
          .getReference()
          .getDeclaringClass()
          .equals(PythonTypes.ENUMERATE_BUILTIN)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public TensorTypeAnalysis performAnalysis(PropagationCallGraphBuilder builder)
      throws CancelException {
    // The wala/ML#365 worklist engine is installed for the whole analysis, not just the seeding
    // loop: post-seeding passes (origin classification, the set_shape/subscript recognizers)
    // dispatch generators whose evidence gates read shapes and dtypes, and those reads need the
    // engine's memoization and cycle convergence just as the seeds do (an unguarded read recurses
    // unboundedly through producer delegation on cyclic subjects).
    WorklistTypeResolver.install(builder);
    try {
      Graph<PointsToSetVariable> dataflow =
          SlowSparseNumberedGraph.duplicate(
              builder.getPropagationSystem().getFlowGraphIncludingImplicitConstraints());

      Set<PointsToSetVariable> sources = getDataflowSources(builder, dataflow);

      // Resolve the per-source types with the wala/ML#365 worklist engine: a single fixpoint over
      // the recorded query-dependency graph replaces the historical round-based resolution
      // (retired in Phase 3), the memo layers divert to the engine, and the seeding reads
      // stabilized values, so the results are independent of the seeding order.
      Map<PointsToSetVariable, Set<TensorType>> init = HashMapFactory.make();

      // The engine's results are order-independent; the reverse-seeds property exists so the
      // invariance is a tested property rather than a design claim (wala/ML#365, Phase 2).
      List<PointsToSetVariable> ordered = new ArrayList<>(sources);
      if (Boolean.getBoolean("ariadne.typeResolution.reverseSeeds")
          || "true".equals(System.getenv("ARIADNE_REVERSE_SEEDS"))) Collections.reverse(ordered);

      // The wala/ML#756 perturbation knob also permutes the demand roots: reversal alone leaves
      // most of the root order's multiplicity unexercised (the source set's iteration order is
      // identity-hash-seeded, so a JVM rerun is an arbitrary permutation, not a reversal), and the
      // root order decides where the evaluation first enters each dependency cycle.
      Long shuffleSeed = WorklistTypeResolver.parseCycleShuffleSeed();
      if (shuffleSeed != null) Collections.shuffle(ordered, new Random(shuffleSeed));
      for (PointsToSetVariable v : ordered) init.put(v, getTensorTypes(v, builder));

      // Second pass: a seed materialized early in the first pass predates the constraints
      // later roots add, and the engine's state converges monotonically across the whole
      // loop — the early snapshot can carry half-resolved members (e.g. an unknown-dtype
      // twin of a member the converged state types fully). After the first pass every query
      // is evaluated, so this pass adds no keys, edges, or evaluations; it only re-reads
      // each seed's composition against the final fixpoint.
      for (PointsToSetVariable v : ordered) init.put(v, getTensorTypes(v, builder));

      // Seed each source's producing library beside its types (wala/ML#724). Origins ride the
      // same dataflow edges as the types, so the seeds are the only generator-side contribution;
      // everything downstream (operators, merges) is the analysis's union.
      Map<PointsToSetVariable, Set<TensorOrigin>> initOrigins = HashMapFactory.make();
      for (PointsToSetVariable v : sources) {
        Set<TensorType> types = init.get(v);
        if (types != null && types.isEmpty()) continue; // ⊥: not a tensor, no origin to record
        Set<TensorOrigin> origins = getTensorOrigins(v, builder);
        if (!origins.isEmpty()) initOrigins.put(v, origins);
      }

      Map<PointsToSetVariable, Set<TensorType>> placeholders =
          handleShapeSourceOp(builder, dataflow, placeholder, 2);
      LOGGER.fine("Placeholders: " + placeholders);

      for (Map.Entry<PointsToSetVariable, Set<TensorType>> e : placeholders.entrySet()) {
        init.put(e.getKey(), e.getValue());
        // `tf.compat.v1.placeholder` is a TensorFlow API (wala/ML#724).
        initOrigins.put(e.getKey(), EnumSet.of(TensorOrigin.TENSORFLOW));
      }

      // wala/ML#509: recognize `x.set_shape(s)` via IR scanning rather than call-graph dispatch on
      // the `Ltensorflow/functions/set_shape` synthetic class. The legacy dispatch path requires
      // the receiver to have the `set_shape` attribute attached (via FixedLenFeature.do's
      // <putfield>) and breaks when the cast pass_through alias is removed.
      Map<PointsToSetVariable, TensorType> setCalls = getSetShapeCallsSyntactic(builder);

      // wala/ML#509: `set_shape` is a user-supplied OVERRIDE of any per-op-generator init seed on
      // the receiver. Remove receivers from `init` so the SetShapeOp edge transfer is the sole
      // source of state for those variables; otherwise the meet-time union re-introduces the
      // generator-seeded type (e.g., Cast generator's (?, float32) on the cast-result variable).
      for (PointsToSetVariable recv : setCalls.keySet()) {
        init.remove(recv);
        // The SetShapeOp edge transfer pins the receiver's origin too (wala/ML#724).
        initOrigins.remove(recv);
      }

      // Route subscript results through `setCalls` so `TensorTypeAnalysis`'s edge-transfer replaces
      // predecessor types rather than unioning them — the receiver's pre-subscript shape would
      // otherwise leak in via the PA assignment graph (wala/ML#405).
      for (PointsToSetVariable src : sources) {
        if (!(src.getPointerKey() instanceof LocalPointerKey)) continue;
        TensorGenerator generator;
        try {
          generator = getGenerator(src, builder);
        } catch (IllegalArgumentException e) {
          continue;
        }
        if (!(generator instanceof SliceBuiltinOperation)
            && !(generator instanceof NdarraySubscriptOperation)) continue;
        Set<TensorType> types = init.get(src);
        if (types == null || types.size() != 1) continue;
        TensorType onlyType = types.iterator().next();
        if (onlyType.getDims() == null) continue;
        setCalls.put(src, onlyType);
      }

      // A shape-constraining consumer proves operand shapes its call alone fixes: an einsum
      // equation fixes each operand's rank and shared-label extents (wala/ML#704), and a constant
      // transpose permutation fixes its input's rank (wala/ML#734). Refine — rather than pin —
      // the operand destinations, so unknown-shape members recover the proven axes while concrete
      // members pass through untouched. An operand whose call sites prove disagreeing constraints
      // is left alone.
      Map<PointsToSetVariable, List<Dimension<?>>> refinements = HashMapFactory.make();
      Set<PointsToSetVariable> conflictingRefinements = HashSetFactory.make();
      for (PointsToSetVariable src : sources) {
        TensorGenerator generator;
        try {
          generator = getGenerator(src, builder);
        } catch (IllegalArgumentException e) {
          continue;
        }
        if (!(generator instanceof OperandShapeConstraining constraining)) continue;
        for (Map.Entry<PointerKey, List<Dimension<?>>> entry :
            constraining.getOperandShapeConstraints(builder).entrySet()) {
          if (builder.getPropagationSystem().isImplicit(entry.getKey())) continue;
          PointsToSetVariable operand =
              builder.getPropagationSystem().findOrCreatePointsToSet(entry.getKey());
          List<Dimension<?>> previous = refinements.putIfAbsent(operand, entry.getValue());
          if (previous != null && !previous.equals(entry.getValue()))
            conflictingRefinements.add(operand);
        }
      }
      refinements.keySet().removeAll(conflictingRefinements);
      LOGGER.fine(() -> "wala/ML#704 einsum operand refinements: " + refinements.size());
      for (Map.Entry<PointsToSetVariable, List<Dimension<?>>> r : refinements.entrySet())
        LOGGER.fine(
            () -> "  refine: " + describe(r.getKey().getPointerKey()) + " -> " + r.getValue());

      // A generator whose shape walk fails seeds an unresolved-shape member that no later
      // evidence can remove, although the operands' types often exist in dataflow state the
      // PTS-based walks cannot see (wala/ML#736, wala/ML#682; the wala/ML#661/wala/ML#570
      // substrate). Replace such a seed with synthetic dataflow edges from the operands the
      // generator declares: the result composes from the operands' converged members per the
      // feed's kind, and the unresolved seed member is never born.
      Map<PointsToSetVariable, TensorTypeAnalysis.FeedPlan> typeFeeds = HashMapFactory.make();
      Map<PointsToSetVariable, Set<TensorType>> suppressedSeeds = HashMapFactory.make();
      Map<PointsToSetVariable, Set<TensorOrigin>> suppressedOrigins = HashMapFactory.make();
      for (PointsToSetVariable src : sources) {
        Set<TensorType> types = init.get(src);
        // Three eligible seed classes. Entirely dtype-unproven seeds: a single pure-⊤ member is
        // replaced by the generator-declared composition, and any other all-unknown-dtype seed
        // (shape-resolved members, several members, or a mix) keeps its members with only the
        // dtype fed (wala/ML#758's DTYPE_FILL, the class the einsum producer registration
        // surfaces, wala/ML#757). A seed with a PROVEN dtype and a ⊤-shape member takes the
        // mirror SHAPE_FILL: its members are retained — the shape-resolved ones verbatim — and
        // only the ⊤-shape members' dims compose from the operands, per the declared kind, so
        // the proven-dtype evidence the engine-side pin computations read is never lost (the
        // vendored Conv1d and gather probes are wala/ML#682's witnesses for wholesale
        // suppression). DTYPE_ONLY declares no operand→result shape relation, so proven-dtype
        // seeds under it stay untouched.
        if (types == null || types.isEmpty()) continue;
        boolean anyProvenDType = types.stream().anyMatch(t -> t.getDType() != DType.UNKNOWN);
        boolean anyTopShape = types.stream().anyMatch(t -> t.getDims() == null);
        TensorTypeAnalysis.FeedMode mode;
        if (!anyProvenDType)
          mode =
              types.size() == 1 && anyTopShape
                  ? TensorTypeAnalysis.FeedMode.REPLACE
                  : TensorTypeAnalysis.FeedMode.DTYPE_FILL;
        else if (anyTopShape) mode = TensorTypeAnalysis.FeedMode.SHAPE_FILL;
        else continue; // Both axes proven on every member: nothing to fill.
        TensorGenerator generator;
        try {
          generator = getGenerator(src, builder);
        } catch (IllegalArgumentException e) {
          continue;
        }
        TensorGenerator.TypeFeed feed = generator.getTypeFeed(builder);
        if (feed == null) continue;
        if (mode == TensorTypeAnalysis.FeedMode.SHAPE_FILL
            && feed.kind() == TensorGenerator.TypeFeedKind.DTYPE_ONLY) continue;
        List<PointsToSetVariable> feedSources = new ArrayList<>();
        for (PointerKey operandKey : feed.operands()) {
          if (builder.getPropagationSystem().isImplicit(operandKey)) continue;
          PointsToSetVariable operand =
              builder.getPropagationSystem().findOrCreatePointsToSet(operandKey);
          if (operand.equals(src)) continue; // A self-loop feeds nothing.
          feedSources.add(operand);
        }
        // A broadcast composes pairs, so any mode that consumes the composed shape needs both
        // operands; the other kinds need at least one located operand. Otherwise keep the seed.
        // A dtype fill always fills from any located operand, since only the dtype is borrowed.
        if (feedSources.isEmpty()
            || (mode != TensorTypeAnalysis.FeedMode.DTYPE_FILL
                && feed.kind() == TensorGenerator.TypeFeedKind.BROADCAST
                && feedSources.size() != 2)) continue;
        for (PointsToSetVariable operand : feedSources) {
          if (!dataflow.containsNode(operand)) dataflow.addNode(operand);
          if (!dataflow.hasEdge(operand, src)) dataflow.addEdge(operand, src);
        }
        typeFeeds.put(
            src,
            new TensorTypeAnalysis.FeedPlan(
                feed.kind(),
                mode,
                feedSources,
                mode == TensorTypeAnalysis.FeedMode.REPLACE ? Collections.emptySet() : types));
        suppressedSeeds.put(src, types);
        Set<TensorOrigin> origins = initOrigins.get(src);
        if (origins != null) suppressedOrigins.put(src, origins);
        init.remove(src);
        initOrigins.remove(src);
      }
      LOGGER.fine(() -> "wala/ML#736 type feeds: " + typeFeeds.size());
      for (Map.Entry<PointsToSetVariable, TensorTypeAnalysis.FeedPlan> f : typeFeeds.entrySet())
        LOGGER.fine(
            () ->
                "  feed "
                    + f.getValue().mode()
                    + "/"
                    + f.getValue().kind()
                    + ": "
                    + describe(f.getKey().getPointerKey()));

      Map<PointsToSetVariable, Set<TensorType>> shapeOps = HashMapFactory.make();
      shapeOps.putAll(handleShapeSourceOp(builder, dataflow, reshape, 2));

      handlePassThroughOp(builder, dataflow, convert_to_tensor, 1);

      Set<PointsToSetVariable> conv2ds = getKeysDefinedByCall(conv2d, builder);

      Set<PointsToSetVariable> conv3ds = getKeysDefinedByCall(conv3d, builder);

      // Detect enumerate-first-field reads (the `step` slot of `for step, x in enumerate(...)`) and
      // pin their tensor-type state to empty via a `DropOp` edge transfer. Without this, the
      // PA assignment graph leaks the underlying dataset's field-0 tensor type into the integer
      // index slot, which then propagates to any function that receives `step` as an argument.
      // The factory already throws `IllegalArgumentException` for these PropertyReads in
      // `TensorGeneratorFactory.getGenerator`, but that only prevents generator-level seeding; it
      // doesn't block the PTS-graph edge. See wala/ML#409.
      Set<PointsToSetVariable> drops = HashSetFactory.make();
      for (PointsToSetVariable v : dataflow) {
        if (isEnumerateFirstFieldRead(v, builder)) drops.add(v);
      }
      // Semantically non-tensor iteration machinery pins to empty-and-fixed too (wala/ML#732):
      // an all-constant-bounds slice constructor (`slice(None, None, None)` under `x[:, 0]`) is a
      // runtime slice object, and an enumerate result is an iterator object whose element typing
      // the element generators serve; both otherwise read cross-caller tensor state through
      // shared builtin frames and count as tensor defs downstream. The subscript-application form
      // is untouched: its first argument is a tensor receiver, not a constant.
      for (PointsToSetVariable v : dataflow) {
        if (!(v.getPointerKey() instanceof LocalPointerKey)) continue;
        LocalPointerKey lpk = (LocalPointerKey) v.getPointerKey();
        if (lpk.getNode().getDU() == null || lpk.getNode().getIR() == null) continue;
        SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
        if (!(def instanceof SSAAbstractInvokeInstruction)) continue;
        SSAAbstractInvokeInstruction invoke = (SSAAbstractInvokeInstruction) def;
        if (isEnumerateCall(lpk.getNode(), invoke, builder)
            || isConstantSliceConstructor(lpk.getNode(), invoke, builder)) drops.add(v);
      }

      LOGGER.fine(() -> "wala/ML#409 drops (enumerate-first-field): " + drops.size());
      for (PointsToSetVariable d : drops)
        LOGGER.fine(() -> "  drop: " + describe(d.getPointerKey()));

      // User-code parameter destinations get the hybridization-frame origin and a barrier against
      // caller-side origin inflow (wala/ML#726): under `tf.function` tracing a tensor parameter is
      // a symbolic tensor regardless of the library that produced its eager feeds, so its
      // provenance is first-class rather than inherited. Types still flow into parameters; only
      // provenance stops at the boundary. Seeding every user parameter unconditionally is safe:
      // non-tensor parameters never surface, since the solution iterator filters empty-state
      // variables. Only user code bodies (AstMethods) are hybridization frames, so synthetic
      // parameters (builtin summaries, XML API summaries, trampolines) neither seed nor barrier:
      // seeding them exported the parameter constant through shared builtin frames onto every
      // caller's derived values (the slice builtin under `edges[:, 0]` was the wala/ML#731
      // witness), and values crossing pass-through API summaries lost their true provenance.
      Set<PointsToSetVariable> parameters = HashSetFactory.make();
      for (PointsToSetVariable v : dataflow)
        if (v.getPointerKey() instanceof LocalPointerKey
            && ((LocalPointerKey) v.getPointerKey()).isParameter()
            && ((LocalPointerKey) v.getPointerKey()).getNode().getMethod() instanceof AstMethod)
          parameters.add(v);
      LOGGER.fine(() -> "wala/ML#726 parameter destinations: " + parameters.size());
      for (PointsToSetVariable p : parameters)
        initOrigins.put(p, EnumSet.of(TensorOrigin.PARAMETER));

      // Iteration products do not inherit the hybridization-frame origin (wala/ML#729): iterating
      // a symbolic tensor raises under tf.function tracing, so a value iterated out of a parameter
      // is an eager-only product of the fed data, whose provenance comes from its own seed's
      // creator walk. The PA aliases iteration results with their iterables, so without a filter
      // the parameter constant crosses onto the products. Collected here are the aliased
      // destinations: enumerate results, each-element reads, and their tuple-field unwraps.
      Set<PointsToSetVariable> iterationProducts = HashSetFactory.make();
      for (PointsToSetVariable v : dataflow) {
        if (!(v.getPointerKey() instanceof LocalPointerKey)) continue;
        LocalPointerKey lpk = (LocalPointerKey) v.getPointerKey();
        if (lpk.getNode().getDU() == null || lpk.getNode().getIR() == null) continue;
        SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
        if (def instanceof EachElementGetInstruction) {
          iterationProducts.add(v);
          continue;
        }
        if (def instanceof PythonPropertyRead) {
          SSAInstruction objDef =
              lpk.getNode().getDU().getDef(((PythonPropertyRead) def).getObjectRef());
          if (objDef instanceof EachElementGetInstruction) iterationProducts.add(v);
          continue;
        }
        if (def instanceof SSAAbstractInvokeInstruction
            && isEnumerateCall(lpk.getNode(), (SSAAbstractInvokeInstruction) def, builder))
          iterationProducts.add(v);
      }
      LOGGER.fine(() -> "wala/ML#729 iteration-product destinations: " + iterationProducts.size());

      // A call site whose constant bindings decide a callee guard must not receive the callee's
      // merged return flow: the merged return-value key unions every arm, including those the
      // site's constants prove untaken (wala/ML#746). Suppress the merged edge per such site and
      // wire the feasible returns' values directly to the call result instead. Sites whose
      // bindings prune nothing, or whose feasible returns the propagation system cannot
      // represent, keep the merged edge untouched.
      Map<PointsToSetVariable, Set<PointsToSetVariable>> armSuppressions = HashMapFactory.make();
      HeapModel heapModel = builder.getPointerAnalysis().getHeapModel();
      // Resolve pointer keys against the duplicated flow graph itself: the graph includes
      // implicit-constraint nodes (synthetic-method locals and returns have implicit pointer
      // keys), which the propagation system's find-or-create cannot represent.
      Map<PointerKey, PointsToSetVariable> flowVarsByKey = HashMapFactory.make();
      for (PointsToSetVariable flowVar : dataflow)
        flowVarsByKey.putIfAbsent(flowVar.getPointerKey(), flowVar);
      for (CGNode caller : builder.getCallGraph()) {
        IR callerIR = caller.getIR();
        if (callerIR == null) continue;
        for (Iterator<SSAInstruction> it = callerIR.iterateAllInstructions(); it.hasNext(); ) {
          SSAInstruction inst = it.next();
          // Synthetic forwarding invokes (trampoline bodies) are not PythonInvokeInstructions but
          // still carry the hop whose callee holds the guard; the bindings computation declines
          // them and the reachability fold still decides via node-wide points-to singletons and
          // constant instance attributes (wala/ML#761).
          if (!(inst instanceof SSAAbstractInvokeInstruction)) continue;
          SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) inst;
          if (call.getNumberOfReturnValues() == 0) continue;
          PointsToSetVariable dest =
              flowVarsByKey.get(heapModel.getPointerKeyForLocal(caller, call.getReturnValue(0)));
          if (dest == null) continue;
          for (CGNode callee :
              builder.getCallGraph().getPossibleTargets(caller, call.getCallSite())) {
            if (callee.getIR() == null) continue;
            Map<Integer, Object> bindings =
                TensorGenerator.computeCallSiteConstantBindings(builder, caller, call, callee);
            Set<ISSABasicBlock> reachable =
                TensorGenerator.computeReachableBlocksUnderBindings(builder, callee, bindings);
            List<Integer> feasibleReturns = new ArrayList<>();
            boolean prunedReturn = false;
            boolean unrepresentable = false;
            for (ISSABasicBlock block : callee.getIR().getControlFlowGraph()) {
              for (SSAInstruction member : block) {
                if (!(member instanceof SSAReturnInstruction)) continue;
                int retVn = ((SSAReturnInstruction) member).getResult();
                if (!reachable.contains(block)) {
                  prunedReturn = true;
                  continue;
                }
                if (retVn < 0) continue;
                if (flowVarsByKey.get(heapModel.getPointerKeyForLocal(callee, retVn)) == null) {
                  unrepresentable = true;
                  continue;
                }
                feasibleReturns.add(retVn);
              }
            }
            {
              boolean prunedF = prunedReturn;
              boolean unrepF = unrepresentable;
              int feasF = feasibleReturns.size();
              CGNode calleeF = callee;
              LOGGER.fine(
                  () ->
                      "ARM-SWEEP callee "
                          + describe(calleeF)
                          + ": pruned="
                          + prunedF
                          + " feasible="
                          + feasF
                          + " unrepresentable="
                          + unrepF
                          + " bindings="
                          + bindings.size()
                          + ".");
            }
            if (!prunedReturn || feasibleReturns.isEmpty() || unrepresentable) continue;
            PointsToSetVariable merged =
                flowVarsByKey.get(heapModel.getPointerKeyForReturnValue(callee));
            if (merged == null || !dataflow.hasEdge(merged, dest)) continue;
            for (int retVn : feasibleReturns) {
              PointsToSetVariable retVar =
                  flowVarsByKey.get(heapModel.getPointerKeyForLocal(callee, retVn));
              if (!dataflow.hasEdge(retVar, dest)) dataflow.addEdge(retVar, dest);
            }
            armSuppressions.computeIfAbsent(dest, k -> HashSetFactory.make()).add(merged);
            LOGGER.fine(
                () ->
                    "Suppressing the merged return edge into "
                        + describe(dest.getPointerKey())
                        + " from "
                        + describe(callee)
                        + " under the site bindings "
                        + bindings
                        + " (wala/ML#746).");
          }
        }
      }
      LOGGER.fine(() -> "wala/ML#746 arm-suppressed call results: " + armSuppressions.size());

      // A φ whose governing branch folds must not receive its infeasible arm's flow either: the
      // dataflow union has no branch sensitivity, so a decided guard's dead arm (a helper call
      // that never runs, a dead reassignment) leaks its members into the φ and onward
      // (wala/ML#763). Suppress the dead arm's edge, keeping at least one live arm.
      int phiSuppressions = 0;
      for (CGNode node : builder.getCallGraph()) {
        IR ir = node.getIR();
        if (ir == null) continue;
        for (Iterator<? extends SSAInstruction> it = ir.iteratePhis(); it.hasNext(); ) {
          SSAInstruction phi = it.next();
          if (!(phi instanceof SSAPhiInstruction)) continue;
          PointsToSetVariable phiVar =
              flowVarsByKey.get(heapModel.getPointerKeyForLocal(node, phi.getDef()));
          if (phiVar == null) continue;
          List<PointsToSetVariable> infeasible = new ArrayList<>();
          boolean liveArm = false;
          for (int i = 0; i < phi.getNumberOfUses(); i++) {
            int useVn = phi.getUse(i);
            PointsToSetVariable armVar =
                useVn > 0 ? flowVarsByKey.get(heapModel.getPointerKeyForLocal(node, useVn)) : null;
            if (Boolean.FALSE.equals(
                TensorGenerator.computePhiArmFeasibility(
                    builder, node, (SSAPhiInstruction) phi, i))) {
              if (armVar != null && dataflow.hasEdge(armVar, phiVar)) infeasible.add(armVar);
            } else {
              liveArm = true;
            }
          }
          if (!liveArm || infeasible.isEmpty()) continue;
          for (PointsToSetVariable armVar : infeasible) {
            armSuppressions.computeIfAbsent(phiVar, k -> HashSetFactory.make()).add(armVar);
            phiSuppressions++;
          }
        }
      }
      int phiCount = phiSuppressions;
      LOGGER.fine(() -> "wala/ML#763 arm-suppressed phi edges: " + phiCount);

      TensorTypeAnalysis tt =
          new TensorTypeAnalysis(
              dataflow,
              init,
              initOrigins,
              shapeOps,
              setCalls,
              refinements,
              typeFeeds,
              armSuppressions,
              conv2ds,
              conv3ds,
              drops,
              parameters,
              iterationProducts,
              errorLog);

      tt.solve(new NullProgressMonitor());

      // A dtype feed that never delivered any operand state left its destination stateless,
      // although the suppressed seed at least asserted an unknown tensor; restore those seeds
      // and continue the (monotone) fixpoint so the restored members propagate (wala/ML#736).
      if (tt.restoreUnfedSeeds(suppressedSeeds, suppressedOrigins))
        tt.solve(new NullProgressMonitor());
      for (PointsToSetVariable fed : typeFeeds.keySet())
        LOGGER.fine(
            () ->
                "post-feed state for "
                    + describe(fed.getPointerKey())
                    + ": "
                    + tt.getOutState(fed));

      recordDepthLimitedResults(tt, builder.getClassHierarchy());
      recordShapeAnnotationCandidates(builder);

      // The wala/ML#753 replay diagnostic runs last, when every demanded query has settled, so
      // its recomputations read only final values and cannot perturb the evaluation order under
      // measurement.
      String replayFilter = System.getProperty(REPLAY_FILTER_PROPERTY);
      if (replayFilter == null) replayFilter = System.getenv(REPLAY_FILTER_VARIABLE);
      if (replayFilter != null) {
        WorklistTypeResolver engine = WorklistTypeResolver.active(builder);
        if (engine != null) engine.replaySettled(replayFilter);
      }

      return tt;
    } finally {
      // The engine (with its query state) is uninstalled and the remaining per-builder memos are
      // cleared now that the analysis is done rather than waiting for the builder to be
      // garbage-collected &mdash; this keeps memory predictable in long-running clients (LSP
      // server, repeated-analysis daemons) where builders may be held in other caches after their
      // analysis completes. The `finally` ensures the state is released and the interpreter-miss
      // counter is reset even when the analysis exits early via `CancelException`, so neither
      // leaks into the next run.
      WorklistTypeResolver.uninstall(builder);
      TensorGenerator.clearCaches(builder);
      reportAndResetInterpreterUnavailableMisses();
    }
  }

  /**
   * Emits the wala/ML#444 end-of-analysis summary when the interpreter was unavailable for one or
   * more shape expressions during the run, then atomically resets the counter for the next run.
   * Surfacing the aggregate precision loss once is easier to notice than the single early WARNING
   * the interpreter emits on its first miss (easily lost in build noise).
   */
  static void reportAndResetInterpreterUnavailableMisses() {
    final int interpreterMisses = PythonInterpreter.getAndResetInterpreterUnavailableMisses();
    if (interpreterMisses > 0) {
      LOGGER.warning(
          () ->
              interpreterMisses
                  + " shape expression(s) could not be evaluated because the Jython interpreter was"
                  + " unavailable; the affected tensor dimensions fell back to symbolic, so shape"
                  + " precision is reduced for this run.");
    }
  }

  /**
   * Returns the set of possible {@link TensorType}s that the given {@link PointsToSetVariable} can
   * take on, or {@code null} if the variable is a recognized tensor source but its type cannot be
   * determined (unknown / ⊤). An empty set means the variable is not a recognized tensor source
   * (⊥).
   *
   * @param source The dataflow source to analyze.
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph and pointer
   *     analysis.
   * @return A set of {@link TensorType}s, or {@code null} if the tensor type is unknown.
   */
  private Set<TensorType> getTensorTypes(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    LOGGER.fine("Getting tensor types for source: " + describe(source) + ".");

    try {
      TensorGenerator generator = getGenerator(source, builder);
      LOGGER.fine("Using tensor generator: " + generator + ".");

      // `getGenerator` returns `null` when the cycle guard added in
      // `wala/ML#435` / `ponder-lab/ML#192` re-encounters a `PointsToSetVariable`
      // along a single dispatch chain. `null` means "unknown / ⊤"; return
      // `null` here to preserve this method's documented semantics
      // (`null` = unknown / ⊤; empty set = ⊥ / not a recognized tensor source)
      // and avoid NPE-ing on the subsequent `getTensorTypes` dispatch.
      if (generator == null) return null;

      Set<TensorType> tensorTypes = generator.getTensorTypes(builder);
      LOGGER.fine(() -> "Found tensor types: " + tensorTypes + ".");

      if (tensorTypes != null && tensorTypes.stream().anyMatch(t -> t.getDims() == null))
        LOGGER.fine(
            () ->
                "NULLDIMS-SEED generator="
                    + generator.getClass().getSimpleName()
                    + " source="
                    + describe(source));

      return tensorTypes;
    } catch (IllegalArgumentException e) {
      LOGGER.log(
          Level.FINER,
          e,
          () -> "Source " + describe(source) + " is not a recognized tensor generator.");
      return HashSetFactory.make();
    }
  }

  /**
   * Returns the libraries whose operations produce the given dataflow source's value (wala/ML#724),
   * classified by the source's dispatched {@link TensorGenerator}.
   *
   * @param source The dataflow source to classify.
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph and pointer
   *     analysis.
   * @return The producing libraries; empty when the source dispatches to no generator (so there is
   *     no origin evidence to seed).
   */
  private Set<TensorOrigin> getTensorOrigins(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    try {
      TensorGenerator generator = getGenerator(source, builder);
      if (generator == null) return EnumSet.noneOf(TensorOrigin.class);
      return generator.getOrigins(builder);
    } catch (IllegalArgumentException e) {
      return EnumSet.noneOf(TensorOrigin.class);
    }
  }

  private Map<PointsToSetVariable, Set<TensorType>> handleShapeSourceOp(
      PropagationCallGraphBuilder builder,
      Graph<PointsToSetVariable> dataflow,
      MethodReference op,
      int shapeSrcOperand) {
    Map<PointsToSetVariable, Set<TensorType>> reshapeTypes =
        getShapeSourceCalls(op, builder, shapeSrcOperand);
    for (PointsToSetVariable to : reshapeTypes.keySet()) {
      assert to.getPointerKey() instanceof LocalPointerKey;
      int toVn = ((LocalPointerKey) to.getPointerKey()).getValueNumber();
      CGNode srcNode = ((LocalPointerKey) to.getPointerKey()).getNode();
      int srcVn = srcNode.getDU().getDef(toVn).getUse(1);
      PointerKey from =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(srcNode, srcVn);

      final PropagationSystem system = builder.getPropagationSystem();

      // If the source is not implicit, we add an edge from the points-to set of the source to the
      // target https://github.com/wala/ML/issues/268.
      if (!system.isImplicit(from)) dataflow.addEdge(system.findOrCreatePointsToSet(from), to);
    }
    return reshapeTypes;
  }

  private void handlePassThroughOp(
      PropagationCallGraphBuilder builder,
      Graph<PointsToSetVariable> dataflow,
      MethodReference op,
      int inputOperand) {
    Set<PointsToSetVariable> lvals = getKeysDefinedByCall(op, builder);
    for (PointsToSetVariable to : lvals) {
      assert to.getPointerKey() instanceof LocalPointerKey;
      int toVn = ((LocalPointerKey) to.getPointerKey()).getValueNumber();
      CGNode srcNode = ((LocalPointerKey) to.getPointerKey()).getNode();
      int srcVn = srcNode.getDU().getDef(toVn).getUse(inputOperand);
      PointerKey from =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(srcNode, srcVn);

      final PropagationSystem system = builder.getPropagationSystem();

      if (!system.isImplicit(from)) dataflow.addEdge(system.findOrCreatePointsToSet(from), to);
    }
  }

  public Map<PointerKey, AnalysisError> getErrors() {
    return errorLog;
  }

  /**
   * The tensor values that resolved to ⊤ at a call-string context saturated to the configured
   * {@code targetedCfaDepth} (wala/ML#601). A non-empty result signals that the depth may be too
   * short: increasing {@code targetedCfaDepth} and re-analyzing to the point where this result is
   * empty tunes the depth to the subject's fixed point, past which every remaining ⊤ is a genuine
   * unknown rather than a merged-context artifact. Cleared and recomputed by each {@link
   * #performAnalysis} run.
   *
   * @return The depth-limited ⊤ values from the most recent analysis; empty when the depth reached
   *     the subject's fixed point or is {@code 0} (context-insensitive, no call-string budget to
   *     deepen).
   */
  public List<DepthLimitedResult> getDepthLimitedResults() {
    return depthLimitedResults;
  }

  /**
   * Returns the triaged wala/ML#370 shape-annotation worklist from the most recent {@link
   * #performAnalysis} run (wala/ML#735): each allocator whose shape argument did not resolve, with
   * the classified reason. A consumer filters to {@link
   * TensorGenerator.ShapeUnresolutionCause#CONTENT_DEPENDENT} for the genuine annotation
   * candidates.
   *
   * @return The shape-annotation triage records; empty when no allocator shape went unresolved.
   */
  public List<ShapeAnnotationCandidate> getShapeAnnotationCandidates() {
    return shapeAnnotationCandidates;
  }

  /**
   * Records the ⊤ values whose nodes sit at a call-string context saturated to {@code
   * targetedCfaDepth}, the wala/ML#601 depth-too-short signal. Scans the solved analysis for ⊤
   * tensor variables at local pointer keys, keeps those whose node is routed through the targeted
   * k-CFA selector (only those depths are governed by {@code targetedCfaDepth}) and whose call
   * string is saturated at the depth budget, and logs a single summary warning when any are found.
   *
   * @param tt The solved tensor-type analysis.
   * @param cha The class hierarchy, used to resolve the {@code tf.keras.Model} class for the
   *     targeted-context test.
   */
  private void recordDepthLimitedResults(TensorTypeAnalysis tt, IClassHierarchy cha) {
    depthLimitedResults.clear();
    // Depth 0 is context-insensitive: there is no call-string budget to deepen, so no ⊤ is
    // attributable to a truncated call string.
    if (targetedCfaDepth < 1) return;

    IClass modelClass = cha.lookupClass(TensorFlowTypes.MODEL.getDeclaringClass());
    for (Iterator<Pair<PointerKey, TensorVariable>> it = tt.iterator(); it.hasNext(); ) {
      Pair<PointerKey, TensorVariable> pair = it.next();
      if (!(pair.fst instanceof LocalPointerKey)) continue;
      if (!hasTopShape(pair.snd)) continue;
      LocalPointerKey key = (LocalPointerKey) pair.fst;
      CGNode node = key.getNode();
      if (!receivesTargetedContext(node.getMethod(), modelClass)) continue;
      int length = measureCallStringLength(node.getContext());
      if (length == targetedCfaDepth)
        depthLimitedResults.add(new DepthLimitedResult(key, node, key.getValueNumber(), length));
    }

    if (!depthLimitedResults.isEmpty())
      LOGGER.warning(
          () ->
              depthLimitedResults.size()
                  + " tensor value(s) resolved to an unknown shape at a call string saturated to"
                  + " the targeted CFA depth of "
                  + targetedCfaDepth
                  + "; a deeper depth may resolve them. See wala/ML#601.");
  }

  /**
   * Records the triaged wala/ML#370 shape-annotation worklist (wala/ML#735): the allocators whose
   * shape argument did not resolve, classified content-dependent vs recoverable-gap. The generators
   * accumulate the raw {@code (source, cause)} pairs during the run (see {@link
   * TensorGenerator#recordShapeAnnotationCandidate}); this reads them before {@link
   * TensorGenerator#clearCaches} drops them, dedups per pointer key (a memoized shape read may
   * record a source more than once), and keeps the strongest classification per key. Every recorded
   * key is an allocator whose shape did not resolve at its ⊤ floor; the floor is taken as the
   * signal rather than intersected with the solved ⊤ set, since a manually anchored allocator's
   * synthetic-return key is absent from the analysis's seeded pointer keys and intersecting would
   * drop exactly those.
   *
   * @param builder The builder whose per-run generator records to read.
   */
  private void recordShapeAnnotationCandidates(PropagationCallGraphBuilder builder) {
    shapeAnnotationCandidates.clear();

    // Keep the strongest classification per source key: a content-dependent finding for any read of
    // a source overrides a recoverable or undetermined finding for the same source. The generators
    // record only at their ⊤ floor, so every recorded key is an allocator whose shape did not
    // resolve; a manually anchored allocator's synthetic-return key is absent from the analysis's
    // seeded pointer keys, so intersecting with the solved ⊤ set would drop exactly those, and the
    // recorded floor is taken as the signal instead.
    Map<PointerKey, TensorGenerator.ShapeUnresolutionCause> byKey = new HashMap<>();
    for (Pair<PointerKey, TensorGenerator.ShapeUnresolutionCause> record :
        TensorGenerator.getShapeAnnotationCandidates(builder)) {
      PointerKey key = record.fst;
      byKey.merge(
          key,
          record.snd,
          (a, b) ->
              a == TensorGenerator.ShapeUnresolutionCause.CONTENT_DEPENDENT
                      || b == TensorGenerator.ShapeUnresolutionCause.CONTENT_DEPENDENT
                  ? TensorGenerator.ShapeUnresolutionCause.CONTENT_DEPENDENT
                  : a == TensorGenerator.ShapeUnresolutionCause.RECOVERABLE_GAP ? a : b);
    }

    byKey.forEach(
        (key, cause) -> shapeAnnotationCandidates.add(new ShapeAnnotationCandidate(key, cause)));

    long contentDependent =
        shapeAnnotationCandidates.stream()
            .filter(c -> c.cause() == TensorGenerator.ShapeUnresolutionCause.CONTENT_DEPENDENT)
            .count();
    if (contentDependent > 0)
      LOGGER.fine(
          () ->
              contentDependent
                  + " content-dependent allocator shape(s) are wala/ML#370 annotation candidates; "
                  + (shapeAnnotationCandidates.size() - contentDependent)
                  + " unresolved allocator shape(s) are recoverable/undetermined precision gaps"
                  + " (wala/ML#735).");
  }

  /**
   * Determines whether a tensor variable carries a ⊤ (unknown-rank) shape: a type whose dimension
   * list is {@code null}. A ⊤ shape is distinct from ⊥ (not a tensor, an empty type set) and from a
   * known-rank shape with unknown sizes.
   *
   * @param variable The tensor variable to test.
   * @return {@code true} if any of {@code variable}'s types has a {@code null} dimension list.
   */
  private static boolean hasTopShape(TensorVariable variable) {
    for (TensorType type : variable.getTypes()) if (type.getDims() == null) return true;
    return false;
  }

  /**
   * Returns the call-string length of a context, or {@code -1} when the context exposes no call
   * string (e.g. a receiver-instance context, whose separation is not governed by {@code
   * targetedCfaDepth}).
   *
   * @param context The context to inspect.
   * @return The number of call sites in the context's call string, or {@code -1} when absent.
   */
  private static int measureCallStringLength(Context context) {
    Object callString = context.get(CallStringContextSelector.CALL_STRING);
    return callString instanceof CallString
        ? ((CallString) callString).getCallSiteRefs().length
        : -1;
  }

  protected void addBypassLogic(IClassHierarchy cha, AnalysisOptions options) {
    super.addBypassLogic(cha, options);
    // Load numpy before tensorflow so tensorflow-level generators can reference numpy types via
    // `NumpyTypes` constants without depending on load order side effects.
    addSummaryBypassLogic(options, "numpy.xml");
    addSummaryBypassLogic(options, "tensorflow.xml");
  }

  /**
   * {@inheritDoc}
   *
   * <p>{@code tensorflow.xml} declares class shells (via {@code <class super="...">}) for the
   * subclassable Keras framework bases (e.g. {@code tf.keras.layers.Layer}), so user subclasses
   * resolve their base in the class hierarchy (wala/ML#118).
   */
  @Override
  protected Collection<String> getSummaryClassShellSummaries() {
    return List.of("tensorflow.xml");
  }
}
