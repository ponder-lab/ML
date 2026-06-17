package com.ibm.wala.cast.python.ml.client;

import static com.google.common.collect.Sets.newHashSet;
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
import com.ibm.wala.cast.lsp.AnalysisError;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorType;
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
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.PropagationSystem;
import com.ibm.wala.ipa.callgraph.propagation.cfa.nCFAContextSelector;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.NullProgressMonitor;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.graph.Graph;
import com.ibm.wala.util.graph.impl.SlowSparseNumberedGraph;
import com.ibm.wala.util.intset.IntSet;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.File;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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
   * testNeuralNetwork1-4}, which recover e.g. {@code accuracy}'s {@code y_pred} as the per-context
   * union {@code {(256, 10) float32, ? float32}} (wala/ML#379, wala/ML#530).
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

    builder.setContextSelector(
        new ContextSelector() {
          @Override
          public Context getCalleeTarget(
              CGNode caller,
              CallSiteReference site,
              IMethod callee,
              InstanceKey[] actualParameters) {
            String calleeClass = callee.getDeclaringClass().getName().toString();
            // Apply k-CFA for any methods in the target framework, which includes internal helpers,
            // as well as methods declared on user-defined `tf.keras.Model` subclasses (e.g.
            // `NeuralNet.call`). Without the latter, a user model called from multiple sites (train
            // vs. test) merges into one context-insensitive node, collapsing its layer-output
            // allocations across callers and losing per-context shape (wala/ML#530).
            if (calleeClass.contains(targetFramework)
                || isModelSubclassMethod(callee, modelClass)
                || isUserModelForwardMethod(calleeClass)) {
              return targetedCFA.getCalleeTarget(caller, site, callee, actualParameters);
            }
            return base.getCalleeTarget(caller, site, callee, actualParameters);
          }

          @Override
          public IntSet getRelevantParameters(CGNode caller, CallSiteReference site) {
            return base.getRelevantParameters(caller, site);
          }
        });

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
          LOGGER.fine("Added dataflow source from binary op: " + src + ".");
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
            // we may be invoking `next()` on a dataset.
            if (def instanceof SSAAbstractInvokeInstruction && def.getNumberOfUses() > 1) {
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
          LOGGER.fine(() -> "Found tensor generator: " + generator + " for source: " + src + ".");
          sources.add(src);
          LOGGER.fine("Added dataflow source from tensor generator: " + src + ".");
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
            IClass type = ctk.getType();
            TypeReference reference = type.getReference();

            if (reference.equals(NEXT.getDeclaringClass())) {
              // it's a call to `next()`. Look up the iterator definition.
              int iterator = instruction.getUse(1);
              SSAInstruction iteratorDef = du.getDef(iterator);

              // Let's see if the iterator is over a tensor dataset. First, check the iterator
              // for a dataset source. NOTE: We can only do this because `iter()` is currently
              // just passing-through its argument.
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
                // Use the original instruction. NOTE: We can only do this because `iter()` is
                // currently just passing-through its argument.
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
          LOGGER.fine("Added dataflow source from tensor iterable: " + src + ".");
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
        IClass concreteType = asin.getConcreteType();
        TypeReference reference = concreteType.getReference();

        if ((reference.equals(DATASET)
                || reference.getName().toString().startsWith(DATA_PACKAGE_PREFIX))
            && isDatasetTensorElement(src, use, pointerAnalysis)) {
          sources.add(src);
          LOGGER.fine("Added dataflow source from tensor dataset: " + src + ".");
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
            IClass type = typeKey.getType();
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
            TypeReference reference = asin.getConcreteType().getReference();

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

  private Map<PointsToSetVariable, TensorType> getShapeSourceCalls(
      MethodReference op, PropagationCallGraphBuilder builder, int param) {
    Map<PointsToSetVariable, TensorType> targets = HashMapFactory.make();
    getSourceCalls(
        op,
        builder,
        (CGNode src, SSAAbstractInvokeInstruction call) -> {
          if (call.getNumberOfUses() > param) {
            PointerKey defKey =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(src, call.getDef());
            // Materializing an implicitly-represented key would make WALA dump the entire call
            // graph's IR (an unconditional debug print), so skip it; implicit keys carry no
            // explicit dataflow variable to pin (wala/ML#573).
            if (!builder.getPropagationSystem().isImplicit(defKey))
              targets.put(
                  builder.getPropagationSystem().findOrCreatePointsToSet(defKey),
                  TensorType.shapeArg(src, call.getUse(param)));
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
                && SET_SHAPE_RECEIVER_TYPES.contains(asin.getConcreteType().getReference())) {
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
            TensorType.shapeArg(caller, call.getUse(1)));
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
    try {
      Graph<PointsToSetVariable> dataflow =
          SlowSparseNumberedGraph.duplicate(
              builder.getPropagationSystem().getFlowGraphIncludingImplicitConstraints());

      Set<PointsToSetVariable> sources = getDataflowSources(builder, dataflow);

      Map<PointsToSetVariable, Set<TensorType>> init = HashMapFactory.make();

      for (PointsToSetVariable v : sources) init.put(v, getTensorTypes(v, builder));

      Map<PointsToSetVariable, TensorType> placeholders =
          handleShapeSourceOp(builder, dataflow, placeholder, 2);
      LOGGER.fine("Placeholders: " + placeholders);

      for (Map.Entry<PointsToSetVariable, TensorType> e : placeholders.entrySet())
        init.put(e.getKey(), Set.of(e.getValue()));

      // wala/ML#509: recognize `x.set_shape(s)` via IR scanning rather than call-graph dispatch on
      // the `Ltensorflow/functions/set_shape` synthetic class. The legacy dispatch path requires
      // the receiver to have the `set_shape` attribute attached (via FixedLenFeature.do's
      // <putfield>) and breaks when the cast pass_through alias is removed.
      Map<PointsToSetVariable, TensorType> setCalls = getSetShapeCallsSyntactic(builder);

      // wala/ML#509: `set_shape` is a user-supplied OVERRIDE of any per-op-generator init seed on
      // the receiver. Remove receivers from `init` so the SetShapeOp edge transfer is the sole
      // source of state for those variables; otherwise the meet-time union re-introduces the
      // generator-seeded type (e.g., Cast generator's (?, float32) on the cast-result variable).
      for (PointsToSetVariable recv : setCalls.keySet()) init.remove(recv);

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

      Map<PointsToSetVariable, TensorType> shapeOps = HashMapFactory.make();
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
      LOGGER.fine(() -> "wala/ML#409 drops (enumerate-first-field): " + drops.size());
      for (PointsToSetVariable d : drops) LOGGER.fine(() -> "  drop: " + d.getPointerKey());

      TensorTypeAnalysis tt =
          new TensorTypeAnalysis(
              dataflow, init, shapeOps, setCalls, conv2ds, conv3ds, drops, errorLog);

      tt.solve(new NullProgressMonitor());

      return tt;
    } finally {
      // `TensorGenerator.getShapes`/`getDTypes` memoize per-builder. Clear those caches now that
      // the analysis is done rather than waiting for the builder to be garbage-collected &mdash;
      // this keeps memory predictable in long-running clients (LSP server, repeated-analysis
      // daemons) where builders may be held in other caches after their analysis completes. The
      // `finally` ensures the caches are cleared and the interpreter-miss counter is reset even
      // when the analysis exits early via `CancelException`, so neither leaks into the next run.
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
    LOGGER.fine("Getting tensor types for source: " + source + ".");

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

      return tensorTypes;
    } catch (IllegalArgumentException e) {
      LOGGER.log(Level.FINER, "Source " + source + " is not a recognized tensor generator.", e);
      return HashSetFactory.make();
    }
  }

  private Map<PointsToSetVariable, TensorType> handleShapeSourceOp(
      PropagationCallGraphBuilder builder,
      Graph<PointsToSetVariable> dataflow,
      MethodReference op,
      int shapeSrcOperand) {
    Map<PointsToSetVariable, TensorType> reshapeTypes =
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

  protected void addBypassLogic(IClassHierarchy cha, AnalysisOptions options) {
    super.addBypassLogic(cha, options);
    // Load numpy before tensorflow so tensorflow-level generators can reference numpy types via
    // `NumpyTypes` constants without depending on load order side effects.
    addSummaryBypassLogic(options, "numpy.xml");
    addSummaryBypassLogic(options, "tensorflow.xml");
  }
}
