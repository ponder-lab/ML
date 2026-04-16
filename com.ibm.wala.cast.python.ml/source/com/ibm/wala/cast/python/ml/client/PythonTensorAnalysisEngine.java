package com.ibm.wala.cast.python.ml.client;

import static com.google.common.collect.Sets.newHashSet;
import static com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory.getGenerator;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATA_PACKAGE_PREFIX;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;

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
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
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
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

public class PythonTensorAnalysisEngine extends PythonAnalysisEngine<TensorTypeAnalysis> {

  public static final String TENSORFLOW = TensorFlowTypes.TENSORFLOW;

  private final String targetFramework;

  public PythonTensorAnalysisEngine() {
    this(TENSORFLOW);
  }

  public PythonTensorAnalysisEngine(List<File> pythonPath) {
    this(pythonPath, TENSORFLOW);
  }

  public PythonTensorAnalysisEngine(String targetFramework) {
    this.targetFramework = targetFramework;
  }

  public PythonTensorAnalysisEngine(List<File> pythonPath, String targetFramework) {
    super(pythonPath);
    this.targetFramework = targetFramework;
  }

  @Override
  protected PythonSSAPropagationCallGraphBuilder getCallGraphBuilder(
      IClassHierarchy cha, AnalysisOptions options, IAnalysisCacheView cache2) {
    PythonSSAPropagationCallGraphBuilder builder = super.getCallGraphBuilder(cha, options, cache2);

    final ContextSelector base = builder.getContextSelector();
    final ContextSelector targeted2CFA =
        new nCFAContextSelector(2, new ContextInsensitiveSelector());

    builder.setContextSelector(
        new ContextSelector() {
          @Override
          public Context getCalleeTarget(
              CGNode caller,
              CallSiteReference site,
              IMethod callee,
              InstanceKey[] actualParameters) {
            String calleeClass = callee.getDeclaringClass().getName().toString();
            // Apply 2-CFA for any methods in the target framework, which includes internal helpers.
            if (calleeClass.contains(targetFramework)) {
              return targeted2CFA.getCalleeTarget(caller, site, callee, actualParameters);
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

  /** A "fake" function name in the summaries that indicates that an API produces a new tensor. */
  public static final String TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME = "read_data";

  /**
   * A "fake" function name in the summaries that indicates that an API produces a tensor iterable.
   */
  private static final String TENSOR_ITERABLE_SYNTHETIC_FUNCTION_NAME = "read_dataset";

  private static final Logger logger = Logger.getLogger(PythonTensorAnalysisEngine.class.getName());

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

  private static final MethodReference set_shape =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/set_shape")),
          AstMethodReference.fnSelector);

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
          logger.info("Added dataflow source from binary op: " + src + ".");
        } else if (inst instanceof EachElementGetInstruction) {
          // We are potentially pulling a tensor out of a tensor iterable.
          EachElementGetInstruction eachElementGetInstruction = (EachElementGetInstruction) inst;

          // Don't add the source if the container has elements in it. In that case, we want to add
          // the individual elements themselves as sources instead.
          if (definitionIsNonScalar(eachElementGetInstruction, du))
            logger.info(
                "Definition of instruction: "
                    + eachElementGetInstruction
                    + " is non-scalar. Skipping...");
          else {
            logger.info(
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
          logger.fine(() -> "Found tensor generator: " + generator + " for source: " + src + ".");
          sources.add(src);
          logger.info("Added dataflow source from tensor generator: " + src + ".");
          ret = true;
        } catch (IllegalArgumentException e) {
          // not a tensor source.
          logger.log(Level.FINE, "Not a tensor source: " + methodName, e);
          e.printStackTrace();
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
                        iteratorDef, iterator, node, src, sources, pointerAnalysis);

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
                    if (iterIK instanceof AllocationSiteInNode) {
                      AllocationSiteInNode asin = (AllocationSiteInNode) iterIK;
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
      logger.fine(() -> "Skipping instruction: " + instruction + ". We've seen it before.");
    else {
      logger.fine(() -> "Processing instruction: " + instruction + ".");
      seen.add(instruction);

      if (instruction != null && instruction.getNumberOfUses() > 0) {
        int use = instruction.getUse(0);
        SSAInstruction def = du.getDef(use);

        // First try intraprocedural analysis.
        if (definesTensorIterable(def, node, callGraph, pointerAnalysis)) {
          sources.add(src);
          logger.info("Added dataflow source from tensor iterable: " + src + ".");
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
    logger.info(
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
          logger.info("Added dataflow source from tensor dataset: " + src + ".");
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

                  logger.finest(
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
    logger.fine("Processing definition: " + def + " of instruction: " + instruction + ".");

    int numberOfUses = du.getNumberOfUses(def);
    logger.fine(
        "Definition: "
            + def
            + " of instruction: "
            + instruction
            + " has "
            + numberOfUses
            + " uses.");

    for (Iterator<SSAInstruction> uses = du.getUses(def); uses.hasNext(); ) {
      SSAInstruction useInstruction = uses.next();
      logger.fine("Processing use: " + useInstruction + ".");

      if (useInstruction instanceof PythonPropertyRead) {
        PythonPropertyRead read = (PythonPropertyRead) useInstruction;
        logger.fine("Found property read use: " + read + ".");

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

      logger.fine(() -> "definesTensorIterable checking instruction: " + invocationInstruction);
      for (CGNode target :
          callGraph.getPossibleTargets(node, invocationInstruction.getCallSite())) {
        logger.fine(() -> "Target: " + target.getMethod().getSignature());
        for (Iterator<CGNode> succNodes = callGraph.getSuccNodes(target); succNodes.hasNext(); ) {
          CGNode callee = succNodes.next();
          IMethod calledMethod = callee.getMethod();
          logger.fine(() -> "  Succ: " + calledMethod.getSignature());

          // Does this method call the synthetic "marker?"
          if (calledMethod.getName().toString().equals(TENSOR_ITERABLE_SYNTHETIC_FUNCTION_NAME)) {
            return true;
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
          try {
            if (call.getNumberOfUses() > param)
              targets.put(
                  builder
                      .getPropagationSystem()
                      .findOrCreatePointsToSet(
                          builder
                              .getPointerAnalysis()
                              .getHeapModel()
                              .getPointerKeyForLocal(src, call.getDef())),
                  TensorType.shapeArg(src, call.getUse(param)));
          } catch (IOException e) {
            throw new RuntimeException("Error while processing shape source call: " + call, e);
          }
        });
    return targets;
  }

  private Set<PointsToSetVariable> getKeysDefinedByCall(
      MethodReference op, PropagationCallGraphBuilder builder) {
    Set<PointsToSetVariable> lvals = HashSetFactory.make();
    getSourceCalls(
        op,
        builder,
        (CGNode src, SSAAbstractInvokeInstruction call) -> {
          lvals.add(
              builder
                  .getPropagationSystem()
                  .findOrCreatePointsToSet(
                      builder
                          .getPointerAnalysis()
                          .getHeapModel()
                          .getPointerKeyForLocal(src, call.getDef())));
        });
    return lvals;
  }

  @Override
  public TensorTypeAnalysis performAnalysis(PropagationCallGraphBuilder builder)
      throws CancelException {
    Graph<PointsToSetVariable> dataflow =
        SlowSparseNumberedGraph.duplicate(
            builder.getPropagationSystem().getFlowGraphIncludingImplicitConstraints());

    Set<PointsToSetVariable> sources = getDataflowSources(builder, dataflow);

    Map<PointsToSetVariable, Set<TensorType>> init = HashMapFactory.make();

    for (PointsToSetVariable v : sources) init.put(v, getTensorTypes(v, builder));

    Map<PointsToSetVariable, TensorType> placeholders = null;
    try {
      placeholders = handleShapeSourceOp(builder, dataflow, placeholder, 2);
    } catch (IOException e) {
      throw new RuntimeException("Error while processing placeholder calls.", e);
    }
    logger.fine("Placeholders: " + placeholders);

    for (Map.Entry<PointsToSetVariable, TensorType> e : placeholders.entrySet())
      init.put(e.getKey(), Set.of(e.getValue()));

    Map<PointsToSetVariable, TensorType> setCalls = HashMapFactory.make();
    Map<PointsToSetVariable, TensorType> set_shapes = getShapeSourceCalls(set_shape, builder, 1);

    for (Map.Entry<PointsToSetVariable, TensorType> x : set_shapes.entrySet()) {
      LocalPointerKey localPointerKey = (LocalPointerKey) x.getKey().getPointerKey();
      CGNode setNode = localPointerKey.getNode();
      int defVn = localPointerKey.getValueNumber();
      SSAInstruction read = setNode.getDU().getDef(defVn);
      SSAInstruction call = setNode.getDU().getDef(read.getUse(0));

      PointerKey setKey =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(setNode, call.getUse(0));

      setCalls.put(builder.getPropagationSystem().findOrCreatePointsToSet(setKey), x.getValue());
    }

    Map<PointsToSetVariable, TensorType> shapeOps = HashMapFactory.make();

    try {
      shapeOps.putAll(handleShapeSourceOp(builder, dataflow, reshape, 2));
    } catch (IOException e) {
      throw new RuntimeException("Error while processing reshape calls.", e);
    }

    handlePassThroughOp(builder, dataflow, convert_to_tensor, 1);

    Set<PointsToSetVariable> conv2ds = getKeysDefinedByCall(conv2d, builder);

    Set<PointsToSetVariable> conv3ds = getKeysDefinedByCall(conv3d, builder);

    TensorTypeAnalysis tt =
        new TensorTypeAnalysis(dataflow, init, shapeOps, setCalls, conv2ds, conv3ds, errorLog);

    tt.solve(new NullProgressMonitor());

    return tt;
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
    logger.fine("Getting tensor types for source: " + source + ".");

    try {
      TensorGenerator generator = getGenerator(source, builder);
      logger.fine("Using tensor generator: " + generator + ".");

      Set<TensorType> tensorTypes = generator.getTensorTypes(builder);
      logger.fine(() -> "Found tensor types: " + tensorTypes + ".");

      return tensorTypes;
    } catch (IllegalArgumentException e) {
      logger.log(Level.FINER, "Source " + source + " is not a recognized tensor generator.", e);
      return HashSetFactory.make();
    }
  }

  private Map<PointsToSetVariable, TensorType> handleShapeSourceOp(
      PropagationCallGraphBuilder builder,
      Graph<PointsToSetVariable> dataflow,
      MethodReference op,
      int shapeSrcOperand)
      throws IOException {
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
