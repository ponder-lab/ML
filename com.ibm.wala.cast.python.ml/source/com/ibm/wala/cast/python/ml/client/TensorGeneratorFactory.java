package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.NumpyTypes.ASTYPE;
import static com.ibm.wala.cast.python.ml.types.NumpyTypes.ASTYPE_METHOD_NAME;
import static com.ibm.wala.cast.python.ml.types.NumpyTypes.RESHAPE_METHOD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARRAY_OPS_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_BATCH_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_CHOOSE_FROM_DATASETS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_CONCATENATE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_ENUMERATE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FILTER_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_GENERATOR_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_TENSORS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_MAP_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_PREFETCH_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_RANDOM_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_RANGE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_REDUCE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_REPEAT_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SAMPLE_FROM_DATASETS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SHUFFLE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_TAKE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_WITH_OPTIONS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_ZIP_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DENSE_CALL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DIVIDE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EQUAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FILL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FLATTEN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_NESTED_ROW_LENGTHS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_NESTED_ROW_SPLITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_NESTED_VALUE_ROWIDS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_ROW_LENGTHS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_ROW_LIMITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_ROW_SPLITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_ROW_STARTS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FROM_VALUE_ROWIDS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GAMMA;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GAMMA_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.INPUT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LOG;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MATMUL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MAX_POOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL_CALL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MULTIPLY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NDARRAY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NORMAL_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONES;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONE_HOT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.PLACEHOLDER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POISSON;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POISSON_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.READ_DATA_SETS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_MEAN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_SUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_FROM_DENSE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SUBTRACT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TF_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.VARIABLE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS_LIKE;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.sanitize;
import static java.util.Map.entry;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.ReturnValueKey;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.graph.Graph;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A factory for creating TensorGenerator instances based on the called TensorFlow function.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorGeneratorFactory {

  /** Logger for this class. */
  private static final Logger LOGGER = getLogger(TensorGeneratorFactory.class.getName());

  /** Attributes of `tf.Tensor` that do not represent tensor elements. */
  private static final Set<String> NON_TENSOR_ATTRIBUTES =
      Set.of("value_index", "dtype", "shape", "name", "graph", "op", "device", "consumers");

  /**
   * Registry of property-name → generator-constructor mappings used by the duck-typing dispatch
   * path in {@link #getGenerator(PointsToSetVariable, PropagationCallGraphBuilder)}. When an invoke
   * instruction's function object came from a {@link PythonPropertyRead} whose member is the
   * constant string key of this map, the factory constructs the corresponding generator regardless
   * of whether WALA resolved the call target to a concrete summary. This reflects Python's dynamic
   * attribute dispatch semantics: {@code x.method_name(...)} resolves by name even when the
   * receiver's static type is unknown.
   *
   * <p>Register a new entry when adding a {@code TensorGenerator} that represents an instance
   * method whose receiver's type may not be statically known at the call site — typical for methods
   * on values that flow through slice operations, tuple destructuring, or other unsummarized
   * property reads. See wala/ML#356 for the broader context.
   */
  private static final Map<String, Function<PointsToSetVariable, TensorGenerator>>
      PROPERTY_NAME_GENERATORS = Map.ofEntries(entry(ASTYPE_METHOD_NAME, AstypeOperation::new));

  /**
   * Resolves the {@link TypeReference} for the function call associated with the given source.
   *
   * <p>This method employs a multi-staged approach:
   *
   * <ol>
   *   <li>If the source represents a local variable (via {@link LocalPointerKey}):
   *       <ul>
   *         <li>If the variable is defined by an invoke instruction, it attempts to resolve the
   *             target's type.
   *         <li>Special handling is provided for calls to generic containers like {@code LCodeBody}
   *             or {@code LRoot}: it inspects the points-to set of the call's receiver (the actual
   *             function object being invoked) to determine the specific concrete type.
   *         <li>If not an invoke, it defaults to the declaring class of the method where the
   *             variable resides.
   *       </ul>
   *   <li>If the source represents a return value (via {@link ReturnValueKey}), it uses the
   *       declaring class of the corresponding method.
   *   <li>As a final fallback, it delegates to {@link Util#getFunction(PointsToSetVariable)}.
   * </ol>
   *
   * @param source the points-to set variable representing the source of the function call
   * @param builder the propagation call graph builder used for the analysis
   * @return the resolved {@link TypeReference}, or {@code null} if it cannot be resolved.
   */
  private static TypeReference getFunction(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    PointerKey k = source.getPointerKey();
    if (k instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) k;
      CGNode node = lpk.getNode();
      int vn = lpk.getValueNumber();
      SSAInstruction def = node.getDU().getDef(vn);

      if (def instanceof SSAAbstractInvokeInstruction) {
        SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) def;
        TypeReference declaredClass = call.getCallSite().getDeclaredTarget().getDeclaringClass();
        if (declaredClass.getName().toString().equals("LCodeBody")
            || declaredClass.getName().toString().equals("LRoot")) {
          // The call target is generic. We inspect the points-to set of the invoked function object
          // (at index 0) to resolve the actual concrete type.
          int funcVn = call.getUse(0);
          PointerKey funcKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, funcVn);
          for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(funcKey)) {
            if (ik instanceof ConcreteTypeKey) {
              return ((ConcreteTypeKey) ik).getType().getReference();
            }
            if (ik instanceof AllocationSiteInNode) {
              return ((AllocationSiteInNode) ik).getConcreteType().getReference();
            }
          }
        }
        return declaredClass;
      } else if (def instanceof SSABinaryOpInstruction) {
        SSABinaryOpInstruction binOp = (SSABinaryOpInstruction) def;
        if (binOp.getOperator() == IBinaryOpInstruction.Operator.ADD) {
          return ADD.getDeclaringClass();
        } else if (binOp.getOperator() == IBinaryOpInstruction.Operator.SUB) {
          return SUBTRACT.getDeclaringClass();
        } else if (binOp.getOperator() == IBinaryOpInstruction.Operator.MUL) {
          return MULTIPLY.getDeclaringClass();
        } else if (binOp.getOperator() == IBinaryOpInstruction.Operator.DIV) {
          return DIVIDE.getDeclaringClass();
        }
        // TODO: Handle other operators:
        // - Modulo (%): tf.math.mod (IBinaryOpInstruction.Operator.REM)
        // - Power (**): tf.math.pow
        // - Bitwise (&, |, ^): tf.bitwise operations
        // - Comparison (==, !=, <, >): tf.equal, etc. (SSAComparisonInstruction)
        // - Unary (-, ~, abs): tf.negative, etc. (SSAUnaryOpInstruction)
      }
      return lpk.getNode().getMethod().getDeclaringClass().getReference();
    } else if (k instanceof ReturnValueKey) {
      return ((ReturnValueKey) k).getNode().getMethod().getDeclaringClass().getReference();
    } else if (k instanceof AllocationSiteInNode) {
      return ((AllocationSiteInNode) k).getConcreteType().getReference();
    }
    return Util.getFunction(source);
  }

  /**
   * Checks if the given type reference matches the expected type reference by name.
   *
   * @param tr the type reference to check
   * @param expected the expected type reference
   * @return {@code true} if the type reference names are equal, {@code false} otherwise or if
   *     either is null
   */
  private static boolean isType(TypeReference tr, TypeReference expected) {
    if (tr == null || expected == null) return false;
    return tr.getName().toString().equals(expected.getName().toString());
  }

  /**
   * Traces the dataflow graph backwards from the given source {@link PointsToSetVariable} to find
   * its creator. The creator is defined as either a return value of a function or a variable
   * defined by a relevant instruction such as an invoke, an iteration instruction, or a property
   * read.
   *
   * @param source The {@link PointsToSetVariable} to trace backwards from.
   * @param builder The {@link PropagationCallGraphBuilder} for the current analysis.
   * @return The {@link PointsToSetVariable} corresponding to the creation site of the value.
   */
  public static PointsToSetVariable findCreator(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    Graph<PointsToSetVariable> assignmentGraph =
        builder.getPropagationSystem().getAssignmentGraph();
    Set<PointsToSetVariable> visited = HashSetFactory.make();
    Queue<PointsToSetVariable> queue = new LinkedList<>();
    queue.add(source);
    visited.add(source);
    LOGGER.fine("findCreator started for source: " + source);

    while (!queue.isEmpty()) {
      PointsToSetVariable current = queue.poll();
      PointerKey pk = current.getPointerKey();
      LOGGER.fine("findCreator visiting: " + current);

      if (pk instanceof ReturnValueKey) {
        LOGGER.fine("findCreator found ReturnValueKey: " + current);
        return current;
      }

      if (pk instanceof LocalPointerKey) {
        LocalPointerKey lpk = (LocalPointerKey) pk;
        SSAInstruction def = lpk.getNode().getDU().getDef(lpk.getValueNumber());
        if (def instanceof SSAAbstractInvokeInstruction
            || def instanceof EachElementGetInstruction
            || def instanceof PythonPropertyRead) {
          LOGGER.fine("findCreator found creator instruction: " + def);
          return current;
        }
      }

      for (Iterator<PointsToSetVariable> it = assignmentGraph.getPredNodes(current);
          it.hasNext(); ) {
        PointsToSetVariable pred = it.next();
        if (visited.add(pred)) {
          LOGGER.fine("findCreator adding pred: " + pred);
          queue.add(pred);
        }
      }
    }

    LOGGER.info("findCreator fallback returning original source: " + source);
    return source;
  }

  private static PointsToSetVariable getPointsToSetVariable(
      PointerKey key, PropagationCallGraphBuilder builder) {
    try {
      return builder.getPropagationSystem().findOrCreatePointsToSet(key);
    } catch (UnimplementedError e) {
      LOGGER.log(Level.FINE, "Could not get points-to set for " + key, e);
      return null;
    }
  }

  private static boolean isNonTensorAttribute(String propertyName) {
    return NON_TENSOR_ATTRIBUTES.contains(propertyName);
  }

  /**
   * Duck-typed generator dispatch by the invoke instruction's function-object property name. If
   * {@code call.getUse(0)}'s def is a {@link PythonPropertyRead} whose member points to a {@link
   * ConstantKey} whose value matches a key in {@link #PROPERTY_NAME_GENERATORS}, the corresponding
   * factory function is invoked to construct a generator for {@code source}. Otherwise returns
   * {@code null}.
   *
   * <p>This is the only stable dispatch path for instance-method calls whose receiver's class is
   * lost through unsummarized ops (slices, tuple destructuring, binop results, etc.). Python's
   * runtime attribute lookup is dynamic, so matching by property name is a sound — if coarse —
   * model of its semantics. See wala/ML#356 for the broader context and wala/ML#359 for the
   * structural improvement that would eventually make this path unnecessary.
   *
   * @param source The {@link PointsToSetVariable} that represents the invocation's result value
   *     (the SSA value number that holds the return of the call), passed to the matched generator's
   *     constructor as its source. This is the same {@code source} the factory's caller wants a
   *     generator for; the helper does not rewrite it.
   * @param call The {@link SSAAbstractInvokeInstruction} whose function object is inspected. {@code
   *     call.getUse(0)} is the callable's SSA value number; its def is checked for a {@link
   *     PythonPropertyRead} pattern.
   * @param node The {@link CGNode} that contains {@code call}. Used to look up the member-ref value
   *     number's def and points-to set.
   * @param vn The SSA value number of {@code source} within {@code node}. Used only for diagnostic
   *     logging so the trace identifies the specific call site.
   * @param builder The {@link PropagationCallGraphBuilder} used to resolve the member-ref points-to
   *     set (which is where the {@link ConstantKey} for the property name lives).
   * @return A new {@link TensorGenerator} constructed by the matched entry in {@link
   *     #PROPERTY_NAME_GENERATORS}, or {@code null} if the function object is not a property read
   *     with a constant-string member, or if the member name has no entry in the registry.
   */
  private static TensorGenerator dispatchByPropertyName(
      PointsToSetVariable source,
      SSAAbstractInvokeInstruction call,
      CGNode node,
      int vn,
      PropagationCallGraphBuilder builder) {
    if (call.getNumberOfUses() == 0) return null;
    SSAInstruction funcDef = node.getDU().getDef(call.getUse(0));
    if (!(funcDef instanceof PythonPropertyRead)) return null;
    PythonPropertyRead funcRead = (PythonPropertyRead) funcDef;
    PointerKey memberKey =
        builder
            .getPointerAnalysis()
            .getHeapModel()
            .getPointerKeyForLocal(node, funcRead.getMemberRef());
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(memberKey)) {
      if (!(ik instanceof ConstantKey)) continue;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (!(value instanceof String)) continue;
      Function<PointsToSetVariable, TensorGenerator> constructor =
          PROPERTY_NAME_GENERATORS.get((String) value);
      if (constructor != null) {
        LOGGER.fine(
            () ->
                "TensorGeneratorFactory: dispatching `."
                    + value
                    + "(...)` call at "
                    + node
                    + " v"
                    + vn
                    + " via property-name registry.");
        return constructor.apply(source);
      }
    }
    return null;
  }

  /**
   * Recursive-call helper for {@link #getGenerator(PointsToSetVariable,
   * PropagationCallGraphBuilder)} that swallows {@link IllegalArgumentException} and returns {@code
   * null} instead. Used at the inner walk sites that recurse into containers, return values, and
   * the objects of property reads: one unresolved branch should not abort dispatch for the whole
   * source. See wala/ML#363.
   */
  private static TensorGenerator tryGetGenerator(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    try {
      return getGenerator(source, builder);
    } catch (IllegalArgumentException e) {
      LOGGER.log(Level.FINE, "tryGetGenerator: swallowed IAE for source=" + source, e);
      return null;
    }
  }

  /**
   * Returns a {@link TensorGenerator} instance for the given source and call graph builder.
   *
   * <p>This method identifies the specific TensorFlow function or operation that produced the value
   * represented by the source. It handles two main cases:
   *
   * <ul>
   *   <li><b>Function Calls (Invoke Instructions):</b> It inspects the call instruction to
   *       determine the invoked function (e.g., {@code tf.add}, {@code tf.constant}) and returns a
   *       corresponding generator. It also handles recursive resolution for return values of calls.
   *   <li><b>Iteration (EachElementGet Instructions):</b> If the source represents an element
   *       obtained from iterating over a collection (e.g., a loop variable), it traces back to the
   *       iterable object (e.g., a {@code tf.data.Dataset}) and delegates to its generator to
   *       determine the element type.
   * </ul>
   *
   * @param source the points-to set variable representing the source of the tensor
   * @param builder the propagation call graph builder used for the analysis
   * @return the corresponding {@link TensorGenerator} for the TensorFlow function
   * @throws IllegalArgumentException if the function call is unknown or not supported
   */
  public static TensorGenerator getGenerator(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    source = findCreator(source, builder);
    PointerKey k = source.getPointerKey();
    if (k instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) k;
      CGNode node = lpk.getNode();
      int vn = lpk.getValueNumber();
      SSAInstruction def = node.getDU().getDef(vn);
      if (def instanceof SSAAbstractInvokeInstruction) {
        SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) def;

        // Duck-typing fallback: dispatch by the property-read member name.
        TensorGenerator byPropertyName = dispatchByPropertyName(source, call, node, vn, builder);
        if (byPropertyName != null) return byPropertyName;

        // Method-style `x.reshape(shape)` where the receiver's PTS is empty (typically because
        // it was lost through a tuple-unpack chain like `x_train, _ = x_train.astype(...), _`).
        // Syntactic discrimination on argument count (2 uses = method call; `tf.reshape(x,
        // shape)` has 3) keeps this disjoint from the `tf.reshape` dispatch path below.
        if (NdarrayReshape.isApplicable(source, builder)) {
          return new NdarrayReshape(source);
        }

        for (CGNode callee : builder.getCallGraph().getPossibleTargets(node, call.getCallSite())) {
          // If we're calling the `enumerate` builtin, we want to return the generator for the
          // underlying iterable (the second element of each tuple returned by the enumerator).
          if (callee
              .getMethod()
              .getReference()
              .getDeclaringClass()
              .equals(PythonTypes.ENUMERATE_BUILTIN)) {
            int iterableVn = call.getUse(1);
            PointerKey iterableKey =
                builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, iterableVn);
            PointsToSetVariable iterableSrc = getPointsToSetVariable(iterableKey, builder);
            return (iterableSrc != null)
                ? new EnumerateGenerator(source, tryGetGenerator(iterableSrc, builder))
                : null;
          }

          // If we're calling `iter`, the result is an iterator over the collection.
          if (callee
              .getMethod()
              .getReference()
              .getDeclaringClass()
              .equals(PythonTypes.ITER_BUILTIN)) {
            int iterableVn = call.getUse(1);
            PointerKey iterableKey =
                builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, iterableVn);
            PointsToSetVariable iterableSrc = getPointsToSetVariable(iterableKey, builder);
            return (iterableSrc != null)
                ? new IteratorGenerator(source, tryGetGenerator(iterableSrc, builder))
                : null;
          }

          // If we're calling `numpy.ndarray.astype`, the result is a new tensor with the same
          // shape as the receiver but a different dtype. See wala/ML#356.
          if (callee
              .getMethod()
              .getReference()
              .getDeclaringClass()
              .equals(ASTYPE.getDeclaringClass())) {
            LOGGER.fine(
                () ->
                    "TensorGeneratorFactory: dispatching astype call at "
                        + node
                        + " v"
                        + vn
                        + " to AstypeOperation.");
            return new AstypeOperation(source);
          }

          // If we're calling `numpy.ndarray.reshape`, the result has the shape specified by the
          // shape argument (resolving `-1` via the receiver's size) and preserves the receiver's
          // dtype. Class-type dispatch rather than property-name dispatch keeps this disjoint
          // from {@code tf.reshape}.
          if (callee
              .getMethod()
              .getReference()
              .getDeclaringClass()
              .equals(RESHAPE_METHOD.getDeclaringClass())) {
            LOGGER.fine(
                () ->
                    "TensorGeneratorFactory: dispatching ndarray.reshape call at "
                        + node
                        + " v"
                        + vn
                        + " to NdarrayReshape.");
            return new NdarrayReshape(source);
          }

          // If we're calling `next`, the result is an element of the collection.
          if (callee
              .getMethod()
              .getReference()
              .getDeclaringClass()
              .equals(PythonTypes.NEXT_BUILTIN)) {
            int iterableVn = call.getUse(1);
            PointerKey iterableKey =
                builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, iterableVn);
            PointsToSetVariable iterableSrc = getPointsToSetVariable(iterableKey, builder);
            if (iterableSrc == null) return null;
            TensorGenerator containerGenerator = tryGetGenerator(iterableSrc, builder);

            while (containerGenerator instanceof DelegatingTensorGenerator) {
              containerGenerator = ((DelegatingTensorGenerator) containerGenerator).getUnderlying();
            }

            // When the iterable comes from a property read on a user-defined class (e.g.,
            // c.some_iter), the generator chain resolves to null. Chase the PA to find the
            // underlying iterator allocation, then resolve through iter()'s argument to the
            // dataset.
            if (containerGenerator == null) {
              OrdinalSet<InstanceKey> iterPTS =
                  builder.getPointerAnalysis().getPointsToSet(iterableKey);
              LOGGER.fine(
                  () -> "next() field-indirection fallback: iterPTS size=" + iterPTS.size());
              for (InstanceKey iterIK : iterPTS) {
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
                    PointerKey iterArgKey =
                        builder
                            .getPointerAnalysis()
                            .getHeapModel()
                            .getPointerKeyForLocal(creatorNode, 2);
                    PointsToSetVariable iterArgSrc = getPointsToSetVariable(iterArgKey, builder);
                    if (iterArgSrc != null) {
                      containerGenerator = tryGetGenerator(iterArgSrc, builder);
                      if (containerGenerator != null) break;
                    }
                  }
                }
              }
            }

            return (containerGenerator instanceof DatasetGenerator)
                ? containerGenerator
                : new TensorElementGenerator(source, containerGenerator);
          }
          PointerKey retKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForReturnValue(callee);
          PointsToSetVariable retSrc = getPointsToSetVariable(retKey, builder);
          if (retSrc != null) {
            // Recursive dispatch on the callee's return value. Swallow IAE so a single
            // unresolved callee doesn't abort dispatch for the whole source — the outer
            // loop should try the remaining candidate callees, and failing that fall through
            // to the `ReturnValueKey` / assignment-graph fallback below. See wala/ML#363.
            TensorGenerator fromRet = tryGetGenerator(retSrc, builder);
            if (fromRet != null) return fromRet;
          }
        }
      } else if (def instanceof EachElementGetInstruction) {
        // We are iterating over a collection (e.g., for loop). Get the generator for the collection
        // itself.
        int iterableVn = def.getUse(0);
        PointerKey iterableKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, iterableVn);
        PointsToSetVariable iterableSrc = getPointsToSetVariable(iterableKey, builder);
        if (iterableSrc == null) return null;
        TensorGenerator containerGenerator = tryGetGenerator(iterableSrc, builder);

        // We have a generator for the container (the object being iterated over).
        // If the container is a `Dataset` (e.g., `tf.data.Dataset`), its generator
        // (`DatasetGenerator`) is defined to return the shapes/dtypes of its *elements* (not the
        // dataset object itself). Therefore, we use it directly.
        //
        // For `Tensors` (e.g., `tf.range`, constants), the generator returns the tensor's own
        // shape. When iterating, we must peel off the first dimension to get the element shape.
        return (containerGenerator instanceof DatasetGenerator)
            ? new DatasetElementGenerator(iterableSrc, containerGenerator)
            : new TensorElementGenerator(source, containerGenerator);
      } else if (def instanceof PythonPropertyRead) {
        // Python iteration may also be translated into property reads (e.g., retrieving an element
        // from a dataset or tensor).
        PythonPropertyRead propRead = (PythonPropertyRead) def;
        int objRef = propRead.getObjectRef();
        PointerKey objKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, objRef);
        PointsToSetVariable objSrc = getPointsToSetVariable(objKey, builder);
        if (objSrc == null) return null;
        TensorGenerator containerGenerator = tryGetGenerator(objSrc, builder);

        TensorGenerator effectiveGenerator = containerGenerator;
        boolean changed = true;
        while (changed) {
          changed = false;
          if (effectiveGenerator instanceof DelegatingTensorGenerator dtg) {
            TensorGenerator next = dtg.getUnderlying();
            if (next != null && next != effectiveGenerator) {
              effectiveGenerator = next;
              changed = true;
            }
          }
          // Exact class check, not instanceof: only plain DatasetGenerator (pass-through wrappers
          // like shuffle/map/repeat/prefetch/take) should be unwrapped. Subclasses have their own
          // shape logic (e.g., DatasetBatchGenerator prepends a batch dim) that would be skipped.
          if (!changed
              && effectiveGenerator != null
              && effectiveGenerator.getClass() == DatasetGenerator.class) {
            DatasetGenerator dg = (DatasetGenerator) effectiveGenerator;
            TensorGenerator receiver = dg.getReceiverGenerator(builder);
            if (receiver != null && receiver != effectiveGenerator) {
              effectiveGenerator = receiver;
              changed = true;
            }
          }
        }

        Integer propertyIndex = null;
        String propertyName = null;
        int memberRef = propRead.getMemberRef();
        PointerKey memberRefKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, memberRef);
        for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(memberRefKey)) {
          if (ik instanceof ConstantKey) {
            Object val = ((ConstantKey<?>) ik).getValue();
            LOGGER.fine(
                "Member ref constant key value: "
                    + val
                    + " (class: "
                    + (val != null ? val.getClass().getName() : "null")
                    + ")");
            if (val instanceof Integer) {
              propertyIndex = (Integer) val;
            } else if (val instanceof String) {
              propertyName = (String) val;
              try {
                propertyIndex = Integer.parseInt((String) val);
              } catch (NumberFormatException e) {
                // Ignore
              }
            } else if (val instanceof Long) {
              propertyIndex = ((Long) val).intValue();
            }
          }
        }

        if (effectiveGenerator instanceof Model
            && (propertyName != null
                && (propertyName.equals("trainable_weights")
                    || propertyName.equals("weights")
                    || propertyName.equals("non_trainable_weights")))) {
          return new ModelWeightsGenerator(source, (Model) effectiveGenerator);
        }

        if (effectiveGenerator instanceof DatasetEnumerateGenerator) {
          DatasetEnumerateGenerator enumGen = (DatasetEnumerateGenerator) effectiveGenerator;
          boolean isFirstElement = propertyIndex != null && propertyIndex == 0;
          boolean isSecondElement = propertyIndex != null && propertyIndex == 1;

          LOGGER.fine(
              "isFirstElement: " + isFirstElement + ", isSecondElement: " + isSecondElement);
          if (isFirstElement) {
            return new EnumerateIndexGenerator(objSrc);
          } else if (isSecondElement) {
            return new DatasetElementGenerator(objSrc, enumGen.getUnderlyingGenerator(builder));
          }
        }

        if (effectiveGenerator instanceof TupleElementProvider tep && propertyIndex != null) {
          if (tep.yieldsTuple(builder)) {
            LOGGER.fine(
                "Found "
                    + TupleElementProvider.class.getName()
                    + " during property read with index "
                    + propertyIndex
                    + "!");
            return new DatasetTupleElementGenerator(objSrc, tep, propertyIndex);
          }
        }

        if (containerGenerator instanceof TensorElementGenerator
            && ((TensorElementGenerator) containerGenerator).getContainerGenerator()
                instanceof EnumerateGenerator) {
          EnumerateGenerator enumGen =
              (EnumerateGenerator)
                  ((TensorElementGenerator) containerGenerator).getContainerGenerator();
          memberRef = propRead.getMemberRef();
          memberRefKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, memberRef);
          boolean isFirstElement = false;
          for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(memberRefKey)) {
            if (ik instanceof ConstantKey) {
              if (((ConstantKey<?>) ik).getValue().equals(0)) {
                isFirstElement = true;
                break;
              }
            }
          }
          if (isFirstElement) {
            throw new IllegalArgumentException("First element of enumerate tuple is not a tensor.");
          }
          return enumGen
              .getUnderlying(); // Return the underlying dataset generator for the second element.
        }

        // Ndarray subscript with ellipsis/newaxis only (`x[..., None]`, `x[None, ...]`,
        // `x[None]`, etc.) — dim-adding patterns that preserve the receiver's tensor-ness.
        // Dispatches ahead of the generic `TensorElementGenerator` fallthrough, which would
        // incorrectly peel the receiver's first dimension for these patterns. See wala/ML#356.
        if (NdarraySubscriptOperation.isApplicable(source, builder)) {
          return new NdarraySubscriptOperation(source);
        }

        // Similar to `EachElementGet`, we check if the container generator represents elements
        // (`Dataset`) or the tensor itself (peeling needed).
        if (propertyName == null || !isNonTensorAttribute(propertyName)) {
          return (containerGenerator instanceof DatasetGenerator)
              ? new DatasetElementGenerator(objSrc, containerGenerator)
              : new TensorElementGenerator(source, containerGenerator);
        }
      }
    }

    TypeReference calledFunction = getFunction(source, builder);
    LOGGER.info("Getting tensor generator for call to: " + calledFunction + ".");

    // sanitize the type name by removing the artificial suffix that is added for synthetic
    // classes to facilitate trampoline generation.
    calledFunction = sanitize(calledFunction);

    LOGGER.info("Getting tensor generator for sanitized call to: " + calledFunction + ".");

    if (isType(calledFunction, ONES.getDeclaringClass())) return new Ones(source);
    else if (isType(calledFunction, CONSTANT.getDeclaringClass())) return new Constant(source);
    else if (isType(calledFunction, RANGE.getDeclaringClass())) return new Range(source);
    else if (isType(calledFunction, UNIFORM.getDeclaringClass())
        || isType(calledFunction, UNIFORM_OP)) return new Uniform(source);
    else if (isType(calledFunction, NORMAL.getDeclaringClass())
        || isType(calledFunction, NORMAL_OP)) return new Normal(source);
    else if (isType(calledFunction, TRUNCATED_NORMAL.getDeclaringClass())
        || isType(calledFunction, TRUNCATED_NORMAL_OP)) return new TruncatedNormal(source);
    else if (isType(calledFunction, ZEROS.getDeclaringClass())) return new Zeros(source);
    else if (isType(calledFunction, ZEROS_LIKE.getDeclaringClass())) return new ZerosLike(source);
    else if (isType(calledFunction, ARRAY_OPS_RESHAPE)
        || calledFunction.getName().equals(TF_RESHAPE)) return new Reshape(source);
    else if (isType(calledFunction, FILL.getDeclaringClass())) return new Fill(source);
    else if (isType(calledFunction, CONVERT_TO_TENSOR.getDeclaringClass()))
      return new ConvertToTensor(source);
    else if (isType(calledFunction, ONE_HOT.getDeclaringClass())) return new OneHot(source);
    else if (isType(calledFunction, EYE.getDeclaringClass())) return new Eye(source);
    else if (isType(calledFunction, SPARSE_EYE.getDeclaringClass())) return new SparseEye(source);
    else if (isType(calledFunction, SPARSE_TENSOR.getDeclaringClass()))
      return new SparseTensor(source);
    else if (isType(calledFunction, GAMMA.getDeclaringClass()) || isType(calledFunction, GAMMA_OP))
      return new Gamma(source);
    else if (isType(calledFunction, INPUT.getDeclaringClass())) return new Input(source);
    else if (isType(calledFunction, POISSON.getDeclaringClass())
        || isType(calledFunction, POISSON_OP)) return new Poisson(source);
    else if (isType(calledFunction, RAGGED_CONSTANT.getDeclaringClass()))
      return new RaggedConstant(source);
    else if (isType(calledFunction, VARIABLE.getDeclaringClass())) return new Variable(source);
    else if (isType(calledFunction, RAGGED_RANGE.getDeclaringClass()))
      return new RaggedRange(source);
    else if (isType(calledFunction, FROM_VALUE_ROWIDS.getDeclaringClass()))
      return new RaggedFromValueRowIds(source);
    else if (isType(calledFunction, FROM_ROW_STARTS.getDeclaringClass()))
      return new RaggedFromRowStarts(source);
    else if (isType(calledFunction, FROM_ROW_SPLITS.getDeclaringClass()))
      return new RaggedFromRowSplits(source);
    else if (isType(calledFunction, FROM_ROW_LENGTHS.getDeclaringClass()))
      return new RaggedFromRowLengths(source);
    else if (isType(calledFunction, FROM_NESTED_ROW_LENGTHS.getDeclaringClass()))
      return new RaggedFromNestedRowLengths(source);
    else if (isType(calledFunction, FROM_NESTED_ROW_SPLITS.getDeclaringClass()))
      return new RaggedFromNestedRowSplits(source);
    else if (isType(calledFunction, FROM_NESTED_VALUE_ROWIDS.getDeclaringClass()))
      return new RaggedFromNestedValueRowIds(source);
    else if (isType(calledFunction, FROM_ROW_LIMITS.getDeclaringClass()))
      return new RaggedFromRowLimits(source);
    else if (isType(calledFunction, MULTIPLY.getDeclaringClass())
        || isType(calledFunction, ADD.getDeclaringClass())
        || isType(calledFunction, SUBTRACT.getDeclaringClass())
        || isType(calledFunction, DIVIDE.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, SPARSE_ADD.getDeclaringClass())) return new SparseAdd(source);
    else if (isType(calledFunction, SPARSE_FROM_DENSE.getDeclaringClass()))
      return new SparseFromDense(source);
    else if (isType(calledFunction, MODEL.getDeclaringClass())) return new Model(source);
    else if (isType(calledFunction, TENSOR.getDeclaringClass())
        || isType(calledFunction, NDARRAY.getDeclaringClass())) return new TensorCall(source);
    else if (isType(calledFunction, DATASET_FROM_TENSOR_SLICES_TYPE))
      return new DatasetFromTensorSlicesGenerator(source);
    else if (isType(calledFunction, DATASET_FROM_TENSORS_TYPE))
      return new DatasetFromTensorsGenerator(source);
    else if (isType(calledFunction, DATASET_BATCH_TYPE)) return new DatasetBatchGenerator(source);
    else if (isType(calledFunction, DATASET_RANGE_TYPE)) return new DatasetRangeGenerator(source);
    else if (isType(calledFunction, DATASET_RANDOM_TYPE)) return new DatasetRandomGenerator(source);
    else if (isType(calledFunction, DATASET_FROM_GENERATOR_TYPE))
      return new DatasetFromGeneratorGenerator(source);
    else if (isType(calledFunction, DATASET_ZIP_TYPE)) return new DatasetZipGenerator(source);
    else if (isType(calledFunction, DATASET_CHOOSE_FROM_DATASETS_TYPE))
      return new DatasetChooseFromDatasetsGenerator(source);
    else if (isType(calledFunction, DATASET_SAMPLE_FROM_DATASETS_TYPE))
      return new DatasetSampleFromDatasetsGenerator(source);
    else if (isType(calledFunction, DATASET_ENUMERATE_TYPE))
      return new DatasetEnumerateGenerator(source);
    else if (isType(calledFunction, DATASET_SHUFFLE_TYPE)
        || isType(calledFunction, DATASET_MAP_TYPE)
        || isType(calledFunction, DATASET_REPEAT_TYPE)
        || isType(calledFunction, DATASET_PREFETCH_TYPE)
        || isType(calledFunction, DATASET_TAKE_TYPE)
        || isType(calledFunction, DATASET_WITH_OPTIONS_TYPE)
        || isType(calledFunction, DATASET_CONCATENATE_TYPE)
        || isType(calledFunction, DATASET_REDUCE_TYPE)
        || isType(calledFunction, DATASET_FILTER_TYPE)
        || isType(calledFunction, DATASET)) return new DatasetGenerator(source);
    else if (isType(calledFunction, IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE))
      return new FlowFromDirectoryGenerator(source);
    else if (isType(calledFunction, READ_DATA_SETS.getDeclaringClass()))
      return new ReadDataSets(source);
    else if (isType(calledFunction, MNIST_X_TRAIN))
      return new MnistInputData(source, MnistInputData.X_TRAIN_SHAPE);
    else if (isType(calledFunction, MNIST_Y_TRAIN))
      return new MnistInputData(source, MnistInputData.Y_TRAIN_SHAPE);
    else if (isType(calledFunction, MNIST_X_TEST))
      return new MnistInputData(source, MnistInputData.X_TEST_SHAPE);
    else if (isType(calledFunction, MNIST_Y_TEST))
      return new MnistInputData(source, MnistInputData.Y_TEST_SHAPE);
    else if (isType(calledFunction, REDUCE_MEAN.getDeclaringClass())) return new ReduceMean(source);
    else if (isType(calledFunction, PLACEHOLDER.getDeclaringClass()))
      return new Placeholder(source);
    else if (isType(calledFunction, EQUAL.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass())
        || isType(calledFunction, SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass()))
      return new SoftmaxCrossEntropy(source);
    else if (isType(calledFunction, LOG.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, REDUCE_SUM.getDeclaringClass())) return new ReduceSum(source);
    else if (isType(calledFunction, MATMUL.getDeclaringClass())) return new MatMul(source);
    else if (isType(calledFunction, DENSE_CALL.getDeclaringClass())) return new DenseCall(source);
    else if (isType(calledFunction, MODEL_CALL.getDeclaringClass())) return new ModelCall(source);
    else if (isType(calledFunction, FLATTEN.getDeclaringClass())) return new Flatten(source);
    else if (isType(calledFunction, MAX_POOL.getDeclaringClass())) return new MaxPool(source);
    else {
      if (source.getPointerKey() instanceof ReturnValueKey) {
        Graph<PointsToSetVariable> assignmentGraph =
            builder.getPropagationSystem().getAssignmentGraph();
        for (Iterator<PointsToSetVariable> it = assignmentGraph.getPredNodes(source);
            it.hasNext(); ) {
          PointsToSetVariable pred = it.next();
          try {
            TensorGenerator gen = getGenerator(pred, builder);
            if (gen != null) {
              return gen;
            }
          } catch (IllegalArgumentException ex) {
            // Ignore and continue searching other predecessors.
          }
        }
      }
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
    }
  }
}
