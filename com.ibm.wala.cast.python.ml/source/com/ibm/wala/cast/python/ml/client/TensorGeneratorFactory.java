package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARRAY_OPS_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_BATCH_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_CONCATENATE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_ENUMERATE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FILTER_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_GENERATOR_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_MAP_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_PREFETCH_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_RANGE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_REDUCE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_REPEAT_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SHUFFLE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_TAKE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_WITH_OPTIONS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DENSE;
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
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.INPUT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LOG;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MATMUL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MAX_POOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
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
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConcreteTypeKey;
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
import java.util.logging.Logger;

/**
 * A factory for creating TensorGenerator instances based on the called TensorFlow function.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorGeneratorFactory {

  /** Logger for this class. */
  private static final Logger LOGGER = getLogger(TensorGeneratorFactory.class.getName());

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
   * @return the resolved type reference of the function
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
    PointerKey k = source.getPointerKey();
    if (k instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) k;
      CGNode node = lpk.getNode();
      int vn = lpk.getValueNumber();
      SSAInstruction def = node.getDU().getDef(vn);
      if (def instanceof SSAAbstractInvokeInstruction) {
        SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) def;
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
            PointsToSetVariable iterableSrc =
                builder.getPropagationSystem().findOrCreatePointsToSet(iterableKey);
            return getGenerator(iterableSrc, builder);
          }

          PointerKey retKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForReturnValue(callee);
          PointsToSetVariable retSrc =
              builder.getPropagationSystem().findOrCreatePointsToSet(retKey);
          return getGenerator(retSrc, builder); // Recursive call for the callee's return value
        }
      } else if (def instanceof EachElementGetInstruction) {
        // We are iterating over a collection (e.g., for loop). Get the generator for the collection
        // itself.
        int iterableVn = def.getUse(0);
        PointerKey iterableKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, iterableVn);
        PointsToSetVariable iterableSrc =
            builder.getPropagationSystem().findOrCreatePointsToSet(iterableKey);
        TensorGenerator containerGenerator = getGenerator(iterableSrc, builder);

        // We have a generator for the container (the object being iterated over).
        // If the container is a `Dataset` (e.g., `tf.data.Dataset`), its generator
        // (`DatasetGenerator`) is defined to return the shapes/dtypes of its *elements* (not the
        // dataset object itself). Therefore, we use it directly.
        //
        // For `Tensors` (e.g., `tf.range`, constants), the generator returns the tensor's own
        // shape. When iterating, we must peel off the first dimension to get the element shape.
        return (containerGenerator instanceof DatasetGenerator)
            ? containerGenerator
            : new TensorElementGenerator(containerGenerator);
      } else if (def instanceof PythonPropertyRead) {
        // Python iteration may also be translated into property reads (e.g., retrieving an element
        // from a dataset or tensor).
        int objRef = ((PythonPropertyRead) def).getObjectRef();
        PointerKey objKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, objRef);
        PointsToSetVariable objSrc = builder.getPropagationSystem().findOrCreatePointsToSet(objKey);
        TensorGenerator containerGenerator = getGenerator(objSrc, builder);

        // Similar to `EachElementGet`, we check if the container generator represents elements
        // (`Dataset`) or the tensor itself (peeling needed).
        return (containerGenerator instanceof DatasetGenerator)
            ? containerGenerator
            : new TensorElementGenerator(containerGenerator);
      }
    }

    TypeReference calledFunction = getFunction(source, builder);
    LOGGER.info("Getting tensor generator for call to: " + calledFunction + ".");

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
    else if (isType(calledFunction, DATASET_BATCH_TYPE)) return new DatasetBatchGenerator(source);
    else if (isType(calledFunction, DATASET_RANGE_TYPE)) return new DatasetRangeGenerator(source);
    else if (isType(calledFunction, DATASET_FROM_GENERATOR_TYPE))
      return new DatasetFromGeneratorGenerator(source);
    else if (isType(calledFunction, DATASET_SHUFFLE_TYPE)
        || isType(calledFunction, DATASET_MAP_TYPE)
        || isType(calledFunction, DATASET_REPEAT_TYPE)
        || isType(calledFunction, DATASET_PREFETCH_TYPE)
        || isType(calledFunction, DATASET_TAKE_TYPE)
        || isType(calledFunction, DATASET_WITH_OPTIONS_TYPE)
        || isType(calledFunction, DATASET_CONCATENATE_TYPE)
        || isType(calledFunction, DATASET_ENUMERATE_TYPE)
        || isType(calledFunction, DATASET_REDUCE_TYPE)
        || isType(calledFunction, DATASET_FILTER_TYPE)
        || isType(calledFunction, DATASET)) return new DatasetGenerator(source);
    else if (isType(calledFunction, READ_DATA_SETS.getDeclaringClass()))
      return new ReadDataSets(source);
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
    else if (isType(calledFunction, DENSE.getDeclaringClass())) return new Dense(source);
    else if (isType(calledFunction, FLATTEN.getDeclaringClass())) return new Flatten(source);
    else if (isType(calledFunction, MAX_POOL.getDeclaringClass())) return new MaxPool(source);
    else
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
  }
}
