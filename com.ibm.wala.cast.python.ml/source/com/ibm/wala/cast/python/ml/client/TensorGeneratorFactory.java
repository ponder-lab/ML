package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.NumpyTypes.ASTYPE;
import static com.ibm.wala.cast.python.ml.types.NumpyTypes.ASTYPE_METHOD_NAME;
import static com.ibm.wala.cast.python.ml.types.NumpyTypes.RESHAPE_METHOD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ACOSH;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADJOINT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARGMAX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARGMIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARRAY_OPS_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ASIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ASINH;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.AS_STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ATAN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ATAN2;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ATANH;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.BOOLEAN_MASK;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.BOSTON_HOUSING_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.BOSTON_HOUSING_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.BOSTON_HOUSING_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.BOSTON_HOUSING_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.BROADCAST_TO;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CAST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CEIL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR100_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR100_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR100_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR100_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR10_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR10_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR10_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CIFAR10_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CLIP_BY_VALUE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONCAT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.COS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.COSH;
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
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DIAG;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DIAG_PART;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DIVIDE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UNKNOWN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EINSUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EMBEDDING_LOOKUP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EQUAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ERF;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ERFC;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EXP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EXPAND_DIMS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EXPM1;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EXTRACT_PATCHES;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FASHION_MNIST_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FASHION_MNIST_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FASHION_MNIST_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FASHION_MNIST_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FILL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FLATTEN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FLATTEN_LAYER_CALL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FLOOR;
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
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GATHER_ND;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GRADIENT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GREATER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GREATER_EQUAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IDENTITY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IMDB_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IMDB_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IMDB_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.IMDB_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.INPUT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LEAKY_RELU;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LESS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LESS_EQUAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LINSPACE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LOG;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LOG1P;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LOG_SOFTMAX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MATMUL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MATRIX_TRANSPOSE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MAXIMUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MAX_POOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MESHGRID;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MINIMUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MNIST_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL_CALL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MULTIPLY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NDARRAY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NEGATIVE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NORMAL_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NOT_EQUAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONES;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONE_HOT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.PLACEHOLDER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POISSON;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POISSON_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POW;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANDOM_NORMAL_INIT_CALL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANK;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.READ_DATA_SETS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RECIPROCAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_ALL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_LOGSUMEXP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_MAX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_MEAN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_MIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_PROD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_SUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RELU;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REUTERS_X_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REUTERS_X_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REUTERS_Y_TEST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REUTERS_Y_TRAIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ROUND;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RSQRT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SEQUENCE_MASK;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SIGMOID;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SIGN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SIN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SINH;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SIZE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SLICE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SOFTMAX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SOFTPLUS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SOFTSIGN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_FROM_DENSE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SQRT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SQUARE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SQUEEZE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.STACK;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.STOP_GRADIENT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SUBTRACT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TAN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORDOT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_SCATTER_ND_UPDATE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TEXT_LINE_DATASET_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TF_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TILE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TOP_K;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRACE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRANSPOSE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL_METHOD_NAME;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM_OP;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNSORTED_SEGMENT_MAX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNSORTED_SEGMENT_MEAN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNSORTED_SEGMENT_SUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.VARIABLE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.WHERE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS_LIKE;
import static com.ibm.wala.cast.python.types.PythonTypes.SLICE_BUILTIN;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.sanitize;
import static java.util.Map.entry;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.ir.ssa.EachElementGetInstruction;
import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.classLoader.IClass;
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
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.graph.Graph;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.WeakHashMap;
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
      PROPERTY_NAME_GENERATORS =
          Map.ofEntries(
              entry(ASTYPE_METHOD_NAME, AstypeOperation::new),
              // wala/ML#449: `tf.random.truncated_normal(...)` doesn't reach the per-class
              // `isType` checks because `calledFunction` resolves to generic `LCodeBody`
              // rather than the specific `TRUNCATED_NORMAL`/`TRUNCATED_NORMAL_OP` class.
              // Duck-typing the property name catches it before the `ReadDataFallback`
              // fallback would.
              entry(TRUNCATED_NORMAL_METHOD_NAME, TruncatedNormal::new));

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

    LOGGER.fine("findCreator fallback returning original source: " + source);
    return source;
  }

  private static PointsToSetVariable getPointsToSetVariable(
      PointerKey key, PropagationCallGraphBuilder builder) {
    // Materializing an implicitly-represented key makes WALA dump the entire call graph's IR via an
    // unconditional debug print (wala/ML#573); skip it (the caller treats null as "no PTS").
    if (builder.getPropagationSystem().isImplicit(key)) return null;
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
      PointsToSetVariable source,
      PropagationCallGraphBuilder builder,
      Set<PointsToSetVariable> visited) {
    try {
      return getGenerator(source, builder, visited);
    } catch (IllegalArgumentException e) {
      LOGGER.log(Level.FINE, "tryGetGenerator: swallowed IAE for source=" + source, e);
      return null;
    }
  }

  /**
   * Returns {@code true} iff {@code source}'s defining instruction is an {@link
   * SSABinaryOpInstruction} and none of its operands have any tensor evidence.
   *
   * <p>Used by {@link #getGeneratorBody} to gate the {@code ElementWiseOperation} dispatch for
   * binop-defined sources: the TF semantics of element-wise add/sub/mul/div only apply when at
   * least one operand is itself a tensor. Without this gate, expressions like {@code n - 1} on
   * Python ints (or comparisons / arithmetic at any non-tensor binop site that happens to also land
   * in {@link #getDataflowSources}'s {@code SSABinaryOpInstruction} branch) get spuriously
   * classified as scalar tensors via the operand-PTS Integer-constant → INT32-dtype path in {@link
   * TensorGenerator#getDTypesOfValue}; under recursion, the spurious type back-propagates to the
   * parameter via the PA assignment graph at the recursive-call edge.
   *
   * <p>Tensor evidence is checked along three axes (per operand):
   *
   * <ol>
   *   <li><b>Implicit PK.</b> Summary-method returns whose PTS hasn't been materialised manifest as
   *       implicit pointer keys. Treat as tensor-relevant — gating on these would over-reject
   *       legitimate tensor binops chained through summary-returning ops.
   *   <li><b>Structural ({@link #tryGetGenerator}).</b> If the operand's own dispatch resolves to a
   *       generator (its def is a TF call, an EachElementGet over a tensor iterable, etc.), the
   *       binop is tensor-relevant.
   *   <li><b>PTS content.</b> If the operand's PTS contains any non-{@link ConstantKey} instance
   *       key (i.e. anything beyond Python {@code int}/{@code float}/{@code bool}/{@code str}
   *       literals), treat as tensor-relevant. This catches parameters whose PTS contains a tensor
   *       allocation flowed in from the caller (e.g. {@code replica_fn(tf.constant(3.0))} makes
   *       {@code input}'s PTS the {@code constant_op}'s {@link AllocationSiteInNode}); also keeps
   *       list/tuple-wrapped tensor flows working without re-implementing the catalog walk here.
   * </ol>
   *
   * <p>Returns {@code false} (i.e. <i>does</i> dispatch to {@code ElementWiseOperation}) for
   * non-binop sources: those reach this gate via the {@code tf.add}/{@code tf.subtract}/etc. TF API
   * path (whose calls have ADD/SUB/MUL/DIV declaring classes), and that path is the pre-existing
   * positive-identification entry that should not be restricted by this fix.
   *
   * @param source The candidate {@link ElementWiseOperation} source.
   * @param builder The propagation call graph builder, used to materialise operand {@link
   *     PointsToSetVariable}s and recurse into their generator dispatch.
   * @param visited The DFS recursion-stack set threaded through {@link #getGenerator}; passed
   *     unchanged to {@link #tryGetGenerator} so operand-side recursion participates in the same
   *     cycle guard as the parent dispatch.
   * @return {@code true} iff {@code source} is binop-defined and no operand has tensor evidence.
   */
  private static boolean isBinopWithoutTensorOperand(
      PointsToSetVariable source,
      PropagationCallGraphBuilder builder,
      Set<PointsToSetVariable> visited) {
    if (!(source.getPointerKey() instanceof LocalPointerKey)) return false;
    LocalPointerKey lpk = (LocalPointerKey) source.getPointerKey();
    CGNode node = lpk.getNode();
    SSAInstruction def = node.getDU().getDef(lpk.getValueNumber());
    if (!(def instanceof SSABinaryOpInstruction)) return false;
    for (int i = 0; i < def.getNumberOfUses(); i++) {
      int operandVn = def.getUse(i);
      if (operandHasTensorEvidence(operandVn, node, builder, visited)) return false;
    }
    return true;
  }

  /**
   * Returns {@code true} iff the operand at {@code operandVn} in {@code node} shows any sign of
   * being a tensor. Used by {@link #isBinopWithoutTensorOperand} to gate the binop → EWO dispatch.
   * See that method's Javadoc for the three-axis tensor-evidence definition.
   *
   * @param operandVn The operand SSA value number.
   * @param node The {@link CGNode} containing the binop.
   * @param builder The propagation call graph builder.
   * @param visited The DFS recursion-stack set threaded through {@link #getGenerator}.
   * @return {@code true} if the operand has tensor evidence; {@code false} otherwise.
   */
  private static boolean operandHasTensorEvidence(
      int operandVn,
      CGNode node,
      PropagationCallGraphBuilder builder,
      Set<PointsToSetVariable> visited) {
    // Python literal constant in the IR symbol table (e.g. `1` in `n - 1`,
    // `2.0` in `input * 2.0`) — not a tensor on its own. The companion operand
    // is what carries any tensor evidence. Check this BEFORE PTS-based
    // tensor-evidence inference: literals can have implicit/empty PTS
    // (depending on the substrate) that would otherwise be misread as
    // "summary-method return — conservatively a tensor."
    if (node.getIR().getSymbolTable().isConstant(operandVn)) return false;
    PointerKey pk =
        builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, operandVn);
    // Implicit pointer keys flag values whose PTS hasn't been materialised by
    // the propagation system — most commonly summary-method returns. Treat
    // those as potential tensors so legitimate chained tensor binops (e.g.
    // `tf.constant(...) - some_summary_op_result`) aren't over-rejected.
    if (builder.getPropagationSystem().isImplicit(pk)) return true;
    PointsToSetVariable operandSrc = getPointsToSetVariable(pk, builder);
    if (operandSrc != null && tryGetGenerator(operandSrc, builder, visited) != null) return true;
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(pk)) {
      if (!(ik instanceof ConstantKey)) return true;
    }
    return false;
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
    return getGenerator(source, builder, HashSetFactory.make());
  }

  /**
   * Cycle-guarded internal version of {@link #getGenerator(PointsToSetVariable,
   * PropagationCallGraphBuilder)}. Threads a set of already-visited sources through the recursive
   * walk so that self-referential functions (e.g. a recursive Python function whose return value
   * flows back into itself) don't drive {@code getGenerator} into unbounded recursion via the
   * return-value follow-through in this method's {@code SSAAbstractInvokeInstruction} handling
   * branch and the assignment-graph predecessor walk (the {@code ReturnValueKey} fallback). When a
   * {@link PointsToSetVariable} is re-encountered along the current call chain, this method returns
   * {@code null} so the outer dispatch loop can try other candidate callees / predecessors. See
   * wala/ML#435.
   *
   * @implNote {@code visited} is used as a DFS recursion-stack set: a source is added on entry and
   *     removed in a {@code finally} block before returning. This means only true cycles on the
   *     current call path short-circuit; sibling sub-dispatches in the same top-level call can
   *     re-visit a source freely without losing precision.
   */
  private static TensorGenerator getGenerator(
      PointsToSetVariable source,
      PropagationCallGraphBuilder builder,
      Set<PointsToSetVariable> visited) {
    source = findCreator(source, builder);
    if (!visited.add(source)) {
      final PointsToSetVariable cycleSource = source;
      LOGGER.log(
          Level.FINE,
          () ->
              "getGenerator: cycle detected at source="
                  + cycleSource
                  + "; returning null so dispatch can try other branches.");
      return null;
    }
    final PointsToSetVariable visitedSource = source;
    try {
      return getGeneratorBody(source, builder, visited);
    } finally {
      visited.remove(visitedSource);
    }
  }

  /**
   * Body of the cycle-guarded {@link #getGenerator(PointsToSetVariable,
   * PropagationCallGraphBuilder, Set)}, extracted so the cycle guard can wrap the entire dispatch
   * in a single {@code try}/{@code finally} (add on entry, remove on exit).
   */
  private static TensorGenerator getGeneratorBody(
      PointsToSetVariable source,
      PropagationCallGraphBuilder builder,
      Set<PointsToSetVariable> visited) {
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
                ? new EnumerateGenerator(source, tryGetGenerator(iterableSrc, builder, visited))
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
                ? new IteratorGenerator(source, tryGetGenerator(iterableSrc, builder, visited))
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
            TensorGenerator containerGenerator = tryGetGenerator(iterableSrc, builder, visited);

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
                      containerGenerator = tryGetGenerator(iterArgSrc, builder, visited);
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
            TensorGenerator fromRet = tryGetGenerator(retSrc, builder, visited);
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
        TensorGenerator containerGenerator = tryGetGenerator(iterableSrc, builder, visited);

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
        TensorGenerator containerGenerator = tryGetGenerator(objSrc, builder, visited);

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
    LOGGER.fine("Getting tensor generator for call to: " + calledFunction + ".");

    // sanitize the type name by removing the artificial suffix that is added for synthetic
    // classes to facilitate trampoline generation.
    calledFunction = sanitize(calledFunction);

    LOGGER.fine("Getting tensor generator for sanitized call to: " + calledFunction + ".");

    if (isType(calledFunction, ONES.getDeclaringClass())) return new Ones(source);
    else if (isType(calledFunction, CONSTANT.getDeclaringClass())) return new Constant(source);
    else if (isType(calledFunction, RANGE.getDeclaringClass())) return new Range(source);
    else if (isType(calledFunction, UNIFORM.getDeclaringClass())
        || isType(calledFunction, UNIFORM_OP)) return new Uniform(source);
    else if (isType(calledFunction, RANDOM_NORMAL_INIT_CALL.getDeclaringClass()))
      return new RandomNormalCall(source);
    else if (isType(calledFunction, NORMAL.getDeclaringClass())
        || isType(calledFunction, NORMAL_OP)) return new Normal(source);
    else if (isType(calledFunction, TRUNCATED_NORMAL.getDeclaringClass())
        || isType(calledFunction, TRUNCATED_NORMAL_OP)) return new TruncatedNormal(source);
    else if (isType(calledFunction, ZEROS.getDeclaringClass())) return new Zeros(source);
    else if (isType(calledFunction, ZEROS_LIKE.getDeclaringClass())) return new ZerosLike(source);
    else if (isType(calledFunction, ARRAY_OPS_RESHAPE)
        || calledFunction.getName().equals(TF_RESHAPE)) return new Reshape(source);
    else if (isType(calledFunction, FILL.getDeclaringClass())) return new Fill(source);
    else if (isType(calledFunction, LINSPACE.getDeclaringClass())) return new Linspace(source);
    else if (isType(calledFunction, BROADCAST_TO.getDeclaringClass()))
      return new BroadcastTo(source);
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
        || isType(calledFunction, DIVIDE.getDeclaringClass())) {
      // For SSABinaryOp sources (`a - b` rather than `tf.subtract(a, b)`), gate
      // ElementWiseOperation dispatch on at least one operand showing tensor
      // evidence — see {@link #operandHasTensorEvidence} for the three-axis
      // definition (implicit PK / structural `tryGetGenerator` resolution /
      // non-`ConstantKey` PTS content). Without this gate, every Python-int
      // arithmetic expression (e.g. `n - 1` in a recursive function whose only
      // call sites pass ints) gets classified as a tensor — `getDataflowSources`
      // adds every binary-op result as a candidate source, and EWO's "always a
      // tensor" assumption turns Integer constants in operand PTS into INT32
      // dtypes, which then back-propagate to parameters via the PA assignment
      // graph at the recursive-call edge. The TF-API path (`tf.add(...)`, etc.)
      // is unaffected: those are SSAAbstractInvoke, not SSABinaryOp, so the
      // structural check below skips them. See wala/ML#451.
      if (isBinopWithoutTensorOperand(source, builder, visited)) {
        LOGGER.fine(
            () ->
                "Rejecting ElementWiseOperation dispatch — no operand has tensor"
                    + " evidence. source="
                    + source);
        throw new IllegalArgumentException("Binary op with no tensor operand: " + source + ".");
      }
      return new ElementWiseOperation(source);
    } else if (isType(calledFunction, SPARSE_ADD.getDeclaringClass())) return new SparseAdd(source);
    else if (isType(calledFunction, SPARSE_FROM_DENSE.getDeclaringClass()))
      return new SparseFromDense(source);
    else if (isType(calledFunction, MODEL.getDeclaringClass())) return new Model(source);
    else if (isType(calledFunction, TENSOR.getDeclaringClass())
        || isType(calledFunction, NDARRAY.getDeclaringClass())) return new TensorCall(source);
    else if (isType(calledFunction, NumpyTypes.ARRAY.getDeclaringClass()))
      return new NpArray(source);
    else if (isType(calledFunction, NumpyTypes.ONES.getDeclaringClass())) return new NpOnes(source);
    else if (isType(calledFunction, NumpyTypes.ZEROS.getDeclaringClass()))
      return new NpZeros(source);
    else if (isType(calledFunction, NumpyTypes.RESHAPE.getDeclaringClass()))
      return new NpReshape(source);
    else if (isType(calledFunction, DATASET_FROM_TENSOR_SLICES_TYPE))
      return new DatasetFromTensorSlicesGenerator(source);
    else if (isType(calledFunction, DATASET_FROM_TENSORS_TYPE))
      return new DatasetFromTensorsGenerator(source);
    else if (isType(calledFunction, DATASET_BATCH_TYPE)) return new DatasetBatchGenerator(source);
    else if (isType(calledFunction, DATASET_RANGE_TYPE)) return new DatasetRangeGenerator(source);
    else if (isType(calledFunction, TEXT_LINE_DATASET_TYPE))
      return new TextLineDatasetGenerator(source);
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
    else if (isType(calledFunction, CIFAR10_X_TRAIN))
      return new Cifar10InputData(source, Cifar10InputData.X_TRAIN_SHAPE);
    else if (isType(calledFunction, CIFAR10_Y_TRAIN))
      return new Cifar10InputData(source, Cifar10InputData.Y_TRAIN_SHAPE);
    else if (isType(calledFunction, CIFAR10_X_TEST))
      return new Cifar10InputData(source, Cifar10InputData.X_TEST_SHAPE);
    else if (isType(calledFunction, CIFAR10_Y_TEST))
      return new Cifar10InputData(source, Cifar10InputData.Y_TEST_SHAPE);
    else if (isType(calledFunction, IMDB_X_TRAIN))
      return new ImdbInputData(source, ImdbInputData.X_TRAIN_SHAPE, ImdbInputData.X_DTYPES);
    else if (isType(calledFunction, IMDB_Y_TRAIN))
      return new ImdbInputData(source, ImdbInputData.Y_TRAIN_SHAPE, ImdbInputData.Y_DTYPES);
    else if (isType(calledFunction, IMDB_X_TEST))
      return new ImdbInputData(source, ImdbInputData.X_TEST_SHAPE, ImdbInputData.X_DTYPES);
    else if (isType(calledFunction, IMDB_Y_TEST))
      return new ImdbInputData(source, ImdbInputData.Y_TEST_SHAPE, ImdbInputData.Y_DTYPES);
    else if (isType(calledFunction, FASHION_MNIST_X_TRAIN))
      return new MnistInputData(source, MnistInputData.X_TRAIN_SHAPE);
    else if (isType(calledFunction, FASHION_MNIST_Y_TRAIN))
      return new MnistInputData(source, MnistInputData.Y_TRAIN_SHAPE);
    else if (isType(calledFunction, FASHION_MNIST_X_TEST))
      return new MnistInputData(source, MnistInputData.X_TEST_SHAPE);
    else if (isType(calledFunction, FASHION_MNIST_Y_TEST))
      return new MnistInputData(source, MnistInputData.Y_TEST_SHAPE);
    else if (isType(calledFunction, CIFAR100_X_TRAIN))
      return new Cifar100InputData(
          source, Cifar100InputData.X_TRAIN_SHAPE, Cifar100InputData.X_DTYPES);
    else if (isType(calledFunction, CIFAR100_Y_TRAIN))
      return new Cifar100InputData(
          source, Cifar100InputData.Y_TRAIN_SHAPE, Cifar100InputData.Y_DTYPES);
    else if (isType(calledFunction, CIFAR100_X_TEST))
      return new Cifar100InputData(
          source, Cifar100InputData.X_TEST_SHAPE, Cifar100InputData.X_DTYPES);
    else if (isType(calledFunction, CIFAR100_Y_TEST))
      return new Cifar100InputData(
          source, Cifar100InputData.Y_TEST_SHAPE, Cifar100InputData.Y_DTYPES);
    else if (isType(calledFunction, REUTERS_X_TRAIN))
      return new ReutersInputData(
          source, ReutersInputData.X_TRAIN_SHAPE, ReutersInputData.X_DTYPES);
    else if (isType(calledFunction, REUTERS_Y_TRAIN))
      return new ReutersInputData(
          source, ReutersInputData.Y_TRAIN_SHAPE, ReutersInputData.Y_DTYPES);
    else if (isType(calledFunction, REUTERS_X_TEST))
      return new ReutersInputData(source, ReutersInputData.X_TEST_SHAPE, ReutersInputData.X_DTYPES);
    else if (isType(calledFunction, REUTERS_Y_TEST))
      return new ReutersInputData(source, ReutersInputData.Y_TEST_SHAPE, ReutersInputData.Y_DTYPES);
    else if (isType(calledFunction, BOSTON_HOUSING_X_TRAIN))
      return new BostonHousingInputData(source, BostonHousingInputData.X_TRAIN_SHAPE);
    else if (isType(calledFunction, BOSTON_HOUSING_Y_TRAIN))
      return new BostonHousingInputData(source, BostonHousingInputData.Y_TRAIN_SHAPE);
    else if (isType(calledFunction, BOSTON_HOUSING_X_TEST))
      return new BostonHousingInputData(source, BostonHousingInputData.X_TEST_SHAPE);
    else if (isType(calledFunction, BOSTON_HOUSING_Y_TEST))
      return new BostonHousingInputData(source, BostonHousingInputData.Y_TEST_SHAPE);
    else if (isType(calledFunction, REDUCE_MEAN.getDeclaringClass())) return new ReduceMean(source);
    else if (isType(calledFunction, REDUCE_MAX.getDeclaringClass())) return new ReduceMax(source);
    else if (isType(calledFunction, REDUCE_MIN.getDeclaringClass())) return new ReduceMin(source);
    else if (isType(calledFunction, REDUCE_PROD.getDeclaringClass())) return new ReduceProd(source);
    else if (isType(calledFunction, REDUCE_LOGSUMEXP.getDeclaringClass()))
      return new ReduceLogSumExp(source);
    else if (isType(calledFunction, UNSORTED_SEGMENT_SUM.getDeclaringClass())
        || isType(calledFunction, UNSORTED_SEGMENT_MAX.getDeclaringClass())
        || isType(calledFunction, UNSORTED_SEGMENT_MEAN.getDeclaringClass()))
      return new UnsortedSegmentReduction(source);
    else if (isType(calledFunction, REDUCE_ALL.getDeclaringClass())) return new ReduceAll(source);
    else if (isType(calledFunction, PLACEHOLDER.getDeclaringClass()))
      return new Placeholder(source);
    else if (isType(calledFunction, EQUAL.getDeclaringClass()))
      return new ComparisonOperation(source);
    else if (isType(calledFunction, NOT_EQUAL.getDeclaringClass()))
      return new ComparisonOperation(source);
    else if (isType(calledFunction, LESS.getDeclaringClass()))
      return new ComparisonOperation(source);
    else if (isType(calledFunction, LESS_EQUAL.getDeclaringClass()))
      return new ComparisonOperation(source);
    else if (isType(calledFunction, GREATER.getDeclaringClass()))
      return new ComparisonOperation(source);
    else if (isType(calledFunction, GREATER_EQUAL.getDeclaringClass()))
      return new ComparisonOperation(source);
    else if (isType(calledFunction, SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass())
        || isType(calledFunction, SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass()))
      return new SoftmaxCrossEntropy(source);
    else if (isType(calledFunction, REDUCE_SUM.getDeclaringClass())) return new ReduceSum(source);
    else if (isType(calledFunction, MATMUL.getDeclaringClass())) return new MatMul(source);
    else if (isType(calledFunction, SIGMOID.getDeclaringClass())) return new Sigmoid(source);
    else if (isType(calledFunction, EXP.getDeclaringClass())) return new Exp(source);
    else if (isType(calledFunction, RSQRT.getDeclaringClass())) return new Rsqrt(source);
    else if (isType(calledFunction, LOG_SOFTMAX.getDeclaringClass())) return new LogSoftmax(source);
    else if (isType(calledFunction, RANK.getDeclaringClass())) return new Rank(source);
    else if (isType(calledFunction, SIZE.getDeclaringClass())) return new Size(source);
    else if (isType(calledFunction, ARGMAX.getDeclaringClass())) return new Argmax(source);
    else if (isType(calledFunction, ARGMIN.getDeclaringClass())) return new Argmin(source);
    else if (isType(calledFunction, TENSORDOT.getDeclaringClass())) return new Tensordot(source);
    else if (isType(calledFunction, TRACE.getDeclaringClass())) return new Trace(source);
    else if (isType(calledFunction, TRANSPOSE.getDeclaringClass())) return new Transpose(source);
    else if (isType(calledFunction, DIAG.getDeclaringClass())) return new Diag(source);
    else if (isType(calledFunction, DIAG_PART.getDeclaringClass())) return new DiagPart(source);
    else if (isType(calledFunction, MATRIX_TRANSPOSE.getDeclaringClass()))
      return new MatrixTranspose(source);
    else if (isType(calledFunction, ADJOINT.getDeclaringClass())) return new Adjoint(source);
    else if (isType(calledFunction, TILE.getDeclaringClass())) return new Tile(source);
    else if (isType(calledFunction, TENSOR_SCATTER_ND_UPDATE.getDeclaringClass()))
      return new TensorScatterNdUpdate(source);
    else if (isType(calledFunction, SEQUENCE_MASK.getDeclaringClass()))
      return new SequenceMask(source);
    else if (isType(calledFunction, EMBEDDING_LOOKUP.getDeclaringClass()))
      return new EmbeddingLookup(source);
    else if (isType(calledFunction, GATHER_ND.getDeclaringClass())) return new GatherNd(source);
    else if (isType(calledFunction, BOOLEAN_MASK.getDeclaringClass()))
      return new BooleanMask(source);
    else if (isType(calledFunction, SLICE.getDeclaringClass())) return new Slice(source);
    else if (isType(calledFunction, SQUEEZE.getDeclaringClass())) return new Squeeze(source);
    else if (isType(calledFunction, EXTRACT_PATCHES.getDeclaringClass()))
      return new ExtractPatches(source);
    else if (isType(calledFunction, TAN.getDeclaringClass())) return new Tan(source);
    else if (isType(calledFunction, ASIN.getDeclaringClass())) return new Asin(source);
    else if (isType(calledFunction, ATAN.getDeclaringClass())) return new Atan(source);
    else if (isType(calledFunction, SINH.getDeclaringClass())) return new Sinh(source);
    else if (isType(calledFunction, COSH.getDeclaringClass())) return new Cosh(source);
    else if (isType(calledFunction, ASINH.getDeclaringClass())) return new Asinh(source);
    else if (isType(calledFunction, ACOSH.getDeclaringClass())) return new Acosh(source);
    else if (isType(calledFunction, ATANH.getDeclaringClass())) return new Atanh(source);
    else if (isType(calledFunction, LOG1P.getDeclaringClass())) return new Log1p(source);
    else if (isType(calledFunction, EXPM1.getDeclaringClass())) return new Expm1(source);
    else if (isType(calledFunction, ROUND.getDeclaringClass())) return new Round(source);
    else if (isType(calledFunction, RECIPROCAL.getDeclaringClass())) return new Reciprocal(source);
    else if (isType(calledFunction, SOFTPLUS.getDeclaringClass())) return new Softplus(source);
    else if (isType(calledFunction, SOFTSIGN.getDeclaringClass())) return new Softsign(source);
    else if (isType(calledFunction, SQUARE.getDeclaringClass())) return new Square(source);
    else if (isType(calledFunction, ERF.getDeclaringClass())) return new Erf(source);
    else if (isType(calledFunction, ERFC.getDeclaringClass())) return new Erfc(source);
    else if (isType(calledFunction, ATAN2.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, MAXIMUM.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, MINIMUM.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, EINSUM.getDeclaringClass())) return new Einsum(source);
    else if (isType(calledFunction, RELU.getDeclaringClass())) return new Relu(source);
    else if (isType(calledFunction, CAST.getDeclaringClass())) return new Cast(source);
    else if (isType(calledFunction, EXPAND_DIMS.getDeclaringClass())) return new ExpandDims(source);
    else if (isType(calledFunction, CLIP_BY_VALUE.getDeclaringClass()))
      return new ClipByValue(source);
    else if (isType(calledFunction, AS_STRING.getDeclaringClass())) return new AsString(source);
    else if (isType(calledFunction, TOP_K.getDeclaringClass())) return new TopK(source);
    else if (isType(calledFunction, MESHGRID.getDeclaringClass())) return new Meshgrid(source);
    else if (isType(calledFunction, WHERE.getDeclaringClass())) return new Where(source);
    else if (isType(calledFunction, LEAKY_RELU.getDeclaringClass())) return new LeakyRelu(source);
    else if (isType(calledFunction, POW.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (isType(calledFunction, CONCAT.getDeclaringClass())) return new Concat(source);
    else if (isType(calledFunction, STACK.getDeclaringClass())) return new Stack(source);
    else if (isType(calledFunction, SQRT.getDeclaringClass())) return new Sqrt(source);
    else if (isType(calledFunction, LOG.getDeclaringClass())) return new Log(source);
    else if (isType(calledFunction, NEGATIVE.getDeclaringClass())) return new Negative(source);
    else if (isType(calledFunction, SIN.getDeclaringClass())) return new Sin(source);
    else if (isType(calledFunction, COS.getDeclaringClass())) return new Cos(source);
    else if (isType(calledFunction, FLOOR.getDeclaringClass())) return new Floor(source);
    else if (isType(calledFunction, CEIL.getDeclaringClass())) return new Ceil(source);
    else if (isType(calledFunction, SIGN.getDeclaringClass())) return new Sign(source);
    else if (isType(calledFunction, IDENTITY.getDeclaringClass())) return new Identity(source);
    else if (isType(calledFunction, STOP_GRADIENT.getDeclaringClass()))
      return new StopGradient(source);
    else if (isType(calledFunction, GRADIENT.getDeclaringClass())) return new Gradient(source);
    else if (isType(calledFunction, SOFTMAX.getDeclaringClass())) return new Softmax(source);
    else if (isType(calledFunction, DENSE_CALL.getDeclaringClass())) return new DenseCall(source);
    else if (isType(calledFunction, MODEL_CALL.getDeclaringClass())) return new ModelCall(source);
    else if (isType(calledFunction, FLATTEN.getDeclaringClass())) return new Flatten(source);
    else if (isType(calledFunction, FLATTEN_LAYER_CALL.getDeclaringClass()))
      return new FlattenCall(source);
    else if (isType(calledFunction, MAX_POOL.getDeclaringClass())) return new MaxPool(source);
    else if (isType(calledFunction, SLICE_BUILTIN)) return new SliceBuiltinOperation(source);
    else {
      if (source.getPointerKey() instanceof ReturnValueKey) {
        Graph<PointsToSetVariable> assignmentGraph =
            builder.getPropagationSystem().getAssignmentGraph();
        for (Iterator<PointsToSetVariable> it = assignmentGraph.getPredNodes(source);
            it.hasNext(); ) {
          PointsToSetVariable pred = it.next();
          try {
            TensorGenerator gen = getGenerator(pred, builder, visited);
            if (gen != null) {
              return gen;
            }
          } catch (IllegalArgumentException ex) {
            // Ignore and continue searching other predecessors.
          }
        }
      }
      // Fallback for `read_data`-pattern XML classes (#437, #380): when the
      // dispatch chain above has no entry, check whether the call's
      // function-object came from a `PythonPropertyRead` whose member-name
      // matches the parent class of an `Ltensorflow/.../<name>/read_data`
      // trampoline. If so, classify as a generic ⊤-shape / UNKNOWN-dtype
      // tensor source — restoring pre-branch-267 identification semantics
      // for ops not yet ported to dedicated `TensorGenerator` subclasses.
      // Specific dispatch entries are checked first; this only fires when
      // nothing in the table matched.
      //
      // The trampoline lookup is restricted to the `Ltensorflow/` namespace
      // so that `np.argmax(...)` and similar non-TF property reads aren't
      // misclassified even if a future numpy.xml ever introduces an
      // `argmax` class. All candidate constant-string member-names from the
      // points-to set are checked, not just the first, to avoid
      // iteration-order-dependent behavior on multi-candidate sites.
      for (String propertyName : getPropertyReadMemberNames(source, builder)) {
        if (!getTensorflowReadDataPropertyNames(builder).contains(propertyName)) continue;
        final String capturedName = propertyName;
        final TypeReference capturedFunction = calledFunction;
        LOGGER.fine(
            () ->
                "TensorGeneratorFactory: dispatching `."
                    + capturedName
                    + "(...)` (calledFunction="
                    + capturedFunction
                    + ") via `read_data`-pattern trampoline-class lookup;"
                    + " no dispatch-table entry — falling back to generic ⊤-shape /"
                    + " UNKNOWN-dtype tensor source.");
        return new ReadDataFallback(source);
      }
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
    }
  }

  /**
   * Returns the set of constant-string member-names that the call's function-object could resolve
   * to via a {@link PythonPropertyRead}. Empty if the call's function-object isn't a property read
   * or no constant strings are in the member-ref points-to set.
   *
   * <p>Used by the `read_data`-pattern fallback to walk back from a call site to the property-read
   * that resolved its function object. Returns all candidate names rather than just the first, so
   * downstream predicate checks are not iteration-order-dependent on multi-candidate sites.
   */
  private static Set<String> getPropertyReadMemberNames(
      PointsToSetVariable source, PropagationCallGraphBuilder builder) {
    PointerKey k = source.getPointerKey();
    if (!(k instanceof LocalPointerKey)) return Collections.emptySet();
    LocalPointerKey lpk = (LocalPointerKey) k;
    CGNode node = lpk.getNode();
    SSAInstruction def = node.getDU().getDef(lpk.getValueNumber());
    if (!(def instanceof SSAAbstractInvokeInstruction)) return Collections.emptySet();
    SSAAbstractInvokeInstruction call = (SSAAbstractInvokeInstruction) def;
    if (call.getNumberOfUses() == 0) return Collections.emptySet();
    SSAInstruction funcDef = node.getDU().getDef(call.getUse(0));
    if (!(funcDef instanceof PythonPropertyRead)) return Collections.emptySet();
    PythonPropertyRead funcRead = (PythonPropertyRead) funcDef;
    PointerKey memberKey =
        builder
            .getPointerAnalysis()
            .getHeapModel()
            .getPointerKeyForLocal(node, funcRead.getMemberRef());
    Set<String> names = HashSetFactory.make();
    for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(memberKey)) {
      if (!(ik instanceof ConstantKey)) continue;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (value instanceof String) names.add((String) value);
    }
    return names;
  }

  /**
   * Per-{@link IClassHierarchy} memoized cache of property names whose {@code read_data} trampoline
   * classes exist in the {@code Ltensorflow/} namespace. Computed once per analysis builder via a
   * single class-hierarchy scan and reused thereafter so the {@code read_data}-pattern fallback's
   * predicate check is O(1) per call instead of O(|CHA|).
   */
  private static final WeakHashMap<IClassHierarchy, Set<String>>
      TENSORFLOW_READ_DATA_PROPERTY_NAMES_CACHE = new WeakHashMap<>();

  /**
   * Returns the set of property names whose {@code Ltensorflow/.../<name>/read_data} trampoline
   * class exists in the given {@link PropagationCallGraphBuilder}'s class hierarchy.
   *
   * <p>XML-registered helper methods like {@code read_data} get transformed by {@link
   * com.ibm.wala.cast.python.client.PythonAnalysisEngine#addSummaryBypassLogic} into synthetic
   * trampoline classes nested under the parent (per the {@code /class/}-namespace trampoline
   * pattern): {@code <class>.read_data} becomes a new {@code Ltensorflow/.../<class>/read_data}
   * class with a {@code do} method. So a class-hierarchy scan for a {@code read_data} method on the
   * parent class via {@link IClass#getDeclaredMethods()} returns nothing — we have to look for the
   * trampoline class instead.
   *
   * <p>Restricted to the {@code Ltensorflow/} namespace so non-TF property reads (e.g. {@code
   * np.argmax(...)}) aren't misclassified even if a future {@code numpy.xml} introduces a class
   * with the same simple name. Memoized via {@link #TENSORFLOW_READ_DATA_PROPERTY_NAMES_CACHE} to
   * avoid repeating the O(|CHA|) scan on every fallback check.
   */
  private static Set<String> getTensorflowReadDataPropertyNames(
      PropagationCallGraphBuilder builder) {
    IClassHierarchy cha = builder.getClassHierarchy();
    synchronized (TENSORFLOW_READ_DATA_PROPERTY_NAMES_CACHE) {
      Set<String> cached = TENSORFLOW_READ_DATA_PROPERTY_NAMES_CACHE.get(cha);
      if (cached != null) return cached;
    }
    String suffix = "/" + PythonTensorAnalysisEngine.TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME;
    Set<String> names = HashSetFactory.make();
    for (IClass cls : cha) {
      String typeName = cls.getName().toString();
      if (!typeName.startsWith("Ltensorflow/")) continue;
      if (!typeName.endsWith(suffix)) continue;
      int suffixStart = typeName.length() - suffix.length();
      int sep = typeName.lastIndexOf('/', suffixStart - 1);
      if (sep < 0) continue;
      names.add(typeName.substring(sep + 1, suffixStart));
    }
    synchronized (TENSORFLOW_READ_DATA_PROPERTY_NAMES_CACHE) {
      TENSORFLOW_READ_DATA_PROPERTY_NAMES_CACHE.put(cha, names);
    }
    return names;
  }

  /**
   * Generic fallback {@link TensorGenerator} for XML-modeled tensor-producing APIs that follow the
   * {@code read_data} pattern but don't have a dedicated subclass yet (per #380). Returns ⊤ shape
   * and {@code UNKNOWN} dtype — the API is identified as a tensor source, but no precise type is
   * claimed.
   *
   * <p>Distinct from {@link Constant}, which inspects the call's value parameter to infer
   * shape/dtype: that inspection can return the input tensor's dimensions rather than the op's
   * actual output dimensions, giving misleading precision for ops where {@code Constant}'s value
   * model doesn't apply. Per-op subclasses with proper precision can replace this fallback as
   * they're added; the fallback is just the safety net during the migration.
   *
   * <p>TODO: remove this class (and its callers in {@link #getGeneratorBody}, plus the helper
   * methods {@link #getPropertyReadMemberNames} and {@link #getTensorflowReadDataPropertyNames})
   * once the {@code read_data} pattern is fully migrated out of {@code tensorflow.xml} — at that
   * point the {@code Ltensorflow/.../<op>/read_data} trampoline classes this fallback keys off
   * disappear from the class hierarchy and the predicate becomes a no-op anyway. Tracked in #380.
   */
  private static final class ReadDataFallback extends TensorGenerator {
    ReadDataFallback(PointsToSetVariable source) {
      super(source);
    }

    @Override
    protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
      return null;
    }

    @Override
    protected int getShapeParameterPosition() {
      return -1;
    }

    @Override
    protected String getShapeParameterName() {
      return null;
    }

    @Override
    protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
      return EnumSet.of(UNKNOWN);
    }

    @Override
    protected int getDTypeParameterPosition() {
      return -1;
    }

    @Override
    protected String getDTypeParameterName() {
      return null;
    }
  }
}
