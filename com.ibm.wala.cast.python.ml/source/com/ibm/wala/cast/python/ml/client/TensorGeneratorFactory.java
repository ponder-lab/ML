package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARGMAX;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CAST;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT_OP_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_BATCH_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_MAP_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_RANGE_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SHUFFLE_TYPE;
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
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.INPUT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LOG;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MATMUL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MAX_POOL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MULTIPLY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NDARRAY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONES;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONE_HOT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.PLACEHOLDER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POISSON;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.READ_DATA_SETS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_MEAN;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.REDUCE_SUM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SOFTMAX_CROSS_ENTROPY_WITH_LOGITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_FROM_DENSE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SUBTRACT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.VARIABLE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS_LIKE;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.types.TypeReference;
import java.util.logging.Logger;

/**
 * A factory for creating TensorGenerator instances based on the called TensorFlow function.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorGeneratorFactory {

  private static final Logger LOGGER = getLogger(TensorGeneratorFactory.class.getName());

  public static TensorGenerator getGenerator(PointsToSetVariable source) {
    TypeReference calledFunction = getFunction(source);
    LOGGER.info("Getting tensor generator for call to: " + calledFunction.getName() + ".");

    if (calledFunction.equals(ONES.getDeclaringClass())) return new Ones(source);
    else if (calledFunction.equals(CONSTANT.getDeclaringClass())
        || calledFunction.equals(CONSTANT_OP_CONSTANT)) return new Constant(source);
    else if (calledFunction.equals(RANGE.getDeclaringClass())) return new Range(source);
    else if (calledFunction.equals(UNIFORM.getDeclaringClass())) return new Uniform(source);
    else if (calledFunction.equals(NORMAL.getDeclaringClass())) return new Normal(source);
    else if (calledFunction.equals(TRUNCATED_NORMAL.getDeclaringClass()))
      return new TruncatedNormal(source);
    else if (calledFunction.equals(ZEROS.getDeclaringClass())) return new Zeros(source);
    else if (calledFunction.equals(ZEROS_LIKE.getDeclaringClass())) return new ZerosLike(source);
    else if (calledFunction.equals(RESHAPE.getDeclaringClass())) return new Reshape(source);
    else if (calledFunction.equals(FILL.getDeclaringClass())) return new Fill(source);
    else if (calledFunction.equals(CONVERT_TO_TENSOR.getDeclaringClass()))
      return new ConvertToTensor(source);
    else if (calledFunction.equals(ONE_HOT.getDeclaringClass())) return new OneHot(source);
    else if (calledFunction.equals(EYE.getDeclaringClass())) return new Eye(source);
    else if (calledFunction.equals(SPARSE_EYE.getDeclaringClass())) return new SparseEye(source);
    else if (calledFunction.equals(SPARSE_TENSOR.getDeclaringClass()))
      return new SparseTensor(source);
    else if (calledFunction.equals(GAMMA.getDeclaringClass())) return new Gamma(source);
    else if (calledFunction.equals(INPUT.getDeclaringClass())) return new Input(source);
    else if (calledFunction.equals(POISSON.getDeclaringClass())) return new Poisson(source);
    else if (calledFunction.equals(RAGGED_CONSTANT.getDeclaringClass()))
      return new RaggedConstant(source);
    else if (calledFunction.equals(VARIABLE.getDeclaringClass())) return new Variable(source);
    else if (calledFunction.equals(RAGGED_RANGE.getDeclaringClass()))
      return new RaggedRange(source);
    else if (calledFunction.equals(FROM_VALUE_ROWIDS.getDeclaringClass()))
      return new RaggedFromValueRowIds(source);
    else if (calledFunction.equals(FROM_ROW_STARTS.getDeclaringClass()))
      return new RaggedFromRowStarts(source);
    else if (calledFunction.equals(FROM_ROW_SPLITS.getDeclaringClass()))
      return new RaggedFromRowSplits(source);
    else if (calledFunction.equals(FROM_ROW_LENGTHS.getDeclaringClass()))
      return new RaggedFromRowLengths(source);
    else if (calledFunction.equals(FROM_NESTED_ROW_LENGTHS.getDeclaringClass()))
      return new RaggedFromNestedRowLengths(source);
    else if (calledFunction.equals(FROM_NESTED_ROW_SPLITS.getDeclaringClass()))
      return new RaggedFromNestedRowSplits(source);
    else if (calledFunction.equals(FROM_NESTED_VALUE_ROWIDS.getDeclaringClass()))
      return new RaggedFromNestedValueRowIds(source);
    else if (calledFunction.equals(FROM_ROW_LIMITS.getDeclaringClass()))
      return new RaggedFromRowLimits(source);
    else if (calledFunction.equals(MULTIPLY.getDeclaringClass())
        || calledFunction.equals(ADD.getDeclaringClass())
        || calledFunction.equals(SUBTRACT.getDeclaringClass())
        || calledFunction.equals(DIVIDE.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (calledFunction.equals(SPARSE_ADD.getDeclaringClass())) return new SparseAdd(source);
    else if (calledFunction.equals(SPARSE_FROM_DENSE.getDeclaringClass()))
      return new SparseFromDense(source);
    else if (calledFunction.equals(MODEL.getDeclaringClass())) return new Model(source);
    else if (calledFunction.equals(TENSOR.getDeclaringClass())
        || calledFunction.equals(NDARRAY.getDeclaringClass())) return new TensorCall(source);
    else if (calledFunction.equals(DATASET_BATCH_TYPE)
        || calledFunction.equals(DATASET_SHUFFLE_TYPE)
        || calledFunction.equals(DATASET_MAP_TYPE)
        || calledFunction.equals(DATASET_RANGE_TYPE)
        || calledFunction.equals(DATASET_FROM_TENSOR_SLICES_TYPE)
        || calledFunction.equals(DATASET)) return new DatasetGenerator(source);
    else if (calledFunction.equals(READ_DATA_SETS.getDeclaringClass()))
      return new ReadDataSets(source);
    else if (calledFunction.equals(REDUCE_MEAN.getDeclaringClass())) return new ReduceMean(source);
    else if (calledFunction.equals(PLACEHOLDER.getDeclaringClass())) return new Placeholder(source);
    else if (calledFunction.equals(ARGMAX.getDeclaringClass())) return new ArgMax(source);
    else if (calledFunction.equals(EQUAL.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (calledFunction.equals(CAST.getDeclaringClass())) return new Cast(source);
    else if (calledFunction.equals(SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass())
        || calledFunction.equals(SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass()))
      return new SoftmaxCrossEntropy(source);
    else if (calledFunction.equals(LOG.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else if (calledFunction.equals(REDUCE_SUM.getDeclaringClass())) return new ReduceSum(source);
    else if (calledFunction.equals(MATMUL.getDeclaringClass())) return new MatMul(source);
    else if (calledFunction.equals(DENSE.getDeclaringClass())) return new Dense(source);
    else if (calledFunction.equals(FLATTEN.getDeclaringClass())) return new Flatten(source);
    else if (calledFunction.equals(MAX_POOL.getDeclaringClass())) return new MaxPool(source);
    else {
      if (calledFunction.getName().toString().startsWith("Lscript ")) {
        throw new IllegalArgumentException(
            "Encountered a tensor source in a script: "
                + calledFunction
                + ". This usually means a function call was not resolved to a summarized"
                + " function.");
      }
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
    }
  }
}
