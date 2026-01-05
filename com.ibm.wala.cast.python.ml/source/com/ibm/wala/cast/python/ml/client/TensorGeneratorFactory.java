package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FILL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.GAMMA;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MULTIPLY;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONES;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ONE_HOT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.POISSON;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TRUNCATED_NORMAL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS_LIKE;
import static com.ibm.wala.cast.python.util.Util.getFunction;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.types.TypeReference;
import java.util.logging.Logger;

/**
 * A factory for creating TensorGenerator instances based on the called TensorFlow function.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorGeneratorFactory {

  private static final Logger LOGGER = Logger.getLogger(TensorGeneratorFactory.class.getName());

  public static TensorGenerator getGenerator(PointsToSetVariable source) {
    TypeReference calledFunction = getFunction(source);
    LOGGER.info("Getting tensor generator for call to: " + calledFunction.getName() + ".");

    if (calledFunction.equals(ONES.getDeclaringClass())) return new Ones(source);
    else if (calledFunction.equals(CONSTANT.getDeclaringClass())) return new Constant(source);
    else if (calledFunction.equals(RANGE.getDeclaringClass())) return new Range(source);
    else if (calledFunction.equals(UNIFORM.getDeclaringClass())) return new Uniform(source);
    else if (calledFunction.equals(NORMAL.getDeclaringClass())) return new Normal(source);
    else if (calledFunction.equals(TRUNCATED_NORMAL.getDeclaringClass()))
      return new TruncatedNormal(source);
    else if (calledFunction.equals(ZEROS.getDeclaringClass())) return new Zeros(source);
    else if (calledFunction.equals(ZEROS_LIKE.getDeclaringClass())) return new ZerosLike(source);
    else if (calledFunction.equals(FILL.getDeclaringClass())) return new Fill(source);
    else if (calledFunction.equals(CONVERT_TO_TENSOR.getDeclaringClass()))
      return new ConvertToTensor(source);
    else if (calledFunction.equals(ONE_HOT.getDeclaringClass())) return new OneHot(source);
    else if (calledFunction.equals(EYE.getDeclaringClass())) return new Eye(source);
    else if (calledFunction.equals(SPARSE_EYE.getDeclaringClass())) return new SparseEye(source);
    else if (calledFunction.equals(SPARSE_TENSOR.getDeclaringClass()))
      return new SparseTensor(source);
    else if (calledFunction.equals(GAMMA.getDeclaringClass())) return new Gamma(source);
    else if (calledFunction.equals(POISSON.getDeclaringClass())) return new Poisson(source);
    else if (calledFunction.equals(RAGGED_CONSTANT.getDeclaringClass()))
      return new RaggedConstant(source);
    else if (calledFunction.equals(RAGGED_RANGE.getDeclaringClass()))
      return new RaggedRange(source);
    else if (calledFunction.equals(MULTIPLY.getDeclaringClass())
        || calledFunction.equals(
            com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADD.getDeclaringClass())
        || calledFunction.equals(
            com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SUBTRACT.getDeclaringClass())
        || calledFunction.equals(
            com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DIVIDE.getDeclaringClass()))
      return new ElementWiseOperation(source);
    else
      throw new IllegalArgumentException(
          "Unknown call: " + calledFunction + " for source: " + source + ".");
  }
}
