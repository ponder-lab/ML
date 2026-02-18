package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ADD;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARRAY_OPS_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
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
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.UNIFORM;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.VARIABLE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ZEROS_LIKE;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static java.util.logging.Logger.getLogger;

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
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.TypeReference;
import java.util.logging.Logger;

/**
 * A factory for creating TensorGenerator instances based on the called TensorFlow function.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorGeneratorFactory {

  private static final Logger LOGGER = getLogger(TensorGeneratorFactory.class.getName());

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
      }
      return lpk.getNode().getMethod().getDeclaringClass().getReference();
    } else if (k instanceof ReturnValueKey) {
      return ((ReturnValueKey) k).getNode().getMethod().getDeclaringClass().getReference();
    }
    return Util.getFunction(source);
  }

  private static boolean isType(TypeReference tr, TypeReference expected) {
    if (tr == null || expected == null) return false;
    return tr.getName().equals(expected.getName());
  }

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
          PointerKey retKey =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForReturnValue(callee);
          PointsToSetVariable retSrc =
              builder.getPropagationSystem().findOrCreatePointsToSet(retKey);
          return getGenerator(retSrc, builder); // Recursive call for the callee's return value
        }
      }
    }

    TypeReference calledFunction = getFunction(source, builder);
    LOGGER.info("Getting tensor generator for call to: " + calledFunction + ".");

    if (isType(calledFunction, ONES.getDeclaringClass())) return new Ones(source);
    else if (isType(calledFunction, CONSTANT.getDeclaringClass())) return new Constant(source);
    else if (isType(calledFunction, RANGE.getDeclaringClass())) return new Range(source);
    else if (isType(calledFunction, UNIFORM.getDeclaringClass())) return new Uniform(source);
    else if (isType(calledFunction, NORMAL.getDeclaringClass())) return new Normal(source);
    else if (isType(calledFunction, TRUNCATED_NORMAL.getDeclaringClass()))
      return new TruncatedNormal(source);
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
    else if (isType(calledFunction, GAMMA.getDeclaringClass())) return new Gamma(source);
    else if (isType(calledFunction, INPUT.getDeclaringClass())) return new Input(source);
    else if (isType(calledFunction, POISSON.getDeclaringClass())) return new Poisson(source);
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
    else if (isType(calledFunction, DATASET_BATCH_TYPE)
        || isType(calledFunction, DATASET_SHUFFLE_TYPE)
        || isType(calledFunction, DATASET_MAP_TYPE)
        || isType(calledFunction, DATASET_RANGE_TYPE)
        || isType(calledFunction, DATASET_FROM_TENSOR_SLICES_TYPE)
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
