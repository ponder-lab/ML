package com.ibm.wala.cast.python.ml.types;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import java.util.Map;

/**
 * Types found in the TensorFlow library.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorFlowTypes extends PythonTypes {

  /**
   * Defined data types used in TensorFlow.
   *
   * @see <a href="https://www.tensorflow.org/api_docs/python/tf/dtypes#other-members">TensorFlow
   *     dtypes</a>.
   */
  public enum DType {
    FLOAT32(true, true, 32),
    FLOAT64(true, true, 64),
    INT32(true, false, 32),
    INT64(true, false, 64),
    STRING(false, false, 0);

    private boolean numeric;

    private boolean floatingPoint;

    private int precision;

    DType(boolean numeric, boolean floatingPoint, int precision) {
      this.numeric = numeric;
      this.floatingPoint = floatingPoint;
      this.precision = precision;
    }

    public boolean canConvertTo(DType other) {
      if (other == null) return false;

      if (!this.numeric || !other.numeric) return this == other;

      if (this.floatingPoint && !other.floatingPoint) return false;

      return this.precision <= other.precision;
    }
  }

  public static final TypeReference TENSORFLOW =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow"));

  public static final TypeReference DATASET =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset"));

  /**
   * Represents the TensorFlow data type.
   *
   * @see <a href="https://www.tensorflow.org/api_docs/python/tf/dtypes/DType">TensorFlow DType</a>.
   */
  public static final TypeReference D_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/dtypes/DType"));

  public static final TypeReference TENSOR_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/framework/ops/Tensor"));

  public static final TypeReference CONVERT_TO_TENSOR_TYPE =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/framework/ops/convert_to_tensor"));

  public static final TypeReference NDARRAY_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/framework/ops/ndarray"));

  public static final TypeReference CONSTANT_OP_CONSTANT =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/framework/constant_op/constant"));

  public static final TypeReference SPARSE_TENSOR_TYPE =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/framework/sparse_tensor/SparseTensor"));

  public static final TypeReference LINALG_OPS_EYE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/linalg_ops/eye"));

  public static final TypeReference ARRAY_OPS_ZEROS =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/array_ops/zeros"));

  public static final TypeReference RAGGED_MATH_OPS_RANGE =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/ops/ragged/ragged_math_ops/range"));

  public static final TypeReference RAGGED_FACTORY_OPS_CONSTANT =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/ops/ragged/ragged_factory_ops/constant"));

  public static final TypeReference VARIABLES_VARIABLE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/variables/Variable"));

  public static final TypeReference DATASET_SHUFFLE_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/shuffle"));

  public static final TypeReference DATASET_BATCH_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/batch"));

  public static final TypeReference DATASET_MAP_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/map"));

  public static final TypeReference DATASET_RANGE_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/range"));

  public static final TypeReference DATASET_FROM_TENSOR_SLICES_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/from_tensor_slices"));

  /** https://www.tensorflow.org/api_docs/python/tf/ones. */
  public static final MethodReference ONES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/ones")),
          AstMethodReference.fnSelector);

  private static final String ONES_SIGNATURE = "tf.ones()";

  /** https://www.tensorflow.org/api_docs/python/tf/constant. */
  public static final MethodReference CONSTANT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/constant")),
          AstMethodReference.fnSelector);

  private static final String CONSTANT_SIGNATURE = "tf.constant()";

  /** https://www.tensorflow.org/api_docs/python/tf/keras/Input. */
  public static final MethodReference INPUT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/keras/layers/Input")),
          AstMethodReference.fnSelector);

  private static final String INPUT_SIGNATURE = "tf.keras.Input()";

  /** https://www.tensorflow.org/api_docs/python/tf/range. */
  public static final MethodReference RANGE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/range")),
          AstMethodReference.fnSelector);

  private static final String RANGE_SIGNATURE = "tf.range()";

  /** https://www.tensorflow.org/api_docs/python/tf/random/uniform. */
  public static final MethodReference UNIFORM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/uniform")),
          AstMethodReference.fnSelector);

  private static final String UNIFORM_SIGNATURE = "tf.random.uniform()";

  /** https://www.tensorflow.org/api_docs/python/tf/random/normal. */
  public static final MethodReference NORMAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/normal")),
          AstMethodReference.fnSelector);

  private static final String NORMAL_SIGNATURE = "tf.random.normal()";

  /** https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal. */
  public static final MethodReference TRUNCATED_NORMAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/truncated_normal")),
          AstMethodReference.fnSelector);

  private static final String TRUNCATED_NORMAL_SIGNATURE = "tf.random.truncated_normal()";

  /** https://www.tensorflow.org/api_docs/python/tf/zeros. */
  public static final MethodReference ZEROS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/zeros")),
          AstMethodReference.fnSelector);

  private static final String ZEROS_SIGNATURE = "tf.zeros()";

  /** https://www.tensorflow.org/api_docs/python/tf/zeros_like. */
  public static final MethodReference ZEROS_LIKE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/zeros_like")),
          AstMethodReference.fnSelector);

  private static final String ZEROS_LIKE_SIGNATURE = "tf.zeros_like()";

  /** https://www.tensorflow.org/api_docs/python/tf/fill. */
  public static final MethodReference FILL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/fill")),
          AstMethodReference.fnSelector);

  private static final String FILL_SIGNATURE = "tf.fill()";

  /** https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor. */
  public static final MethodReference CONVERT_TO_TENSOR =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/convert_to_tensor")),
          AstMethodReference.fnSelector);

  private static final String CONVERT_TO_TENSOR_SIGNATURE = "tf.convert_to_tensor()";

  /** https://www.tensorflow.org/api_docs/python/tf/one_hot. */
  public static final MethodReference ONE_HOT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/one_hot")),
          AstMethodReference.fnSelector);

  private static final String ONE_HOT_SIGNATURE = "tf.one_hot()";

  /** https://www.tensorflow.org/api_docs/python/tf/eye. */
  public static final MethodReference EYE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/eye")),
          AstMethodReference.fnSelector);

  private static final String EYE_SIGNATURE = "tf.eye()";

  /** https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor. */
  public static final MethodReference SPARSE_TENSOR =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/SparseTensor")),
          AstMethodReference.fnSelector);

  private static final String SPARSE_TENSOR_SIGNATURE = "tf.sparse.SparseTensor";

  /** https://www.tensorflow.org/api_docs/python/tf/sparse/eye. */
  public static final MethodReference SPARSE_EYE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/sparse_eye")),
          AstMethodReference.fnSelector);

  private static final String SPARSE_EYE_SIGNATURE = "tf.sparse.eye()";

  /** https://www.tensorflow.org/api_docs/python/tf/sparse/add. */
  public static final MethodReference SPARSE_ADD =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/sparse_add")),
          AstMethodReference.fnSelector);

  private static final String SPARSE_ADD_SIGNATURE = "tf.sparse.add()";

  /** https://www.tensorflow.org/api_docs/python/tf/sparse/from_dense. */
  public static final MethodReference SPARSE_FROM_DENSE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/sparse_from_dense")),
          AstMethodReference.fnSelector);

  private static final String SPARSE_FROM_DENSE_SIGNATURE = "tf.sparse.from_dense()";

  /** https://www.tensorflow.org/api_docs/python/tf/gamma. */
  public static final MethodReference GAMMA =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/gamma")),
          AstMethodReference.fnSelector);

  private static final String GAMMA_SIGNATURE = "tf.random.gamma()";

  /** https://www.tensorflow.org/api_docs/python/tf/poisson. */
  public static final MethodReference POISSON =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/poisson")),
          AstMethodReference.fnSelector);

  private static final String POISSON_SIGNATURE = "tf.random.poisson()";

  /** https://www.tensorflow.org/api_docs/python/tf/Variable. */
  public static final MethodReference VARIABLE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/Variable")),
          AstMethodReference.fnSelector);

  private static final String VARIABLE_SIGNATURE = "tf.Variable()";

  /** https://www.tensorflow.org/api_docs/python/tf/ragged/constant. */
  public static final MethodReference RAGGED_CONSTANT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/ragged_constant")),
          AstMethodReference.fnSelector);

  private static final String RAGGED_CONSTANT_SIGNATURE = "tf.ragged.constant()";

  /** https://www.tensorflow.org/api_docs/python/tf/ragged/range. */
  public static final MethodReference RAGGED_RANGE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/ragged_range")),
          AstMethodReference.fnSelector);

  private static final String RAGGED_RANGE_SIGNATURE = "tf.ragged.range()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_value_rowids. */
  public static final MethodReference FROM_VALUE_ROWIDS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_value_rowids")),
          AstMethodReference.fnSelector);

  private static final String FROM_VALUE_ROWIDS_SIGNATURE = "tf.RaggedTensor.from_value_rowids()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_starts. */
  public static final MethodReference FROM_ROW_STARTS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_row_starts")),
          AstMethodReference.fnSelector);

  private static final String FROM_ROW_STARTS_SIGNATURE = "tf.RaggedTensor.from_row_starts()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_splits. */
  public static final MethodReference FROM_ROW_SPLITS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_row_splits")),
          AstMethodReference.fnSelector);

  private static final String FROM_ROW_SPLITS_SIGNATURE = "tf.RaggedTensor.from_row_splits()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_lengths. */
  public static final MethodReference FROM_ROW_LENGTHS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_row_lengths")),
          AstMethodReference.fnSelector);

  private static final String FROM_ROW_LENGTHS_SIGNATURE = "tf.RaggedTensor.from_row_lengths()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_row_limits. */
  public static final MethodReference FROM_ROW_LIMITS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_row_limits")),
          AstMethodReference.fnSelector);

  private static final String FROM_ROW_LIMITS_SIGNATURE = "tf.RaggedTensor.from_row_limits()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_row_lengths. */
  public static final MethodReference FROM_NESTED_ROW_LENGTHS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_nested_row_lengths")),
          AstMethodReference.fnSelector);

  private static final String FROM_NESTED_ROW_LENGTHS_SIGNATURE =
      "tf.RaggedTensor.from_nested_row_lengths()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_row_splits. */
  public static final MethodReference FROM_NESTED_ROW_SPLITS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_nested_row_splits")),
          AstMethodReference.fnSelector);

  private static final String FROM_NESTED_ROW_SPLITS_SIGNATURE =
      "tf.RaggedTensor.from_nested_row_splits()";

  /** https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#from_nested_value_rowids. */
  public static final MethodReference FROM_NESTED_VALUE_ROWIDS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/from_nested_value_rowids")),
          AstMethodReference.fnSelector);

  private static final String FROM_NESTED_VALUE_ROWIDS_SIGNATURE =
      "tf.RaggedTensor.from_nested_value_rowids()";

  public static final MethodReference MULTIPLY =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/multiply")),
          AstMethodReference.fnSelector);

  private static final String MULTIPLY_SIGNATURE = "tf.multiply()";

  public static final MethodReference ADD =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/add")),
          AstMethodReference.fnSelector);

  private static final String ADD_SIGNATURE = "tf.add()";

  public static final MethodReference SUBTRACT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/subtract")),
          AstMethodReference.fnSelector);

  private static final String SUBTRACT_SIGNATURE = "tf.subtract()";

  public static final MethodReference DIVIDE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/divide")),
          AstMethodReference.fnSelector);

  private static final String DIVIDE_SIGNATURE = "tf.divide()";

  public static final MethodReference MODEL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/keras/models/Model")),
          AstMethodReference.fnSelector);

  private static final String MODEL_SIGNATURE = "tf.keras.Model()";

  public static final MethodReference TENSOR =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/Tensor")),
          AstMethodReference.fnSelector);

  private static final String TENSOR_SIGNATURE = "tf.Tensor()";

  public static final MethodReference NDARRAY =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/ndarray")),
          AstMethodReference.fnSelector);

  private static final String NDARRAY_SIGNATURE = "tf.ndarray()";

  public static final MethodReference READ_DATA_SETS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/examples/tutorials/mnist/read_data_sets")),
          AstMethodReference.fnSelector);

  private static final String READ_DATA_SETS_SIGNATURE =
      "tf.contrib.learn.datasets.mnist.read_data_sets()";

  public static final MethodReference RESHAPE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/reshape")),
          AstMethodReference.fnSelector);

  private static final String RESHAPE_SIGNATURE = "tf.reshape()";

  public static final MethodReference DATASET_BATCH =
      MethodReference.findOrCreate(DATASET, AstMethodReference.fnSelector);

  public static final MethodReference DATASET_SHUFFLE =
      MethodReference.findOrCreate(DATASET, AstMethodReference.fnSelector);

  public static final MethodReference DATASET_MAP =
      MethodReference.findOrCreate(DATASET, AstMethodReference.fnSelector);

  public static final MethodReference DATASET_FROM_TENSOR_SLICES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/data/Dataset/from_tensor_slices")),
          AstMethodReference.fnSelector);

  /** A mapping from a {@link TypeReference} to its associated TensorFlow signature. */
  public static final Map<TypeReference, String> TYPE_REFERENCE_TO_SIGNATURE =
      Map.ofEntries(
          Map.entry(RESHAPE.getDeclaringClass(), RESHAPE_SIGNATURE),
          Map.entry(CONSTANT.getDeclaringClass(), CONSTANT_SIGNATURE),
          Map.entry(RANGE.getDeclaringClass(), RANGE_SIGNATURE),
          Map.entry(NORMAL.getDeclaringClass(), NORMAL_SIGNATURE),
          Map.entry(TRUNCATED_NORMAL.getDeclaringClass(), TRUNCATED_NORMAL_SIGNATURE),
          Map.entry(ZEROS_LIKE.getDeclaringClass(), ZEROS_LIKE_SIGNATURE),
          Map.entry(FILL.getDeclaringClass(), FILL_SIGNATURE),
          Map.entry(CONVERT_TO_TENSOR.getDeclaringClass(), CONVERT_TO_TENSOR_SIGNATURE),
          Map.entry(EYE.getDeclaringClass(), EYE_SIGNATURE),
          Map.entry(SPARSE_TENSOR.getDeclaringClass(), SPARSE_TENSOR_SIGNATURE),
          Map.entry(SPARSE_EYE.getDeclaringClass(), SPARSE_EYE_SIGNATURE),
          Map.entry(SPARSE_ADD.getDeclaringClass(), SPARSE_ADD_SIGNATURE),
          Map.entry(SPARSE_FROM_DENSE.getDeclaringClass(), SPARSE_FROM_DENSE_SIGNATURE),
          Map.entry(ONES.getDeclaringClass(), ONES_SIGNATURE),
          Map.entry(ZEROS.getDeclaringClass(), ZEROS_SIGNATURE),
          Map.entry(ONE_HOT.getDeclaringClass(), ONE_HOT_SIGNATURE),
          Map.entry(UNIFORM.getDeclaringClass(), UNIFORM_SIGNATURE),
          Map.entry(GAMMA.getDeclaringClass(), GAMMA_SIGNATURE),
          Map.entry(POISSON.getDeclaringClass(), POISSON_SIGNATURE),
          Map.entry(VARIABLE.getDeclaringClass(), VARIABLE_SIGNATURE),
          Map.entry(INPUT.getDeclaringClass(), INPUT_SIGNATURE),
          Map.entry(RAGGED_CONSTANT.getDeclaringClass(), RAGGED_CONSTANT_SIGNATURE),
          Map.entry(RAGGED_RANGE.getDeclaringClass(), RAGGED_RANGE_SIGNATURE),
          Map.entry(FROM_VALUE_ROWIDS.getDeclaringClass(), FROM_VALUE_ROWIDS_SIGNATURE),
          Map.entry(FROM_ROW_STARTS.getDeclaringClass(), FROM_ROW_STARTS_SIGNATURE),
          Map.entry(FROM_ROW_SPLITS.getDeclaringClass(), FROM_ROW_SPLITS_SIGNATURE),
          Map.entry(FROM_ROW_LENGTHS.getDeclaringClass(), FROM_ROW_LENGTHS_SIGNATURE),
          Map.entry(FROM_ROW_LIMITS.getDeclaringClass(), FROM_ROW_LIMITS_SIGNATURE),
          Map.entry(FROM_NESTED_ROW_LENGTHS.getDeclaringClass(), FROM_NESTED_ROW_LENGTHS_SIGNATURE),
          Map.entry(FROM_NESTED_ROW_SPLITS.getDeclaringClass(), FROM_NESTED_ROW_SPLITS_SIGNATURE),
          Map.entry(
              FROM_NESTED_VALUE_ROWIDS.getDeclaringClass(), FROM_NESTED_VALUE_ROWIDS_SIGNATURE),
          Map.entry(MULTIPLY.getDeclaringClass(), MULTIPLY_SIGNATURE),
          Map.entry(ADD.getDeclaringClass(), ADD_SIGNATURE),
          Map.entry(SUBTRACT.getDeclaringClass(), SUBTRACT_SIGNATURE),
          Map.entry(DIVIDE.getDeclaringClass(), DIVIDE_SIGNATURE),
          Map.entry(MODEL.getDeclaringClass(), MODEL_SIGNATURE),
          Map.entry(TENSOR.getDeclaringClass(), TENSOR_SIGNATURE),
          Map.entry(NDARRAY.getDeclaringClass(), NDARRAY_SIGNATURE),
          Map.entry(READ_DATA_SETS.getDeclaringClass(), READ_DATA_SETS_SIGNATURE));

  /**
   * Represents the TensorFlow float32 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#float32">TensorFlow
   *     float32 DType</a>.
   */
  public static final FieldReference FLOAT_32 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(FLOAT32.name().toLowerCase()), D_TYPE);

  /**
   * Represents the TensorFlow float64 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#float64">TensorFlow
   *     float64 DType</a>.
   */
  public static final FieldReference FLOAT_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(FLOAT64.name().toLowerCase()), D_TYPE);

  /**
   * Represents the TensorFlow int32 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#int32">TensorFlow
   *     int32 DType</a>.
   */
  public static final FieldReference INT_32 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(INT32.name().toLowerCase()), D_TYPE);

  /**
   * Represents the TensorFlow int64 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#int64">TensorFlow
   *     int64 DType</a>.
   */
  public static final FieldReference INT_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(INT64.name().toLowerCase()), D_TYPE);

  /**
   * Represents the TensorFlow string data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#string">TensorFlow
   *     string DType</a>.
   */
  public static final FieldReference STRING =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(DType.STRING.name().toLowerCase()), D_TYPE);

  /** A mapping from a field reference to its associated {@link DType}, if any. */
  public static final Map<FieldReference, DType> FIELD_REFERENCE_TO_DTYPE =
      Map.of(
          FLOAT_32, FLOAT32, FLOAT_64, FLOAT64, INT_32, INT32, INT_64, INT64, STRING, DType.STRING);

  private TensorFlowTypes() {}
}
