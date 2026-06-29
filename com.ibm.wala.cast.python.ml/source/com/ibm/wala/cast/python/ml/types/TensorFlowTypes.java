package com.ibm.wala.cast.python.ml.types;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX128;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.COMPLEX64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.UINT8;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import java.util.Locale;
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
    FLOAT32(true, true, false, 32),
    FLOAT64(true, true, false, 64),
    INT32(true, false, false, 32),
    INT64(true, false, false, 64),
    UINT8(true, false, false, 8),
    // `complex64` is 2x float32 and `complex128` is 2x float64. Modeled with a dedicated `complex`
    // flag (rather than reusing `floatingPoint`) so that a real value can widen into a complex one
    // but a complex value never narrows back to a real, and a complex only widens to a wider
    // complex. See wala/ML#637.
    COMPLEX64(true, false, true, 64),
    COMPLEX128(true, false, true, 128),
    BOOL(false, false, false, 1),
    STRING(false, false, false, 0),
    // numpy `object` dtype: an ndarray of variable-length Python sequences (e.g. `reuters`/`imdb`
    // `x_train`). Non-numeric, so `canConvertTo` only matches itself — it does not auto-convert to
    // numeric dtypes, matching numpy's strict semantics. See wala/ML#488.
    OBJECT(false, false, false, 0),
    UNKNOWN(false, false, false, 0);

    private boolean numeric;

    private boolean floatingPoint;

    private boolean complex;

    private int precision;

    DType(boolean numeric, boolean floatingPoint, boolean complex, int precision) {
      this.numeric = numeric;
      this.floatingPoint = floatingPoint;
      this.complex = complex;
      this.precision = precision;
    }

    public boolean canConvertTo(DType other) {
      if (other == null) return false;

      if (!this.numeric || !other.numeric) return this == other;

      // Complex is a distinct kind: a complex value never narrows to a real (int/float) one, a real
      // value widens into any complex, and a complex only widens to a complex of at least its
      // precision.
      if (this.complex || other.complex) {
        if (this.complex && !other.complex) return false;
        if (!this.complex && other.complex) return true;
        return this.precision <= other.precision;
      }

      if (this.floatingPoint && !other.floatingPoint) return false;

      return this.precision <= other.precision;
    }
  }

  public static final String TENSORFLOW = "tensorflow";

  public static final TypeReference TENSORFLOW_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow"));

  public static final TypeReference NUMPY_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Lnumpy"));

  public static final String DATA_PACKAGE_PREFIX = "Ltensorflow/data/";

  public static final TypeReference DATASET =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate(DATA_PACKAGE_PREFIX + "Dataset"));

  public static final String DATASET_SIGNATURE = "tf.data.Dataset()";

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

  public static final TypeReference TENSOR_FUNCTIONS_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/functions/Tensor"));

  public static final TypeReference CONVERT_TO_TENSOR_TYPE =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/framework/ops/convert_to_tensor"));

  public static final TypeReference NDARRAY_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/framework/ops/ndarray"));

  public static final TypeReference OPERATION =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/framework/ops/Operation"));

  public static final TypeReference FEATURE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/objects/feature"));

  public static final TypeReference CONSTANT_OP_CONSTANT =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/framework/constant_op/constant"));

  public static final FieldReference CONSTANT_VALUE =
      FieldReference.findOrCreate(CONSTANT_OP_CONSTANT, findOrCreateAsciiAtom("value"), Root);

  public static final FieldReference CONSTANT_DTYPE =
      FieldReference.findOrCreate(CONSTANT_OP_CONSTANT, findOrCreateAsciiAtom("dtype"), Root);

  public static final TypeReference SPARSE_TENSOR_TYPE =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/framework/sparse_tensor/SparseTensor"));

  public static final TypeReference SPARSE_TENSOR_FUNCTIONS_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/functions/SparseTensor"));

  public static final TypeReference LINALG_OPS_EYE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/linalg_ops/eye"));

  public static final TypeReference ARRAY_OPS_ZEROS =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/array_ops/zeros"));

  public static final TypeReference ARRAY_OPS_RESHAPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/array_ops/reshape"));

  public static final TypeName TF_RESHAPE = TypeName.findOrCreate("Ltensorflow/functions/reshape");

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

  /**
   * Modeled type for {@code tf.newaxis} (see {@code tensorflow.xml}). At Python runtime {@code
   * tf.newaxis is None}, but WALA represents attribute access as a synthetic allocation, so we give
   * it a distinct sentinel class that {@link
   * com.ibm.wala.cast.python.ml.client.NdarraySubscriptOperation#classifyField} can match.
   */
  public static final TypeReference NEWAXIS =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/newaxis"));

  public static final TypeReference DATASET_SHUFFLE_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/shuffle"));

  public static final String DATASET_SHUFFLE_SIGNATURE = "tf.data.Dataset.shuffle()";

  public static final TypeReference DATASET_BATCH_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/batch"));

  public static final String DATASET_BATCH_SIGNATURE = "tf.data.Dataset.batch()";

  public static final TypeReference DATASET_PADDED_BATCH_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/padded_batch"));

  public static final String DATASET_PADDED_BATCH_SIGNATURE = "tf.data.Dataset.padded_batch()";

  public static final TypeReference DATASET_MAP_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/map"));

  public static final String DATASET_MAP_SIGNATURE = "tf.data.Dataset.map()";

  public static final TypeReference DATASET_REPEAT_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/repeat"));

  public static final String DATASET_REPEAT_SIGNATURE = "tf.data.Dataset.repeat()";

  public static final TypeReference DATASET_PREFETCH_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/prefetch"));

  public static final String DATASET_PREFETCH_SIGNATURE = "tf.data.Dataset.prefetch()";

  public static final TypeReference DATASET_TAKE_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/take"));

  public static final String DATASET_TAKE_SIGNATURE = "tf.data.Dataset.take()";

  public static final TypeReference DATASET_WITH_OPTIONS_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/with_options"));

  public static final String DATASET_WITH_OPTIONS_SIGNATURE = "tf.data.Dataset.with_options()";

  public static final TypeReference DATASET_CONCATENATE_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/concatenate"));

  public static final String DATASET_CONCATENATE_SIGNATURE = "tf.data.Dataset.concatenate()";

  public static final TypeReference DATASET_CHOOSE_FROM_DATASETS_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/choose_from_datasets"));

  public static final TypeReference DATASET_ENUMERATE_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/enumerate"));

  public static final String DATASET_ENUMERATE_SIGNATURE = "tf.data.Dataset.enumerate()";

  public static final TypeReference DATASET_REDUCE_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/reduce"));

  public static final String DATASET_REDUCE_SIGNATURE = "tf.data.Dataset.reduce()";

  public static final TypeReference DATASET_FILTER_TYPE =
      TypeReference.findOrCreate(pythonLoader, TypeName.findOrCreate("Ltensorflow/data/filter"));

  public static final String DATASET_FILTER_SIGNATURE = "tf.data.Dataset.filter()";

  public static final TypeReference IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/keras/preprocessing/image/flow_from_directory"));

  public static final String FLOW_FROM_DIRECTORY_SIGNATURE =
      "tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory()";

  /**
   * The type of `tf.data.Dataset.from_generator`.
   *
   * @see <a
   *     href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator">tf.data.Dataset.from_generator</a>
   */
  public static final TypeReference DATASET_FROM_GENERATOR_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/from_generator"));

  public static final String DATASET_FROM_GENERATOR_SIGNATURE = "tf.data.Dataset.from_generator()";

  public static final TypeReference DATASET_ZIP_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/zip"));

  public static final String DATASET_ZIP_SIGNATURE = "tf.data.Dataset.zip()";

  public static final TypeReference DATASET_RANDOM_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/random"));

  public static final String DATASET_RANDOM_SIGNATURE = "tf.data.Dataset.random()";

  public static final TypeReference DATASET_SAMPLE_FROM_DATASETS_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/sample_from_datasets"));

  public static final String DATASET_SAMPLE_FROM_DATASETS_SIGNATURE =
      "tf.data.Dataset.sample_from_datasets()";

  public static final String DATASET_CHOOSE_FROM_DATASETS_SIGNATURE =
      "tf.data.Dataset.choose_from_datasets()";

  /**
   * The type of `tf.TensorSpec`.
   *
   * @see <a href="https://www.tensorflow.org/api_docs/python/tf/TensorSpec">tf.TensorSpec</a>
   */
  public static final TypeReference TENSOR_SPEC =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/framework/TensorSpec"));

  public static final String TENSOR_SPEC_SIGNATURE = "tf.TensorSpec()";

  /**
   * The type of `tf.RaggedTensorSpec`.
   *
   * @see <a
   *     href="https://www.tensorflow.org/api_docs/python/tf/RaggedTensorSpec">tf.RaggedTensorSpec</a>
   */
  public static final TypeReference RAGGED_TENSOR_SPEC =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/framework/RaggedTensorSpec"));

  public static final String RAGGED_TENSOR_SPEC_SIGNATURE = "tf.RaggedTensorSpec()";

  public static final FieldReference SPEC_SHAPE =
      FieldReference.findOrCreate(TENSOR_SPEC, findOrCreateAsciiAtom("shape"), Root);

  public static final FieldReference SPEC_DTYPE =
      FieldReference.findOrCreate(TENSOR_SPEC, findOrCreateAsciiAtom("dtype"), Root);

  public static final FieldReference RAGGED_SPEC_SHAPE =
      FieldReference.findOrCreate(RAGGED_TENSOR_SPEC, findOrCreateAsciiAtom("shape"), Root);

  public static final FieldReference RAGGED_SPEC_DTYPE =
      FieldReference.findOrCreate(RAGGED_TENSOR_SPEC, findOrCreateAsciiAtom("dtype"), Root);

  public static final TypeReference DATASET_RANGE_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/range"));

  public static final String DATASET_RANGE_SIGNATURE = "tf.data.Dataset.range()";

  public static final TypeReference TEXT_LINE_DATASET_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/TextLineDataset"));

  public static final String TEXT_LINE_DATASET_SIGNATURE = "tf.data.TextLineDataset()";

  public static final TypeReference DATASET_FROM_TENSOR_SLICES_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/from_tensor_slices"));

  public static final String DATASET_FROM_TENSOR_SLICES_SIGNATURE =
      "tf.data.Dataset.from_tensor_slices()";

  public static final TypeReference DATASET_FROM_TENSORS_TYPE =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/data/Dataset/from_tensors"));

  public static final String DATASET_FROM_TENSORS_SIGNATURE = "tf.data.Dataset.from_tensors()";

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

  public static final TypeReference UNIFORM_OP =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/random_ops/uniform"));

  private static final String UNIFORM_SIGNATURE = "tf.random.uniform()";

  /** https://www.tensorflow.org/api_docs/python/tf/random/normal. */
  public static final MethodReference NORMAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/normal")),
          AstMethodReference.fnSelector);

  public static final TypeReference NORMAL_OP =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/random_ops/normal"));

  /**
   * https://www.tensorflow.org/api_docs/python/tf/keras/initializers/RandomNormal#__call__.
   *
   * <p>The {@code __call__} method on a {@code tf.initializers.RandomNormal} instance: {@code
   * instance(shape, dtype=None)} returns a tensor of the requested shape drawn from the normal
   * distribution.
   */
  public static final MethodReference RANDOM_NORMAL_INIT_CALL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName(
                  "Ltensorflow/initializers/RandomNormal/" + CALLABLE_METHOD_NAME)),
          AstMethodReference.fnSelector);

  private static final String NORMAL_SIGNATURE = "tf.random.normal()";

  /** Method name used in {@code tensorflow.xml} for {@link #TRUNCATED_NORMAL}. */
  public static final String TRUNCATED_NORMAL_METHOD_NAME = "truncated_normal";

  /** https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal. */
  public static final MethodReference TRUNCATED_NORMAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/truncated_normal")),
          AstMethodReference.fnSelector);

  public static final TypeReference TRUNCATED_NORMAL_OP =
      TypeReference.findOrCreate(
          pythonLoader,
          TypeName.findOrCreate("Ltensorflow/python/ops/random_ops/truncated_normal"));

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

  /** https://www.tensorflow.org/api_docs/python/tf/linspace. */
  public static final MethodReference LINSPACE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/linspace")),
          AstMethodReference.fnSelector);

  private static final String LINSPACE_SIGNATURE = "tf.linspace()";

  /** https://www.tensorflow.org/api_docs/python/tf/broadcast_to. */
  public static final MethodReference BROADCAST_TO =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/broadcast_to")),
          AstMethodReference.fnSelector);

  private static final String BROADCAST_TO_SIGNATURE = "tf.broadcast_to()";

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

  public static final TypeReference GAMMA_OP =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/random_ops/gamma"));

  private static final String GAMMA_SIGNATURE = "tf.random.gamma()";

  /** https://www.tensorflow.org/api_docs/python/tf/poisson. */
  public static final MethodReference POISSON =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/poisson")),
          AstMethodReference.fnSelector);

  public static final TypeReference POISSON_OP =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/python/ops/random_ops/poisson"));

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

  public static final MethodReference REDUCE_MEAN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reduce_mean")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_MEAN_SIGNATURE = "tf.reduce_mean()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reduce_max. */
  public static final MethodReference REDUCE_MAX =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reduce_max")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_MAX_SIGNATURE = "tf.reduce_max()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reduce_prod. */
  public static final MethodReference REDUCE_PROD =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reduce_prod")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_PROD_SIGNATURE = "tf.reduce_prod()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reduce_logsumexp. */
  public static final MethodReference REDUCE_LOGSUMEXP =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/math/reduce_logsumexp")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_LOGSUMEXP_SIGNATURE = "tf.reduce_logsumexp()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_sum. */
  public static final MethodReference UNSORTED_SEGMENT_SUM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/math/unsorted_segment_sum")),
          AstMethodReference.fnSelector);

  private static final String UNSORTED_SEGMENT_SUM_SIGNATURE = "tf.math.unsorted_segment_sum()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_max. */
  public static final MethodReference UNSORTED_SEGMENT_MAX =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/math/unsorted_segment_max")),
          AstMethodReference.fnSelector);

  private static final String UNSORTED_SEGMENT_MAX_SIGNATURE = "tf.math.unsorted_segment_max()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/unsorted_segment_mean. */
  public static final MethodReference UNSORTED_SEGMENT_MEAN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/math/unsorted_segment_mean")),
          AstMethodReference.fnSelector);

  private static final String UNSORTED_SEGMENT_MEAN_SIGNATURE = "tf.math.unsorted_segment_mean()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reduce_all. */
  public static final MethodReference REDUCE_ALL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reduce_all")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_ALL_SIGNATURE = "tf.reduce_all()";

  public static final MethodReference MODEL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/keras/models/Model")),
          AstMethodReference.fnSelector);

  public static final TypeReference MODEL_ATTRIBUTE =
      TypeReference.findOrCreate(
          PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/keras/Model/attribute"));

  private static final String MODEL_SIGNATURE = "tf.keras.Model()";

  public static final MethodReference MODEL_CALL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/keras/models/Model/" + CALLABLE_METHOD_NAME)),
          AstMethodReference.fnSelector);

  private static final String MODEL_CALL_SIGNATURE =
      "tf.keras.models.Model." + CALLABLE_METHOD_NAME + "()";

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

  public static final TypeReference MNIST_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/mnist/x_train"));

  public static final String MNIST_X_TRAIN_SIGNATURE = "tf.keras.datasets.mnist.load_data/x_train";

  public static final TypeReference MNIST_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/mnist/y_train"));

  public static final String MNIST_Y_TRAIN_SIGNATURE = "tf.keras.datasets.mnist.load_data/y_train";

  public static final TypeReference MNIST_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/mnist/x_test"));

  public static final String MNIST_X_TEST_SIGNATURE = "tf.keras.datasets.mnist.load_data/x_test";

  public static final TypeReference MNIST_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/mnist/y_test"));

  public static final String MNIST_Y_TEST_SIGNATURE = "tf.keras.datasets.mnist.load_data/y_test";

  public static final TypeReference CIFAR10_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar10/x_train"));

  public static final String CIFAR10_X_TRAIN_SIGNATURE =
      "tf.keras.datasets.cifar10.load_data/x_train";

  public static final TypeReference CIFAR10_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar10/y_train"));

  public static final String CIFAR10_Y_TRAIN_SIGNATURE =
      "tf.keras.datasets.cifar10.load_data/y_train";

  public static final TypeReference CIFAR10_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar10/x_test"));

  public static final String CIFAR10_X_TEST_SIGNATURE =
      "tf.keras.datasets.cifar10.load_data/x_test";

  public static final TypeReference CIFAR10_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar10/y_test"));

  public static final String CIFAR10_Y_TEST_SIGNATURE =
      "tf.keras.datasets.cifar10.load_data/y_test";

  public static final TypeReference IMDB_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/imdb/x_train"));

  public static final String IMDB_X_TRAIN_SIGNATURE = "tf.keras.datasets.imdb.load_data/x_train";

  public static final TypeReference IMDB_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/imdb/y_train"));

  public static final String IMDB_Y_TRAIN_SIGNATURE = "tf.keras.datasets.imdb.load_data/y_train";

  public static final TypeReference IMDB_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/imdb/x_test"));

  public static final String IMDB_X_TEST_SIGNATURE = "tf.keras.datasets.imdb.load_data/x_test";

  public static final TypeReference IMDB_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/imdb/y_test"));

  public static final String IMDB_Y_TEST_SIGNATURE = "tf.keras.datasets.imdb.load_data/y_test";
  public static final TypeReference FASHION_MNIST_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/fashion_mnist/x_train"));

  public static final String FASHION_MNIST_X_TRAIN_SIGNATURE =
      "tf.keras.datasets.fashion_mnist.load_data/x_train";

  public static final TypeReference FASHION_MNIST_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/fashion_mnist/y_train"));

  public static final String FASHION_MNIST_Y_TRAIN_SIGNATURE =
      "tf.keras.datasets.fashion_mnist.load_data/y_train";

  public static final TypeReference FASHION_MNIST_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/fashion_mnist/x_test"));

  public static final String FASHION_MNIST_X_TEST_SIGNATURE =
      "tf.keras.datasets.fashion_mnist.load_data/x_test";

  public static final TypeReference FASHION_MNIST_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/fashion_mnist/y_test"));

  public static final String FASHION_MNIST_Y_TEST_SIGNATURE =
      "tf.keras.datasets.fashion_mnist.load_data/y_test";

  public static final TypeReference CIFAR100_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar100/x_train"));

  public static final String CIFAR100_X_TRAIN_SIGNATURE =
      "tf.keras.datasets.cifar100.load_data/x_train";

  public static final TypeReference CIFAR100_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar100/y_train"));

  public static final String CIFAR100_Y_TRAIN_SIGNATURE =
      "tf.keras.datasets.cifar100.load_data/y_train";

  public static final TypeReference CIFAR100_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar100/x_test"));

  public static final String CIFAR100_X_TEST_SIGNATURE =
      "tf.keras.datasets.cifar100.load_data/x_test";

  public static final TypeReference CIFAR100_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/cifar100/y_test"));

  public static final String CIFAR100_Y_TEST_SIGNATURE =
      "tf.keras.datasets.cifar100.load_data/y_test";

  public static final TypeReference REUTERS_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/reuters/x_train"));

  public static final String REUTERS_X_TRAIN_SIGNATURE =
      "tf.keras.datasets.reuters.load_data/x_train";

  public static final TypeReference REUTERS_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/reuters/y_train"));

  public static final String REUTERS_Y_TRAIN_SIGNATURE =
      "tf.keras.datasets.reuters.load_data/y_train";

  public static final TypeReference REUTERS_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/reuters/x_test"));

  public static final String REUTERS_X_TEST_SIGNATURE =
      "tf.keras.datasets.reuters.load_data/x_test";

  public static final TypeReference REUTERS_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/reuters/y_test"));

  public static final String REUTERS_Y_TEST_SIGNATURE =
      "tf.keras.datasets.reuters.load_data/y_test";

  public static final TypeReference BOSTON_HOUSING_X_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/boston_housing/x_train"));

  public static final String BOSTON_HOUSING_X_TRAIN_SIGNATURE =
      "tf.keras.datasets.boston_housing.load_data/x_train";

  public static final TypeReference BOSTON_HOUSING_Y_TRAIN =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/boston_housing/y_train"));

  public static final String BOSTON_HOUSING_Y_TRAIN_SIGNATURE =
      "tf.keras.datasets.boston_housing.load_data/y_train";

  public static final TypeReference BOSTON_HOUSING_X_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/boston_housing/x_test"));

  public static final String BOSTON_HOUSING_X_TEST_SIGNATURE =
      "tf.keras.datasets.boston_housing.load_data/x_test";

  public static final TypeReference BOSTON_HOUSING_Y_TEST =
      TypeReference.findOrCreate(
          pythonLoader, TypeName.findOrCreate("Ltensorflow/keras/datasets/boston_housing/y_test"));

  public static final String BOSTON_HOUSING_Y_TEST_SIGNATURE =
      "tf.keras.datasets.boston_housing.load_data/y_test";

  /** https://www.tensorflow.org/api_docs/python/tf/placeholder. */
  public static final MethodReference PLACEHOLDER =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/placeholder")),
          AstMethodReference.fnSelector);

  private static final String PLACEHOLDER_SIGNATURE = "tf.placeholder()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/argmax. */
  public static final MethodReference ARGMAX =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/argmax")),
          AstMethodReference.fnSelector);

  private static final String ARGMAX_SIGNATURE = "tf.argmax()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/argmin. */
  public static final MethodReference ARGMIN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/argmin")),
          AstMethodReference.fnSelector);

  private static final String ARGMIN_SIGNATURE = "tf.argmin()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/tensordot. */
  public static final MethodReference TENSORDOT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/tensordot")),
          AstMethodReference.fnSelector);

  private static final String TENSORDOT_SIGNATURE = "tf.linalg.tensordot()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/trace. */
  public static final MethodReference TRACE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/trace")),
          AstMethodReference.fnSelector);

  private static final String TRACE_SIGNATURE = "tf.linalg.trace()";

  /** https://www.tensorflow.org/api_docs/python/tf/transpose. */
  public static final MethodReference TRANSPOSE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/transpose")),
          AstMethodReference.fnSelector);

  private static final String TRANSPOSE_SIGNATURE = "tf.transpose()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/diag. */
  public static final MethodReference DIAG =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/diag")),
          AstMethodReference.fnSelector);

  private static final String DIAG_SIGNATURE = "tf.linalg.diag()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/diag_part. */
  public static final MethodReference DIAG_PART =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/diag_part")),
          AstMethodReference.fnSelector);

  private static final String DIAG_PART_SIGNATURE = "tf.linalg.diag_part()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/matrix_transpose. */
  public static final MethodReference MATRIX_TRANSPOSE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/math/matrix_transpose")),
          AstMethodReference.fnSelector);

  private static final String MATRIX_TRANSPOSE_SIGNATURE = "tf.linalg.matrix_transpose()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/adjoint. */
  public static final MethodReference ADJOINT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/adjoint")),
          AstMethodReference.fnSelector);

  private static final String ADJOINT_SIGNATURE = "tf.linalg.adjoint()";

  /** https://www.tensorflow.org/api_docs/python/tf/tile. */
  public static final MethodReference TILE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/tile")),
          AstMethodReference.fnSelector);

  private static final String TILE_SIGNATURE = "tf.tile()";

  /** https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update. */
  public static final MethodReference TENSOR_SCATTER_ND_UPDATE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/tensor_scatter_nd_update")),
          AstMethodReference.fnSelector);

  private static final String TENSOR_SCATTER_ND_UPDATE_SIGNATURE = "tf.tensor_scatter_nd_update()";

  /** https://www.tensorflow.org/api_docs/python/tf/sequence_mask. */
  public static final MethodReference SEQUENCE_MASK =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/sequence_mask")),
          AstMethodReference.fnSelector);

  private static final String SEQUENCE_MASK_SIGNATURE = "tf.sequence_mask()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup. */
  public static final MethodReference EMBEDDING_LOOKUP =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/embedding_lookup")),
          AstMethodReference.fnSelector);

  private static final String EMBEDDING_LOOKUP_SIGNATURE = "tf.nn.embedding_lookup()";

  /** https://www.tensorflow.org/api_docs/python/tf/gather_nd. */
  public static final MethodReference GATHER_ND =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/gather_nd")),
          AstMethodReference.fnSelector);

  private static final String GATHER_ND_SIGNATURE = "tf.gather_nd()";

  /** https://www.tensorflow.org/api_docs/python/tf/boolean_mask. */
  public static final MethodReference BOOLEAN_MASK =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/boolean_mask")),
          AstMethodReference.fnSelector);

  private static final String BOOLEAN_MASK_SIGNATURE = "tf.boolean_mask()";

  /** https://www.tensorflow.org/api_docs/python/tf/slice. */
  public static final MethodReference SLICE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/slice")),
          AstMethodReference.fnSelector);

  private static final String SLICE_SIGNATURE = "tf.slice()";

  /** https://www.tensorflow.org/api_docs/python/tf/squeeze. */
  public static final MethodReference SQUEEZE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/squeeze")),
          AstMethodReference.fnSelector);

  private static final String SQUEEZE_SIGNATURE = "tf.squeeze()";

  /** https://www.tensorflow.org/api_docs/python/tf/image/extract_patches. */
  public static final MethodReference EXTRACT_PATCHES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/image/extract_patches")),
          AstMethodReference.fnSelector);

  private static final String EXTRACT_PATCHES_SIGNATURE = "tf.image.extract_patches()";

  // Tier-A math ops (continued, wala/ML#422). Most are shape and dtype passthrough on their primary
  // tensor argument (named `x` for most unary ops; `features` for `softplus` / `softsign`). The
  // binary ops `atan2` / `maximum` / `minimum` are the exception: they're routed through
  // `ElementWiseOperation`, which derives dtype from the `x` parameter at position 0 and computes
  // broadcast shape from both operands &mdash; not passthrough.

  /** https://www.tensorflow.org/api_docs/python/tf/math/tan. */
  public static final MethodReference TAN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/tan")),
          AstMethodReference.fnSelector);

  private static final String TAN_SIGNATURE = "tf.math.tan()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/asin. */
  public static final MethodReference ASIN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/asin")),
          AstMethodReference.fnSelector);

  private static final String ASIN_SIGNATURE = "tf.math.asin()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/atan. */
  public static final MethodReference ATAN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/atan")),
          AstMethodReference.fnSelector);

  private static final String ATAN_SIGNATURE = "tf.math.atan()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/sinh. */
  public static final MethodReference SINH =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/sinh")),
          AstMethodReference.fnSelector);

  private static final String SINH_SIGNATURE = "tf.math.sinh()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/cosh. */
  public static final MethodReference COSH =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/cosh")),
          AstMethodReference.fnSelector);

  private static final String COSH_SIGNATURE = "tf.math.cosh()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/asinh. */
  public static final MethodReference ASINH =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/asinh")),
          AstMethodReference.fnSelector);

  private static final String ASINH_SIGNATURE = "tf.math.asinh()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/acosh. */
  public static final MethodReference ACOSH =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/acosh")),
          AstMethodReference.fnSelector);

  private static final String ACOSH_SIGNATURE = "tf.math.acosh()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/atanh. */
  public static final MethodReference ATANH =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/atanh")),
          AstMethodReference.fnSelector);

  private static final String ATANH_SIGNATURE = "tf.math.atanh()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/log1p. */
  public static final MethodReference LOG1P =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/log1p")),
          AstMethodReference.fnSelector);

  private static final String LOG1P_SIGNATURE = "tf.math.log1p()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/expm1. */
  public static final MethodReference EXPM1 =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/expm1")),
          AstMethodReference.fnSelector);

  private static final String EXPM1_SIGNATURE = "tf.math.expm1()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/round. */
  public static final MethodReference ROUND =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/round")),
          AstMethodReference.fnSelector);

  private static final String ROUND_SIGNATURE = "tf.math.round()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reciprocal. */
  public static final MethodReference RECIPROCAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reciprocal")),
          AstMethodReference.fnSelector);

  private static final String RECIPROCAL_SIGNATURE = "tf.math.reciprocal()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/softplus. */
  public static final MethodReference SOFTPLUS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/softplus")),
          AstMethodReference.fnSelector);

  private static final String SOFTPLUS_SIGNATURE = "tf.math.softplus()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/softsign. */
  public static final MethodReference SOFTSIGN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/softsign")),
          AstMethodReference.fnSelector);

  private static final String SOFTSIGN_SIGNATURE = "tf.math.softsign()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/square. */
  public static final MethodReference SQUARE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/square")),
          AstMethodReference.fnSelector);

  private static final String SQUARE_SIGNATURE = "tf.math.square()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/erf. */
  public static final MethodReference ERF =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/erf")),
          AstMethodReference.fnSelector);

  private static final String ERF_SIGNATURE = "tf.math.erf()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/erfc. */
  public static final MethodReference ERFC =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/erfc")),
          AstMethodReference.fnSelector);

  private static final String ERFC_SIGNATURE = "tf.math.erfc()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/atan2. */
  public static final MethodReference ATAN2 =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/atan2")),
          AstMethodReference.fnSelector);

  private static final String ATAN2_SIGNATURE = "tf.math.atan2()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/maximum. */
  public static final MethodReference MAXIMUM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/maximum")),
          AstMethodReference.fnSelector);

  private static final String MAXIMUM_SIGNATURE = "tf.math.maximum()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/minimum. */
  public static final MethodReference MINIMUM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/minimum")),
          AstMethodReference.fnSelector);

  private static final String MINIMUM_SIGNATURE = "tf.math.minimum()";

  /** https://www.tensorflow.org/api_docs/python/tf/linalg/einsum. */
  public static final MethodReference EINSUM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/einsum")),
          AstMethodReference.fnSelector);

  private static final String EINSUM_SIGNATURE = "tf.einsum()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/relu. */
  public static final MethodReference RELU =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/relu")),
          AstMethodReference.fnSelector);

  private static final String RELU_SIGNATURE = "tf.nn.relu()";

  /** https://www.tensorflow.org/api_docs/python/tf/expand_dims. */
  public static final MethodReference EXPAND_DIMS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/expand_dims")),
          AstMethodReference.fnSelector);

  private static final String EXPAND_DIMS_SIGNATURE = "tf.expand_dims()";

  /** https://www.tensorflow.org/api_docs/python/tf/clip_by_value. */
  public static final MethodReference CLIP_BY_VALUE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/clip_by_value")),
          AstMethodReference.fnSelector);

  private static final String CLIP_BY_VALUE_SIGNATURE = "tf.clip_by_value()";

  /** https://www.tensorflow.org/api_docs/python/tf/strings/as_string. */
  public static final MethodReference AS_STRING =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/strings/as_string")),
          AstMethodReference.fnSelector);

  private static final String AS_STRING_SIGNATURE = "tf.strings.as_string()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/top_k. */
  public static final MethodReference TOP_K =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/top_k")),
          AstMethodReference.fnSelector);

  private static final String TOP_K_SIGNATURE = "tf.math.top_k()";

  /** https://www.tensorflow.org/api_docs/python/tf/meshgrid. */
  public static final MethodReference MESHGRID =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/meshgrid")),
          AstMethodReference.fnSelector);

  private static final String MESHGRID_SIGNATURE = "tf.meshgrid()";

  /** https://www.tensorflow.org/api_docs/python/tf/where. */
  public static final MethodReference WHERE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/where")),
          AstMethodReference.fnSelector);

  private static final String WHERE_SIGNATURE = "tf.where()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu. */
  public static final MethodReference LEAKY_RELU =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/leaky_relu")),
          AstMethodReference.fnSelector);

  private static final String LEAKY_RELU_SIGNATURE = "tf.nn.leaky_relu()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reduce_min. */
  public static final MethodReference REDUCE_MIN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reduce_min")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_MIN_SIGNATURE = "tf.reduce_min()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/pow. */
  public static final MethodReference POW =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/pow")),
          AstMethodReference.fnSelector);

  private static final String POW_SIGNATURE = "tf.math.pow()";

  /** https://www.tensorflow.org/api_docs/python/tf/concat. */
  public static final MethodReference CONCAT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/concat")),
          AstMethodReference.fnSelector);

  private static final String CONCAT_SIGNATURE = "tf.concat()";

  /** https://www.tensorflow.org/api_docs/python/tf/stack. */
  public static final MethodReference STACK =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/stack")),
          AstMethodReference.fnSelector);

  private static final String STACK_SIGNATURE = "tf.stack()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/sqrt. */
  public static final MethodReference SQRT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/sqrt")),
          AstMethodReference.fnSelector);

  private static final String SQRT_SIGNATURE = "tf.math.sqrt()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/negative. */
  public static final MethodReference NEGATIVE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/negative")),
          AstMethodReference.fnSelector);

  private static final String NEGATIVE_SIGNATURE = "tf.math.negative()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/sin. */
  public static final MethodReference SIN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/sin")),
          AstMethodReference.fnSelector);

  private static final String SIN_SIGNATURE = "tf.math.sin()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/cos. */
  public static final MethodReference COS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/cos")),
          AstMethodReference.fnSelector);

  private static final String COS_SIGNATURE = "tf.math.cos()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/floor. */
  public static final MethodReference FLOOR =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/floor")),
          AstMethodReference.fnSelector);

  private static final String FLOOR_SIGNATURE = "tf.math.floor()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/ceil. */
  public static final MethodReference CEIL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/ceil")),
          AstMethodReference.fnSelector);

  private static final String CEIL_SIGNATURE = "tf.math.ceil()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/sign. */
  public static final MethodReference SIGN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/sign")),
          AstMethodReference.fnSelector);

  private static final String SIGN_SIGNATURE = "tf.math.sign()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/equal. */
  public static final MethodReference EQUAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/equal")),
          AstMethodReference.fnSelector);

  private static final String EQUAL_SIGNATURE = "tf.equal()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/not_equal. */
  public static final MethodReference NOT_EQUAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/not_equal")),
          AstMethodReference.fnSelector);

  private static final String NOT_EQUAL_SIGNATURE = "tf.not_equal()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/less. */
  public static final MethodReference LESS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/less")),
          AstMethodReference.fnSelector);

  private static final String LESS_SIGNATURE = "tf.less()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/less_equal. */
  public static final MethodReference LESS_EQUAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/less_equal")),
          AstMethodReference.fnSelector);

  private static final String LESS_EQUAL_SIGNATURE = "tf.less_equal()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/greater. */
  public static final MethodReference GREATER =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/greater")),
          AstMethodReference.fnSelector);

  private static final String GREATER_SIGNATURE = "tf.greater()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/greater_equal. */
  public static final MethodReference GREATER_EQUAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/greater_equal")),
          AstMethodReference.fnSelector);

  private static final String GREATER_EQUAL_SIGNATURE = "tf.greater_equal()";

  /** https://www.tensorflow.org/api_docs/python/tf/cast. */
  public static final MethodReference CAST =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/cast")),
          AstMethodReference.fnSelector);

  private static final String CAST_SIGNATURE = "tf.cast()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits. */
  public static final MethodReference SOFTMAX_CROSS_ENTROPY_WITH_LOGITS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/softmax_cross_entropy_with_logits")),
          AstMethodReference.fnSelector);

  private static final String SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SIGNATURE =
      "tf.nn.softmax_cross_entropy_with_logits()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits. */
  public static final MethodReference SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName(
                  "Ltensorflow/functions/sparse_softmax_cross_entropy_with_logits")),
          AstMethodReference.fnSelector);

  private static final String SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SIGNATURE =
      "tf.nn.sparse_softmax_cross_entropy_with_logits()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/log. */
  public static final MethodReference LOG =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/log")),
          AstMethodReference.fnSelector);

  private static final String LOG_SIGNATURE = "tf.log()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/reduce_sum. */
  public static final MethodReference REDUCE_SUM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/reduce_sum")),
          AstMethodReference.fnSelector);

  private static final String REDUCE_SUM_SIGNATURE = "tf.reduce_sum()";

  /** https://www.tensorflow.org/api_docs/python/tf/matmul. */
  public static final MethodReference MATMUL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/matmul")),
          AstMethodReference.fnSelector);

  private static final String MATMUL_SIGNATURE = "tf.matmul()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid. */
  public static final MethodReference SIGMOID =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/sigmoid")),
          AstMethodReference.fnSelector);

  private static final String SIGMOID_SIGNATURE = "tf.nn.sigmoid()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/exp. */
  public static final MethodReference EXP =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/exp")),
          AstMethodReference.fnSelector);

  private static final String EXP_SIGNATURE = "tf.math.exp()";

  /** https://www.tensorflow.org/api_docs/python/tf/math/rsqrt. */
  public static final MethodReference RSQRT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/rsqrt")),
          AstMethodReference.fnSelector);

  private static final String RSQRT_SIGNATURE = "tf.math.rsqrt()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/log_softmax. */
  public static final MethodReference LOG_SOFTMAX =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/math/log_softmax")),
          AstMethodReference.fnSelector);

  private static final String LOG_SOFTMAX_SIGNATURE = "tf.nn.log_softmax()";

  /** https://www.tensorflow.org/api_docs/python/tf/rank. */
  public static final MethodReference RANK =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/rank")),
          AstMethodReference.fnSelector);

  private static final String RANK_SIGNATURE = "tf.rank()";

  /** https://www.tensorflow.org/api_docs/python/tf/size. */
  public static final MethodReference SIZE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/size")),
          AstMethodReference.fnSelector);

  private static final String SIZE_SIGNATURE = "tf.size()";

  /** https://www.tensorflow.org/api_docs/python/tf/identity. */
  public static final MethodReference IDENTITY =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/identity")),
          AstMethodReference.fnSelector);

  private static final String IDENTITY_SIGNATURE = "tf.identity()";

  /** https://www.tensorflow.org/api_docs/python/tf/stop_gradient. */
  public static final MethodReference STOP_GRADIENT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/stop_gradient")),
          AstMethodReference.fnSelector);

  private static final String STOP_GRADIENT_SIGNATURE = "tf.stop_gradient()";

  /** https://www.tensorflow.org/api_docs/python/tf/GradientTape#gradient. */
  public static final MethodReference GRADIENT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/gradient")),
          AstMethodReference.fnSelector);

  private static final String GRADIENT_SIGNATURE = "tf.GradientTape.gradient()";

  /** https://www.tensorflow.org/api_docs/python/tf/nn/softmax. */
  public static final MethodReference SOFTMAX =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/softmax")),
          AstMethodReference.fnSelector);

  private static final String SOFTMAX_SIGNATURE = "tf.nn.softmax()";

  /**
   * https://github.com/keras-team/keras/blob/f6c4ac55692c132cd16211f4877fac6dbeead749/keras/src/layers/core/dense.py#L149-L155.
   */
  public static final MethodReference DENSE_CALL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/keras/layers/Dense/" + CALLABLE_METHOD_NAME)),
          AstMethodReference.fnSelector);

  private static final String DENSE_CALL_SIGNATURE =
      "tf.keras.layers.Dense." + CALLABLE_METHOD_NAME + "()";

  /** https://www.tensorflow.org/api_docs/python/tf/layers/flatten. */
  public static final MethodReference FLATTEN =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/flatten")),
          AstMethodReference.fnSelector);

  private static final String FLATTEN_SIGNATURE = "tf.layers.flatten()";

  /**
   * The {@code __call__} synthetic method on a {@code tf.keras.layers.Flatten} instance.
   *
   * @see <a
   *     href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten">tf.keras.layers.Flatten</a>
   */
  public static final MethodReference FLATTEN_LAYER_CALL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/keras/layers/Flatten/" + CALLABLE_METHOD_NAME)),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/nn/max_pool. */
  public static final MethodReference MAX_POOL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/max_pool")),
          AstMethodReference.fnSelector);

  private static final String MAX_POOL_SIGNATURE = "tf.nn.max_pool()";

  /** https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer. */
  public static final MethodReference ADAM_OPTIMIZER =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/AdamOptimizer")),
          AstMethodReference.fnSelector);

  private static final String ADAM_OPTIMIZER_SIGNATURE = "tf.train.AdamOptimizer()";

  /** https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer. */
  public static final MethodReference GRADIENT_DESCENT_OPTIMIZER =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/GradientDescentOptimizer")),
          AstMethodReference.fnSelector);

  private static final String GRADIENT_DESCENT_OPTIMIZER_SIGNATURE =
      "tf.train.GradientDescentOptimizer()";

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
          Map.entry(DATASET, DATASET_SIGNATURE),
          Map.entry(DATASET_BATCH_TYPE, DATASET_BATCH_SIGNATURE),
          Map.entry(DATASET_PADDED_BATCH_TYPE, DATASET_PADDED_BATCH_SIGNATURE),
          Map.entry(DATASET_SHUFFLE_TYPE, DATASET_SHUFFLE_SIGNATURE),
          Map.entry(DATASET_MAP_TYPE, DATASET_MAP_SIGNATURE),
          Map.entry(DATASET_REPEAT_TYPE, DATASET_REPEAT_SIGNATURE),
          Map.entry(DATASET_PREFETCH_TYPE, DATASET_PREFETCH_SIGNATURE),
          Map.entry(DATASET_TAKE_TYPE, DATASET_TAKE_SIGNATURE),
          Map.entry(DATASET_WITH_OPTIONS_TYPE, DATASET_WITH_OPTIONS_SIGNATURE),
          Map.entry(DATASET_CONCATENATE_TYPE, DATASET_CONCATENATE_SIGNATURE),
          Map.entry(DATASET_ENUMERATE_TYPE, DATASET_ENUMERATE_SIGNATURE),
          Map.entry(DATASET_REDUCE_TYPE, DATASET_REDUCE_SIGNATURE),
          Map.entry(DATASET_FILTER_TYPE, DATASET_FILTER_SIGNATURE),
          Map.entry(DATASET_FROM_TENSOR_SLICES_TYPE, DATASET_FROM_TENSOR_SLICES_SIGNATURE),
          Map.entry(DATASET_RANGE_TYPE, DATASET_RANGE_SIGNATURE),
          Map.entry(TEXT_LINE_DATASET_TYPE, TEXT_LINE_DATASET_SIGNATURE),
          Map.entry(IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE, FLOW_FROM_DIRECTORY_SIGNATURE),
          Map.entry(DATASET_FROM_GENERATOR_TYPE, DATASET_FROM_GENERATOR_SIGNATURE),
          Map.entry(DATASET_FROM_TENSORS_TYPE, DATASET_FROM_TENSORS_SIGNATURE),
          Map.entry(DATASET_CHOOSE_FROM_DATASETS_TYPE, DATASET_CHOOSE_FROM_DATASETS_SIGNATURE),
          Map.entry(DATASET_SAMPLE_FROM_DATASETS_TYPE, DATASET_SAMPLE_FROM_DATASETS_SIGNATURE),
          Map.entry(DATASET_ZIP_TYPE, DATASET_ZIP_SIGNATURE),
          Map.entry(DATASET_RANDOM_TYPE, DATASET_RANDOM_SIGNATURE),
          Map.entry(TENSOR_SPEC, TENSOR_SPEC_SIGNATURE),
          Map.entry(RAGGED_TENSOR_SPEC, RAGGED_TENSOR_SPEC_SIGNATURE),
          Map.entry(RESHAPE.getDeclaringClass(), RESHAPE_SIGNATURE),
          Map.entry(CONSTANT.getDeclaringClass(), CONSTANT_SIGNATURE),
          Map.entry(RANGE.getDeclaringClass(), RANGE_SIGNATURE),
          Map.entry(NORMAL.getDeclaringClass(), NORMAL_SIGNATURE),
          Map.entry(TRUNCATED_NORMAL.getDeclaringClass(), TRUNCATED_NORMAL_SIGNATURE),
          Map.entry(ZEROS_LIKE.getDeclaringClass(), ZEROS_LIKE_SIGNATURE),
          Map.entry(FILL.getDeclaringClass(), FILL_SIGNATURE),
          Map.entry(LINSPACE.getDeclaringClass(), LINSPACE_SIGNATURE),
          Map.entry(BROADCAST_TO.getDeclaringClass(), BROADCAST_TO_SIGNATURE),
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
          Map.entry(REDUCE_MEAN.getDeclaringClass(), REDUCE_MEAN_SIGNATURE),
          Map.entry(REDUCE_MAX.getDeclaringClass(), REDUCE_MAX_SIGNATURE),
          Map.entry(REDUCE_PROD.getDeclaringClass(), REDUCE_PROD_SIGNATURE),
          Map.entry(REDUCE_LOGSUMEXP.getDeclaringClass(), REDUCE_LOGSUMEXP_SIGNATURE),
          Map.entry(UNSORTED_SEGMENT_SUM.getDeclaringClass(), UNSORTED_SEGMENT_SUM_SIGNATURE),
          Map.entry(UNSORTED_SEGMENT_MAX.getDeclaringClass(), UNSORTED_SEGMENT_MAX_SIGNATURE),
          Map.entry(UNSORTED_SEGMENT_MEAN.getDeclaringClass(), UNSORTED_SEGMENT_MEAN_SIGNATURE),
          Map.entry(REDUCE_ALL.getDeclaringClass(), REDUCE_ALL_SIGNATURE),
          Map.entry(MODEL.getDeclaringClass(), MODEL_SIGNATURE),
          Map.entry(MODEL_CALL.getDeclaringClass(), MODEL_CALL_SIGNATURE),
          Map.entry(TENSOR.getDeclaringClass(), TENSOR_SIGNATURE),
          Map.entry(NDARRAY.getDeclaringClass(), NDARRAY_SIGNATURE),
          Map.entry(READ_DATA_SETS.getDeclaringClass(), READ_DATA_SETS_SIGNATURE),
          Map.entry(MNIST_X_TRAIN, MNIST_X_TRAIN_SIGNATURE),
          Map.entry(MNIST_Y_TRAIN, MNIST_Y_TRAIN_SIGNATURE),
          Map.entry(MNIST_X_TEST, MNIST_X_TEST_SIGNATURE),
          Map.entry(MNIST_Y_TEST, MNIST_Y_TEST_SIGNATURE),
          Map.entry(CIFAR10_X_TRAIN, CIFAR10_X_TRAIN_SIGNATURE),
          Map.entry(CIFAR10_Y_TRAIN, CIFAR10_Y_TRAIN_SIGNATURE),
          Map.entry(CIFAR10_X_TEST, CIFAR10_X_TEST_SIGNATURE),
          Map.entry(CIFAR10_Y_TEST, CIFAR10_Y_TEST_SIGNATURE),
          Map.entry(IMDB_X_TRAIN, IMDB_X_TRAIN_SIGNATURE),
          Map.entry(IMDB_Y_TRAIN, IMDB_Y_TRAIN_SIGNATURE),
          Map.entry(IMDB_X_TEST, IMDB_X_TEST_SIGNATURE),
          Map.entry(IMDB_Y_TEST, IMDB_Y_TEST_SIGNATURE),
          Map.entry(FASHION_MNIST_X_TRAIN, FASHION_MNIST_X_TRAIN_SIGNATURE),
          Map.entry(FASHION_MNIST_Y_TRAIN, FASHION_MNIST_Y_TRAIN_SIGNATURE),
          Map.entry(FASHION_MNIST_X_TEST, FASHION_MNIST_X_TEST_SIGNATURE),
          Map.entry(FASHION_MNIST_Y_TEST, FASHION_MNIST_Y_TEST_SIGNATURE),
          Map.entry(CIFAR100_X_TRAIN, CIFAR100_X_TRAIN_SIGNATURE),
          Map.entry(CIFAR100_Y_TRAIN, CIFAR100_Y_TRAIN_SIGNATURE),
          Map.entry(CIFAR100_X_TEST, CIFAR100_X_TEST_SIGNATURE),
          Map.entry(CIFAR100_Y_TEST, CIFAR100_Y_TEST_SIGNATURE),
          Map.entry(REUTERS_X_TRAIN, REUTERS_X_TRAIN_SIGNATURE),
          Map.entry(REUTERS_Y_TRAIN, REUTERS_Y_TRAIN_SIGNATURE),
          Map.entry(REUTERS_X_TEST, REUTERS_X_TEST_SIGNATURE),
          Map.entry(REUTERS_Y_TEST, REUTERS_Y_TEST_SIGNATURE),
          Map.entry(BOSTON_HOUSING_X_TRAIN, BOSTON_HOUSING_X_TRAIN_SIGNATURE),
          Map.entry(BOSTON_HOUSING_Y_TRAIN, BOSTON_HOUSING_Y_TRAIN_SIGNATURE),
          Map.entry(BOSTON_HOUSING_X_TEST, BOSTON_HOUSING_X_TEST_SIGNATURE),
          Map.entry(BOSTON_HOUSING_Y_TEST, BOSTON_HOUSING_Y_TEST_SIGNATURE),
          Map.entry(PLACEHOLDER.getDeclaringClass(), PLACEHOLDER_SIGNATURE),
          Map.entry(ARGMAX.getDeclaringClass(), ARGMAX_SIGNATURE),
          Map.entry(ARGMIN.getDeclaringClass(), ARGMIN_SIGNATURE),
          Map.entry(TENSORDOT.getDeclaringClass(), TENSORDOT_SIGNATURE),
          Map.entry(TRACE.getDeclaringClass(), TRACE_SIGNATURE),
          Map.entry(TRANSPOSE.getDeclaringClass(), TRANSPOSE_SIGNATURE),
          Map.entry(DIAG.getDeclaringClass(), DIAG_SIGNATURE),
          Map.entry(DIAG_PART.getDeclaringClass(), DIAG_PART_SIGNATURE),
          Map.entry(MATRIX_TRANSPOSE.getDeclaringClass(), MATRIX_TRANSPOSE_SIGNATURE),
          Map.entry(ADJOINT.getDeclaringClass(), ADJOINT_SIGNATURE),
          Map.entry(TILE.getDeclaringClass(), TILE_SIGNATURE),
          Map.entry(
              TENSOR_SCATTER_ND_UPDATE.getDeclaringClass(), TENSOR_SCATTER_ND_UPDATE_SIGNATURE),
          Map.entry(SEQUENCE_MASK.getDeclaringClass(), SEQUENCE_MASK_SIGNATURE),
          Map.entry(EMBEDDING_LOOKUP.getDeclaringClass(), EMBEDDING_LOOKUP_SIGNATURE),
          Map.entry(GATHER_ND.getDeclaringClass(), GATHER_ND_SIGNATURE),
          Map.entry(BOOLEAN_MASK.getDeclaringClass(), BOOLEAN_MASK_SIGNATURE),
          Map.entry(SLICE.getDeclaringClass(), SLICE_SIGNATURE),
          Map.entry(SQUEEZE.getDeclaringClass(), SQUEEZE_SIGNATURE),
          Map.entry(EXTRACT_PATCHES.getDeclaringClass(), EXTRACT_PATCHES_SIGNATURE),
          Map.entry(TAN.getDeclaringClass(), TAN_SIGNATURE),
          Map.entry(ASIN.getDeclaringClass(), ASIN_SIGNATURE),
          Map.entry(ATAN.getDeclaringClass(), ATAN_SIGNATURE),
          Map.entry(SINH.getDeclaringClass(), SINH_SIGNATURE),
          Map.entry(COSH.getDeclaringClass(), COSH_SIGNATURE),
          Map.entry(ASINH.getDeclaringClass(), ASINH_SIGNATURE),
          Map.entry(ACOSH.getDeclaringClass(), ACOSH_SIGNATURE),
          Map.entry(ATANH.getDeclaringClass(), ATANH_SIGNATURE),
          Map.entry(LOG1P.getDeclaringClass(), LOG1P_SIGNATURE),
          Map.entry(EXPM1.getDeclaringClass(), EXPM1_SIGNATURE),
          Map.entry(ROUND.getDeclaringClass(), ROUND_SIGNATURE),
          Map.entry(RECIPROCAL.getDeclaringClass(), RECIPROCAL_SIGNATURE),
          Map.entry(SOFTPLUS.getDeclaringClass(), SOFTPLUS_SIGNATURE),
          Map.entry(SOFTSIGN.getDeclaringClass(), SOFTSIGN_SIGNATURE),
          Map.entry(SQUARE.getDeclaringClass(), SQUARE_SIGNATURE),
          Map.entry(ERF.getDeclaringClass(), ERF_SIGNATURE),
          Map.entry(ERFC.getDeclaringClass(), ERFC_SIGNATURE),
          Map.entry(ATAN2.getDeclaringClass(), ATAN2_SIGNATURE),
          Map.entry(MAXIMUM.getDeclaringClass(), MAXIMUM_SIGNATURE),
          Map.entry(MINIMUM.getDeclaringClass(), MINIMUM_SIGNATURE),
          Map.entry(EINSUM.getDeclaringClass(), EINSUM_SIGNATURE),
          Map.entry(RELU.getDeclaringClass(), RELU_SIGNATURE),
          Map.entry(EXPAND_DIMS.getDeclaringClass(), EXPAND_DIMS_SIGNATURE),
          Map.entry(CLIP_BY_VALUE.getDeclaringClass(), CLIP_BY_VALUE_SIGNATURE),
          Map.entry(AS_STRING.getDeclaringClass(), AS_STRING_SIGNATURE),
          Map.entry(TOP_K.getDeclaringClass(), TOP_K_SIGNATURE),
          Map.entry(MESHGRID.getDeclaringClass(), MESHGRID_SIGNATURE),
          Map.entry(WHERE.getDeclaringClass(), WHERE_SIGNATURE),
          Map.entry(LEAKY_RELU.getDeclaringClass(), LEAKY_RELU_SIGNATURE),
          Map.entry(REDUCE_MIN.getDeclaringClass(), REDUCE_MIN_SIGNATURE),
          Map.entry(POW.getDeclaringClass(), POW_SIGNATURE),
          Map.entry(CONCAT.getDeclaringClass(), CONCAT_SIGNATURE),
          Map.entry(STACK.getDeclaringClass(), STACK_SIGNATURE),
          Map.entry(SQRT.getDeclaringClass(), SQRT_SIGNATURE),
          Map.entry(NEGATIVE.getDeclaringClass(), NEGATIVE_SIGNATURE),
          Map.entry(SIN.getDeclaringClass(), SIN_SIGNATURE),
          Map.entry(COS.getDeclaringClass(), COS_SIGNATURE),
          Map.entry(FLOOR.getDeclaringClass(), FLOOR_SIGNATURE),
          Map.entry(CEIL.getDeclaringClass(), CEIL_SIGNATURE),
          Map.entry(SIGN.getDeclaringClass(), SIGN_SIGNATURE),
          Map.entry(EQUAL.getDeclaringClass(), EQUAL_SIGNATURE),
          Map.entry(NOT_EQUAL.getDeclaringClass(), NOT_EQUAL_SIGNATURE),
          Map.entry(LESS.getDeclaringClass(), LESS_SIGNATURE),
          Map.entry(LESS_EQUAL.getDeclaringClass(), LESS_EQUAL_SIGNATURE),
          Map.entry(GREATER.getDeclaringClass(), GREATER_SIGNATURE),
          Map.entry(GREATER_EQUAL.getDeclaringClass(), GREATER_EQUAL_SIGNATURE),
          Map.entry(CAST.getDeclaringClass(), CAST_SIGNATURE),
          Map.entry(
              SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass(),
              SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SIGNATURE),
          Map.entry(
              SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS.getDeclaringClass(),
              SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SIGNATURE),
          Map.entry(LOG.getDeclaringClass(), LOG_SIGNATURE),
          Map.entry(REDUCE_SUM.getDeclaringClass(), REDUCE_SUM_SIGNATURE),
          Map.entry(MATMUL.getDeclaringClass(), MATMUL_SIGNATURE),
          Map.entry(SIGMOID.getDeclaringClass(), SIGMOID_SIGNATURE),
          Map.entry(EXP.getDeclaringClass(), EXP_SIGNATURE),
          Map.entry(RSQRT.getDeclaringClass(), RSQRT_SIGNATURE),
          Map.entry(LOG_SOFTMAX.getDeclaringClass(), LOG_SOFTMAX_SIGNATURE),
          Map.entry(RANK.getDeclaringClass(), RANK_SIGNATURE),
          Map.entry(SIZE.getDeclaringClass(), SIZE_SIGNATURE),
          Map.entry(IDENTITY.getDeclaringClass(), IDENTITY_SIGNATURE),
          Map.entry(STOP_GRADIENT.getDeclaringClass(), STOP_GRADIENT_SIGNATURE),
          Map.entry(GRADIENT.getDeclaringClass(), GRADIENT_SIGNATURE),
          Map.entry(SOFTMAX.getDeclaringClass(), SOFTMAX_SIGNATURE),
          Map.entry(DENSE_CALL.getDeclaringClass(), DENSE_CALL_SIGNATURE),
          Map.entry(FLATTEN.getDeclaringClass(), FLATTEN_SIGNATURE),
          Map.entry(MAX_POOL.getDeclaringClass(), MAX_POOL_SIGNATURE),
          Map.entry(ADAM_OPTIMIZER.getDeclaringClass(), ADAM_OPTIMIZER_SIGNATURE),
          Map.entry(
              GRADIENT_DESCENT_OPTIMIZER.getDeclaringClass(),
              GRADIENT_DESCENT_OPTIMIZER_SIGNATURE));

  /**
   * Represents the TensorFlow float32 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#float32">TensorFlow
   *     float32 DType</a>.
   */
  public static final FieldReference FLOAT_32 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(FLOAT32.name().toLowerCase(Locale.ROOT)), D_TYPE);

  /**
   * Represents the TensorFlow float64 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#float64">TensorFlow
   *     float64 DType</a>.
   */
  public static final FieldReference FLOAT_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(FLOAT64.name().toLowerCase(Locale.ROOT)), D_TYPE);

  /**
   * Represents the TensorFlow int32 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#int32">TensorFlow
   *     int32 DType</a>.
   */
  public static final FieldReference INT_32 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(INT32.name().toLowerCase(Locale.ROOT)), D_TYPE);

  /**
   * Represents the TensorFlow int64 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#int64">TensorFlow
   *     int64 DType</a>.
   */
  public static final FieldReference INT_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(INT64.name().toLowerCase(Locale.ROOT)), D_TYPE);

  /**
   * Represents the TensorFlow uint8 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#uint8">TensorFlow
   *     uint8 DType</a>.
   */
  public static final FieldReference UINT_8 =
      FieldReference.findOrCreate(
          PythonTypes.Root, findOrCreateAsciiAtom(UINT8.name().toLowerCase(Locale.ROOT)), D_TYPE);

  /**
   * Represents the TensorFlow string data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#string">TensorFlow
   *     string DType</a>.
   */
  public static final FieldReference STRING =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(DType.STRING.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  /**
   * Represents the TensorFlow complex64 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#complex64">TensorFlow
   *     complex64 DType</a>.
   */
  public static final FieldReference COMPLEX_64 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(COMPLEX64.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  /**
   * Represents the TensorFlow complex128 data type.
   *
   * @see <a
   *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/dtypes#complex128">TensorFlow
   *     complex128 DType</a>.
   */
  public static final FieldReference COMPLEX_128 =
      FieldReference.findOrCreate(
          PythonTypes.Root,
          findOrCreateAsciiAtom(COMPLEX128.name().toLowerCase(Locale.ROOT)),
          D_TYPE);

  /** A mapping from a field reference to its associated {@link DType}, if any. */
  public static final Map<FieldReference, DType> FIELD_REFERENCE_TO_DTYPE =
      Map.ofEntries(
          Map.entry(FLOAT_32, FLOAT32),
          Map.entry(FLOAT_64, FLOAT64),
          Map.entry(INT_32, INT32),
          Map.entry(INT_64, INT64),
          Map.entry(UINT_8, UINT8),
          Map.entry(COMPLEX_64, COMPLEX64),
          Map.entry(COMPLEX_128, COMPLEX128),
          Map.entry(STRING, DType.STRING));

  private TensorFlowTypes() {}
}
