package com.ibm.wala.cast.python.ml.types;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
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

  /** https://www.tensorflow.org/api_docs/python/tf/ones. */
  public static final MethodReference ONES =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/ones")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/constant. */
  public static final MethodReference CONSTANT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/constant")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/keras/Input. */
  public static final MethodReference INPUT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/keras/layers/Input")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/range. */
  public static final MethodReference RANGE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/range")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/random/uniform. */
  public static final MethodReference UNIFORM =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/uniform")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/random/normal. */
  public static final MethodReference NORMAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/normal")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/random/truncated_normal. */
  public static final MethodReference TRUNCATED_NORMAL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/truncated_normal")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/zeros. */
  public static final MethodReference ZEROS =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/zeros")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/zeros_like. */
  public static final MethodReference ZEROS_LIKE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/zeros_like")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/fill. */
  public static final MethodReference FILL =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/fill")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor. */
  public static final MethodReference CONVERT_TO_TENSOR =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/convert_to_tensor")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/one_hot. */
  public static final MethodReference ONE_HOT =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/one_hot")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/eye. */
  public static final MethodReference EYE =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/eye")),
          AstMethodReference.fnSelector);

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

  /** https://www.tensorflow.org/api_docs/python/tf/sparse/add. */
  public static final MethodReference SPARSE_ADD =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader,
              TypeName.string2TypeName("Ltensorflow/functions/sparse_add")),
          AstMethodReference.fnSelector);

  private static final String SPARSE_ADD_SIGNATURE = "tf.sparse.add()";

  /** https://www.tensorflow.org/api_docs/python/tf/gamma. */
  public static final MethodReference GAMMA =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/gamma")),
          AstMethodReference.fnSelector);

  /** https://www.tensorflow.org/api_docs/python/tf/poisson. */
  public static final MethodReference POISSON =
      MethodReference.findOrCreate(
          TypeReference.findOrCreate(
              PythonTypes.pythonLoader, TypeName.string2TypeName("Ltensorflow/functions/poisson")),
          AstMethodReference.fnSelector);

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

  /** A mapping from a {@link TypeReference} to its associated TensorFlow signature. */
  public static final Map<TypeReference, String> TYPE_REFERENCE_TO_SIGNATURE =
      Map.of(
          SPARSE_TENSOR.getDeclaringClass(),
          SPARSE_TENSOR_SIGNATURE,
          SPARSE_ADD.getDeclaringClass(),
          SPARSE_ADD_SIGNATURE,
          RAGGED_CONSTANT.getDeclaringClass(),
          RAGGED_CONSTANT_SIGNATURE,
          RAGGED_RANGE.getDeclaringClass(),
          RAGGED_RANGE_SIGNATURE,
          MULTIPLY.getDeclaringClass(),
          MULTIPLY_SIGNATURE,
          ADD.getDeclaringClass(),
          ADD_SIGNATURE,
          SUBTRACT.getDeclaringClass(),
          SUBTRACT_SIGNATURE,
          DIVIDE.getDeclaringClass(),
          DIVIDE_SIGNATURE);

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
   * Represents the TensorFlow float65 data type.
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

  /** A mapping from a field reference to its associated {@link DType}, if any. */
  public static final Map<FieldReference, DType> FIELD_REFERENCE_TO_DTYPE =
      Map.of(FLOAT_32, FLOAT32, FLOAT_64, FLOAT64, INT_32, INT32);

  private TensorFlowTypes() {}
}
