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
