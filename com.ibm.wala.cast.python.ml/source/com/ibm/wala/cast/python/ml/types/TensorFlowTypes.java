package com.ibm.wala.cast.python.ml.types;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;

import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;

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
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    STRING;
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

  private TensorFlowTypes() {}
}
