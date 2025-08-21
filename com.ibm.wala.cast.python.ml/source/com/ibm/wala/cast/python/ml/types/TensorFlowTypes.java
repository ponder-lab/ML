package com.ibm.wala.cast.python.ml.types;

import com.ibm.wala.cast.python.types.PythonTypes;
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
   * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
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

  private TensorFlowTypes() {}
}
