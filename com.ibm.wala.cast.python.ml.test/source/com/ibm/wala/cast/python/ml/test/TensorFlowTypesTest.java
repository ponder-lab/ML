package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static org.junit.Assert.*;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import org.junit.Test;

public class TensorFlowTypesTest {

  @Test
  public void testCanConvertTo() {
    assertTrue(FLOAT32.canConvertTo(FLOAT32));
    assertTrue(FLOAT32.canConvertTo(FLOAT64));
    assertFalse(FLOAT64.canConvertTo(FLOAT32));
    assertFalse(FLOAT64.canConvertTo(DType.STRING));
    assertFalse(STRING.canConvertTo(FLOAT32));
    assertTrue(STRING.canConvertTo(DType.STRING));
    assertTrue(INT32.canConvertTo(DType.INT32));
    assertTrue(INT32.canConvertTo(DType.FLOAT32));
    assertFalse(INT32.canConvertTo(DType.STRING));
    assertFalse(STRING.canConvertTo(DType.INT32));
    assertTrue(INT64.canConvertTo(DType.FLOAT64));
    assertFalse(INT64.canConvertTo(DType.FLOAT32));
    assertFalse(INT64.canConvertTo(DType.INT32));
    assertTrue(INT32.canConvertTo(DType.INT64));
    assertTrue(FLOAT64.canConvertTo(DType.FLOAT64));
    assertFalse(FLOAT32.canConvertTo(DType.INT32));
    assertTrue(FLOAT32.canConvertTo(DType.FLOAT32));
    assertFalse(FLOAT64.canConvertTo(DType.INT64));
    assertFalse(FLOAT64.canConvertTo(DType.INT32));
  }
}
