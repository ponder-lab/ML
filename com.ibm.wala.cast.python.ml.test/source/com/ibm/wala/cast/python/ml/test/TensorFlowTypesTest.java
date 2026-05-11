package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static java.util.Collections.emptyList;
import static org.junit.Assert.*;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
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

  @Test
  public void testTensorTypeGetDTypeRoundtripFloat32() {
    TensorType t = new TensorType(FLOAT32.name().toLowerCase(), emptyList());
    assertEquals(FLOAT32, t.getDType());
  }

  @Test
  public void testTensorTypeGetDTypeRoundtripInt64() {
    TensorType t = new TensorType(INT64.name().toLowerCase(), emptyList());
    assertEquals(INT64, t.getDType());
  }

  @Test
  public void testTensorTypeGetDTypeUnknownCellTypeThrowsIllegalStateException() {
    TensorType t = new TensorType("not_a_real_dtype", emptyList());
    assertThrows(IllegalStateException.class, t::getDType);
  }

  @Test
  public void testTensorTypeGetDTypeUnknownEnumValueRoundtrips() {
    // DType.UNKNOWN is a real enum value; its cellType "unknown" must round-trip too.
    TensorType t = new TensorType(DType.UNKNOWN.name().toLowerCase(), emptyList());
    assertEquals(DType.UNKNOWN, t.getDType());
  }
}
