package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import org.junit.Test;

public class TensorTypeTest {

  @Test
  public void testGetDTypeRoundtripFloat32() {
    TensorType t = new TensorType(FLOAT32.name().toLowerCase(), emptyList());
    assertEquals(FLOAT32, t.getDType());
  }

  @Test
  public void testGetDTypeRoundtripInt64() {
    TensorType t = new TensorType(INT64.name().toLowerCase(), emptyList());
    assertEquals(INT64, t.getDType());
  }

  @Test
  public void testGetDTypeUnknownCellTypeThrowsIllegalStateException() {
    TensorType t = new TensorType("not_a_real_dtype", emptyList());
    assertThrows(IllegalStateException.class, t::getDType);
  }

  @Test
  public void testGetDTypeUnknownEnumValueRoundtrips() {
    // DType.UNKNOWN is a real enum value; its cellType "unknown" must round-trip too.
    TensorType t = new TensorType(DType.UNKNOWN.name().toLowerCase(), emptyList());
    assertEquals(DType.UNKNOWN, t.getDType());
  }
}
