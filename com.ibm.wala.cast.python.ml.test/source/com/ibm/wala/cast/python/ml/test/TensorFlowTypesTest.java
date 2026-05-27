package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT64;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Locale.ROOT;
import static org.junit.Assert.*;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
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
    TensorType t = new TensorType(FLOAT32.name().toLowerCase(ROOT), emptyList());
    assertEquals(FLOAT32, t.getDType());
  }

  @Test
  public void testTensorTypeGetDTypeRoundtripInt64() {
    TensorType t = new TensorType(INT64.name().toLowerCase(ROOT), emptyList());
    assertEquals(INT64, t.getDType());
  }

  @Test
  public void testTensorTypeUnknownCellTypeRejectedAtConstruction() {
    // wala/ML#533: dtype is the internal source of truth; the String ctor must reject any
    // cellType that doesn't map to a known DType at construction time (eager, fail-fast).
    assertThrows(
        IllegalArgumentException.class, () -> new TensorType("not_a_real_dtype", emptyList()));
  }

  @Test
  public void testTensorTypeGetDTypeUnknownEnumValueRoundtrips() {
    // DType.UNKNOWN is a real enum value; its cellType "unknown" must round-trip too.
    TensorType t = new TensorType(DType.UNKNOWN.name().toLowerCase(ROOT), emptyList());
    assertEquals(DType.UNKNOWN, t.getDType());
  }

  // wala/ML#533: TensorType(DType, List<Dimension<?>>) ctor — equivalence with the String form.

  @Test
  public void testTensorTypeDTypeCtorEquivalentToStringCtorFloat32() {
    TensorType viaDType = new TensorType(FLOAT32, emptyList());
    TensorType viaString = new TensorType(FLOAT32.name().toLowerCase(ROOT), emptyList());
    assertEquals(viaString, viaDType);
    assertEquals(viaString.hashCode(), viaDType.hashCode());
    assertEquals(FLOAT32, viaDType.getDType());
  }

  @Test
  public void testTensorTypeDTypeCtorEquivalentToStringCtorInt64() {
    TensorType viaDType = new TensorType(INT64, emptyList());
    TensorType viaString = new TensorType(INT64.name().toLowerCase(ROOT), emptyList());
    assertEquals(viaString, viaDType);
    assertEquals(INT64, viaDType.getDType());
  }

  @Test
  public void testTensorTypeDTypeCtorNullDtypeThrows() {
    assertThrows(NullPointerException.class, () -> new TensorType((DType) null, emptyList()));
  }

  @Test
  public void testTensorTypeDTypeCtorNullDimsAllowed() {
    // Null dims is a valid ⊤ shape per the String-ctor contract; the DType ctor must accept it too.
    TensorType t = new TensorType(FLOAT32, null);
    assertEquals(FLOAT32, t.getDType());
    assertEquals(new TensorType(FLOAT32.name().toLowerCase(ROOT), null), t);
  }

  // wala/ML#532 regression guards: TensorType honors equals/hashCode on (cellType, dims).

  @Test
  public void testTensorTypeEqualsSameCellTypeAndDims() {
    TensorType a = new TensorType("float32", asList(new NumericDim(2), new NumericDim(3)));
    TensorType b = new TensorType("float32", asList(new NumericDim(2), new NumericDim(3)));
    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  public void testTensorTypeNotEqualsDifferentCellType() {
    TensorType a = new TensorType("float32", asList(new NumericDim(3)));
    TensorType b = new TensorType("int32", asList(new NumericDim(3)));
    assertNotEquals(a, b);
  }

  @Test
  public void testTensorTypeNotEqualsDifferentDims() {
    TensorType a = new TensorType("float32", asList(new NumericDim(2), new NumericDim(3)));
    TensorType b = new TensorType("float32", asList(new NumericDim(3), new NumericDim(2)));
    assertNotEquals(a, b);
  }

  @Test
  public void testTensorTypeNotEqualsNumericVsSymbolicDim() {
    TensorType a = new TensorType("float32", asList(new NumericDim(3)));
    TensorType b = new TensorType("float32", asList(new SymbolicDim("3")));
    assertNotEquals(a, b);
  }

  @Test
  public void testTensorTypeEqualsBothNullDims() {
    TensorType a = new TensorType("float32", null);
    TensorType b = new TensorType("float32", null);
    assertEquals(a, b);
    assertEquals(a.hashCode(), b.hashCode());
  }

  @Test
  public void testTensorTypeNotEqualsOneNullDims() {
    TensorType a = new TensorType("float32", null);
    TensorType b = new TensorType("float32", emptyList());
    assertNotEquals(a, b);
  }

  @Test
  public void testTensorTypeNotEqualsNull() {
    TensorType a = new TensorType("float32", asList(new NumericDim(3)));
    assertNotEquals(a, null);
  }

  @Test
  public void testDynamicDimRendering() {
    // Cover `DynamicDim`'s package-private overrides through `TensorType`'s public
    // string-rendering methods (https://github.com/wala/ML/issues/545).
    TensorType t = new TensorType("float32", asList(DynamicDim.INSTANCE, new NumericDim(4)));
    assertTrue(t.toMDString().contains("*dynamic*"));
    assertTrue(t.toCString(false).contains("dynamic"));
    assertTrue(t.toCString(true).contains("*dynamic*"));
  }

  @Test
  public void testVoidPayloadDimensionToString() {
    // `Void`-payload sentinels skip the ",value" suffix so they don't collide visually
    // with the ", " dim separator in shape renderings (https://github.com/wala/ML/issues/558).
    assertEquals("D:Dynamic", DynamicDim.INSTANCE.toString());
    assertEquals("D:Ragged", RaggedDim.INSTANCE.toString());
    // Value-carrying dims still include the value suffix.
    assertEquals("D:Constant,32", new NumericDim(32).toString());
    assertEquals("D:Symbolic,?", new SymbolicDim("?").toString());
  }

  @Test
  public void testDynamicDimSymbolicAndConcrete() {
    // `DynamicDim` reports as a symbolic dim, mirroring `RaggedDim`'s semantics for the
    // unknown-size axis (https://github.com/wala/ML/issues/545). A `TensorType` whose dims are
    // {DynamicDim, NumericDim(4)} reports `symbolicDims() == 1` because exactly one dim is
    // symbolic (the DynamicDim contributes 1 to the count).
    TensorType t = new TensorType("float32", asList(DynamicDim.INSTANCE, new NumericDim(4)));
    assertEquals(1, t.symbolicDims());
  }

  @Test
  public void testDynamicDimNotEqualsRaggedDim() {
    // Same `Void` payload value, different class — must not collapse to equal.
    assertNotEquals(DynamicDim.INSTANCE, RaggedDim.INSTANCE);
  }

  @Test
  public void testNullPayloadSentinelsHaveDistinctHashCodes() {
    // Both `DynamicDim` and `RaggedDim` carry `Void` payload (always `null`). Without per-class
    // `hashCode` overrides, the inherited `Dimension.hashCode` would hash only the payload and
    // they'd collide in hash buckets — degrading `HashSet<Dimension<?>>` lookups to O(n) when
    // both kinds are stored. Verify the per-class overrides keep them in distinct buckets.
    assertNotEquals(DynamicDim.INSTANCE.hashCode(), RaggedDim.INSTANCE.hashCode());
  }
}
