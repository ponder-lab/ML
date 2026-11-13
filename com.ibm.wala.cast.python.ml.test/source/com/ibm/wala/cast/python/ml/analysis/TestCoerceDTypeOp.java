package com.ibm.wala.cast.python.ml.analysis;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis.CoerceDTypeOp;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * {@link CoerceDTypeOp} keeps an argument's dtype where the argument may be a TensorFlow tensor and
 * casts it where TensorFlow may convert it, by the argument's origins (<a
 * href="https://github.com/wala/ML/issues/1253">wala/ML#1253</a>).
 */
public class TestCoerceDTypeOp {

  /** The fed type. */
  private static final TensorType FED = TensorType.of(DType.FLOAT32, 2);

  /** The fed type cast to the dtype a consumer imposes. */
  private static final TensorType CAST = TensorType.of(DType.INT32, 2);

  /**
   * The types a parameter's node transfer gives an argument of {@link #FED} with some origins,
   * where a consumer imposes {@code int32}.
   *
   * @param origins The argument's origins.
   * @return The parameter's types.
   */
  private static Set<TensorType> coerce(TensorOrigin... origins) {
    TensorVariable rhs = new TensorVariable();
    rhs.state.add(FED);
    rhs.origins.addAll(List.of(origins));
    TensorVariable lhs = new TensorVariable();
    new CoerceDTypeOp(EnumSet.of(DType.INT32), false).evaluate(lhs, rhs);
    return lhs.state;
  }

  /** A TensorFlow tensor keeps its dtype. */
  @Test
  public void testTensorFlowKept() {
    assertEquals(Set.of(FED), coerce(TensorOrigin.TENSORFLOW));
  }

  /** A Keras symbolic tensor keeps its dtype. */
  @Test
  public void testKerasSymbolicKept() {
    assertEquals(Set.of(FED), coerce(TensorOrigin.KERAS_SYMBOLIC));
  }

  /** A NumPy array is cast. */
  @Test
  public void testNumPyCast() {
    assertEquals(Set.of(CAST), coerce(TensorOrigin.NUMPY));
  }

  /** A value that may be a tensor or an array is kept and cast. */
  @Test
  public void testMixedKeptAndCast() {
    assertEquals(Set.of(FED, CAST), coerce(TensorOrigin.TENSORFLOW, TensorOrigin.NUMPY));
  }

  /** A parameter's origin does not say which the argument is, so it is kept and cast. */
  @Test
  public void testParameterKeptAndCast() {
    assertEquals(Set.of(FED, CAST), coerce(TensorOrigin.PARAMETER));
  }

  /** Nor does an annotation's. */
  @Test
  public void testAnnotationKeptAndCast() {
    assertEquals(Set.of(FED, CAST), coerce(TensorOrigin.ANNOTATION));
  }

  /** Nor does the absence of any origin. */
  @Test
  public void testNoOriginKeptAndCast() {
    assertEquals(Set.of(FED, CAST), coerce());
  }

  /**
   * The edge's transfer carries the argument's origins into the parameter's incoming variable; the
   * node's does not.
   */
  @Test
  public void testEdgeCarriesOrigins() {
    for (boolean edge : new boolean[] {true, false}) {
      TensorVariable rhs = new TensorVariable();
      rhs.state.add(FED);
      rhs.origins.add(TensorOrigin.TENSORFLOW);
      TensorVariable lhs = new TensorVariable();
      new CoerceDTypeOp(EnumSet.of(DType.INT32), edge).evaluate(lhs, rhs);
      assertEquals(
          edge ? EnumSet.of(TensorOrigin.TENSORFLOW) : EnumSet.noneOf(TensorOrigin.class),
          lhs.origins);
    }
  }

  /** Two operators are equal exactly when they impose the same dtypes in the same form. */
  @Test
  public void testEquality() {
    CoerceDTypeOp op = new CoerceDTypeOp(EnumSet.of(DType.INT32), true);
    CoerceDTypeOp same = new CoerceDTypeOp(EnumSet.of(DType.INT32), true);
    assertEquals(op, same);
    assertEquals(op.hashCode(), same.hashCode());
    assertNotEquals(op, new CoerceDTypeOp(EnumSet.of(DType.INT32), false));
    assertNotEquals(op, new CoerceDTypeOp(EnumSet.of(DType.FLOAT32), true));
    assertNotEquals(op, new Object());
  }
}
