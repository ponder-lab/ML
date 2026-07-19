package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_0_RAGGED_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_0_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_10_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_RAGGED_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_RAGGED_RAGGED_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_RAGGED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_RAGGED_RAGGED_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_RAGGED_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_RAGGED_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of ragged-tensor shape and dtype inference, carved from the {@code TestTensorflow2Model}
 * monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestRaggedTensors extends AbstractTensorTest {

  @Test
  public void testRaggedFromRowStartsFull()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_starts_full.py",
        "test_ragged_from_row_starts_full",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_5_RAGGED_INT32),
            3, Set.of(TENSOR_5_RAGGED_INT32),
            4, Set.of(TENSOR_5_RAGGED_INT32),
            5, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowStarts}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>). When {@code row_starts} has a
   * non-{@code NumericDim} first dim — here, a {@code DynamicDim} from {@code tf.keras.Input}'s
   * symbolic batch axis — the generator emits {@code DynamicDim.INSTANCE} for the inferred nrows,
   * yielding a {@code (DynamicDim, RaggedDim)} shape rather than emitting raw {@code null}.
   */
  @Test
  public void testRaggedFromRowStartsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_starts_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_lengths.py",
        "test_ragged_from_row_lengths",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromRowLimits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_limits.py",
        "test_ragged_from_row_limits",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromValueRowIds()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_value_rowids.py",
        "test_ragged_from_value_rowids",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_4_RAGGED_INT32),
            3, Set.of(TENSOR_4_RAGGED_INT32),
            4, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowLengths}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>).
   */
  @Test
  public void testRaggedFromRowLengthsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_lengths_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowLimits}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>).
   */
  @Test
  public void testRaggedFromRowLimitsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_limits_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  /**
   * Coverage guard for the {@code DynamicDim} fallback branch in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromRowSplits}'s nrows inference (<a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>).
   */
  @Test
  public void testRaggedFromRowSplitsDynamicRowDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_splits_dynamic.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_nested.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_RAGGED_INT32),
            5, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsAdditional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_more.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_5_RAGGED_INT32),
            3, Set.of(TENSOR_4_RAGGED_INT32),
            4, Set.of(TENSOR_4_RAGGED_INT32),
            5, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_INT32),
            5, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsV2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_keyword_args_v2.py",
        "test_keywords",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_INT32),
            5, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedKeywordArgsMixed()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_mixed_new.py",
        "test_ragged_mixed_args_new",
        3,
        3,
        Map.of(
            // rt1: positional values, keyword value_rowids. inferred nrows=3.
            2,
            Set.of(TENSOR_3_RAGGED_INT32),

            // rt2: positional values, keyword value_rowids, keyword nrows=5.
            3,
            Set.of(TENSOR_5_RAGGED_INT32),

            // rt3: positional values, positional value_rowids, keyword nrows=3.
            4,
            Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNrowsPositional()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_nrows_positional.py",
        "test_ragged_nrows_positional",
        1,
        1,
        Map.of(
            // rt: positional values, positional value_rowids, positional nrows=3.
            2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  /**
   * Regression test for the wala/ML#518 throw path in {@link
   * com.ibm.wala.cast.python.ml.client.RaggedFromNestedValueRowIds#getShapes}'s {@code
   * nested_nrows} arg-collection loop: when the arg contains a non-numeric string {@link
   * com.ibm.wala.ipa.callgraph.propagation.ConstantKey}, the {@code Long.parseLong((String) val)}
   * site catches {@link NumberFormatException} and rethrows as {@link IllegalStateException} (with
   * the original NFE as {@code cause}). The test exercises that branch by passing {@code
   * nested_nrows=["abc"]} in the Python fixture, and {@code assertThrows} captures the rethrow so
   * the test can assert on the cause—tighter than a bare {@code @Test(expected = …)}, which would
   * pass on any {@code IllegalStateException} raised during analysis regardless of origin. Closes
   * part of <a href="https://github.com/wala/ML/issues/520">wala/ML#520</a> (the {@code
   * RaggedFromNestedValueRowIds} portion).
   */
  @Test
  public void testRaggedNrowsNonNumeric() {
    IllegalStateException ise =
        assertThrows(
            IllegalStateException.class,
            () -> test("tf2_test_ragged_nrows_non_numeric.py", "f", 1, 1, Map.of()));
    assertTrue(
        "Expected cause to be NumberFormatException; got " + ise.getCause(),
        ise.getCause() instanceof NumberFormatException);
  }

  @Test
  public void testRaggedConstant() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant5() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant5.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRaggedConstant7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedConstant8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant9() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant9.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant10() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_1_FLOAT32)));
  }

  @Test
  public void testRaggedConstant11() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant12() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_FLOAT32)));
  }

  @Test
  public void testRaggedConstant13() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant13.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_0_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedConstant14() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant14.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_0_RAGGED_3_FLOAT32)));
  }

  @Test
  public void testRaggedConstant15() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant15.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_INT32)));
  }

  @Test
  public void testRaggedConstant16() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant16.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_INT32)));
  }

  /**
   * Test non-uniform inner dimensions.
   *
   * <p>TODO: Remove expected assertion error once https://github.com/wala/ML/issues/350 is fixed.
   */
  @Test(expected = AssertionError.class)
  public void testRaggedConstant17() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant17.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_3_INT32)));
  }

  /** This one works because the inner dimensions are uniform. */
  @Test
  public void testRaggedConstant18() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_constant18.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_RAGGED_2_2_INT32)));
  }

  @Test
  public void testRaggedConstantKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_keyword.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_2_RAGGED_INT32),
            3, Set.of(TENSOR_2_RAGGED_INT32),
            4, Set.of(TENSOR_2_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromRowSplits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_row_splits.py",
        "test_ragged_from_row_splits",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_5_RAGGED_INT32),
            3, Set.of(TENSOR_5_RAGGED_INT32),
            4, Set.of(TENSOR_5_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_10_INT32)));
  }

  @Test
  public void testRaggedRange2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_10_INT32)));
  }

  @Test
  public void testRaggedRange3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRange7() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_RAGGED_INT32)));
  }

  /**
   * Canonical case for <a href="https://github.com/wala/ML/issues/546">wala/ML#546</a>: {@code
   * tf.ragged.range(3, 18, 3)} — all three scalar args are compile-time literals, so the inner
   * length is statically computable: {@code ceil((18 - 3) / 3) = 5}. The analyzer pins {@code (1,
   * 5)} instead of {@code (1, ragged)}.
   */
  @Test
  public void testRaggedRange8() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_5_INT32)));
  }

  /**
   * Multi-length fallback case for <a href="https://github.com/wala/ML/issues/546">wala/ML#546</a>:
   * {@code start} resolves to two literal values via an if/else, so the cross-product yields
   * lengths {@code {10, 8}}. {@code computeStaticInnerLength} returns {@code null} and the inner
   * dim falls back to {@code RaggedDim}.
   */
  @Test
  public void testRaggedRange9() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_ragged_range9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_RAGGED_INT32)));
  }

  @Test
  public void testRaggedRangeKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_range_keyword.py",
        "test",
        5,
        5,
        Map.of(
            2, Set.of(TENSOR_1_5_INT32),
            3, Set.of(TENSOR_1_5_INT32),
            4, Set.of(TENSOR_1_5_INT32),
            5, Set.of(TENSOR_1_5_INT32),
            6, Set.of(TENSOR_1_5_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengths()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths.py",
        "test_ragged_from_nested_row_lengths",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengthsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_keyword.py",
        "test_ragged_from_nested_row_lengths_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowLengthsKeyword2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_keyword2.py",
        "test_ragged_from_nested_row_lengths_keyword2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsPositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_positional.py",
        "test_ragged_from_nested_row_splits_positional",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_keyword.py",
        "test_ragged_from_nested_row_splits_keyword",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedFromNestedRowSplitsMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_splits_mixed.py",
        "test_ragged_from_nested_row_splits_mixed",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositionalLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsPositionalTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_positional_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_RAGGED_RAGGED_STRING)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeywordLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsKeywordTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_keyword_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixedLists()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed_lists.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedNestedValueRowidsMixedTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_nested_value_rowids_mixed_tuples.py",
        "check_rt",
        1,
        1,
        Map.of(2, Set.of(TENSOR_1_RAGGED_RAGGED_FLOAT32)));
  }

  @Test
  public void testRaggedFromNestedValueRowIdsComplete()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // check_case_1: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_1",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_2: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_2",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_3: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_3",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_4: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_4",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));

    // check_case_5: [2, None, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_5",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_RAGGED_INT32)));

    // check_case_6: [2, None], float32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_6",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_RAGGED_FLOAT32)));

    // check_case_7: [4, None], int32
    test(
        "tf2_test_ragged_nested_value_rowids_complete.py",
        "check_case_7",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_RAGGED_INT32)));
  }

  /**
   * Captured-gap regression for the {@code RaggedFromNested} shape floor (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>): {@code
   * tf.RaggedTensor.from_nested_row_lengths} with an opaque (unresolvable) {@code
   * nested_row_lengths} floors the shape to ⊤ (unknown) rather than aborting the whole analysis
   * with "Could not calculate shapes". The dtype still resolves to {@code int32} from the {@code
   * flat_values}.
   */
  @Test
  public void testRaggedFromNestedRowLengthsUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_row_lengths_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedFromNestedValueRowIds} shape and dtype floors (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>): {@code
   * tf.RaggedTensor.from_nested_value_rowids} with opaque (unresolvable) {@code flat_values} and
   * {@code nested_value_rowids} floors both the shape and the dtype to ⊤ rather than aborting with
   * "Could not calculate shapes" / "Could not determine dtypes".
   */
  @Test
  public void testRaggedFromNestedValueRowIdsUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_from_nested_value_rowids_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedConstant} shape and dtype floors (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>): {@code tf.ragged.constant} whose
   * {@code pylist} comes from an unmodeled {@code json.loads} &mdash; so its points-to set is empty
   * even though the values are inline &mdash; floors both the shape and the dtype to ⊤ rather than
   * aborting with "Empty points-to set".
   *
   * <p>TODO: The runtime tensor is {@code (2, None)} {@code int32} (asserted in the fixture); the
   * static result floors both axes to ⊤ because {@code json.loads} is unmodeled. This is a modeling
   * gap, not a content-dependent (opaque) value, tracked by <a
   * href="https://github.com/wala/ML/issues/536">wala/ML#536</a> (model {@code json.loads} for
   * compile-time-constant string inputs). ⊤ is the correct floor until then; it is not on the
   * input-signature eval path.
   */
  @Test
  public void testRaggedConstantUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedConstant} shape and dtype floors (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>) along the structural-walk path: a
   * {@code pylist} whose outer list is resolvable but whose first element is an {@code np.ndarray}
   * (neither a {@code list} nor a {@code tuple}) floors both the shape and the dtype to ⊤ rather
   * than aborting the whole analysis with "Expected a list or tuple". Complements {@link
   * #testRaggedConstantUnresolvable()}, which exercises the empty-points-to-set floor.
   *
   * <p>TODO: The runtime tensor is {@code (2, None)} {@code int32} (asserted in the fixture). The
   * dtype floor is inherent until numpy dtype-promotion is modeled (<a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>): an {@code np.ndarray} element's
   * dtype is soundly ⊤ (numpy promotes {@code int} to {@code int64}, not {@code int32}), so the
   * union floors to ⊤. The shape floor reflects the unmodeled ragged rank over a tensor row;
   * delegating the element to its producer generator would recover the shape (<a
   * href="https://github.com/wala/ML/issues/652">wala/ML#652</a>).
   */
  @Test
  public void testRaggedConstantUnresolvableElement()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_unresolvable_element.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * Captured-gap regression for the {@code RaggedConstant} shape floor (<a
   * href="https://github.com/wala/ML/issues/612">wala/ML#612</a>) along the depth-walk path, and a
   * precision guard for the dtype. A {@code pylist} whose first row is a resolvable scalar list but
   * whose second row is an {@code np.ndarray} trips the structural floor in {@code
   * getMaximumDepthOfScalars} (a different site than {@link
   * #testRaggedConstantUnresolvableElement()}, which trips {@code containsScalars}), flooring the
   * shape to ⊤. The dtype is still resolved to {@code int32}, because the leading scalar row lets
   * {@code getDefaultDTypes} confirm scalars before the {@code np.ndarray} element &mdash; so the
   * floor is not the all-⊤ result of {@link #testRaggedConstantUnresolvableElement()}, where the
   * {@code np.ndarray} element precedes any confirmable scalar.
   *
   * <p>TODO: The runtime shape is {@code (2, None)} (asserted in the fixture); the static shape
   * floors to ⊤ over the {@code np.ndarray} row (the unmodeled ragged rank over a tensor element).
   * Delegating the element to its producer generator would recover the shape (<a
   * href="https://github.com/wala/ML/issues/652">wala/ML#652</a>).
   *
   * <p>The {@code numpy.array} producer registration (wala/ML#625) adds the ⊤ {@code int64} member:
   * the {@code np.ndarray} row's own type (numpy's integer default) joins the union through the
   * flow-insensitive points-to substrate, alongside the ragged result's runtime {@code int32}.
   */
  @Test
  public void testRaggedConstantUnresolvableDepth()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ragged_constant_unresolvable_depth.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE, new TensorType(INT_64, null))));
  }
}
