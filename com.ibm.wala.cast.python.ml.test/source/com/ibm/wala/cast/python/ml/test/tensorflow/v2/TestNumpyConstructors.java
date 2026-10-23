package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the NumPy array constructors and reductions a data-preparation prefix builds its
 * arrays with carry their NumPy dtype and the shape the analysis can compute (<a
 * href="https://github.com/wala/ML/issues/1008">wala/ML#1008</a>). An extent written with {@code
 * int(...)} or computed from one is a fixed runtime integer the analysis may not fold, so it reads
 * as unresolved; the dtypes are exact.
 */
public class TestNumpyConstructors extends AbstractTensorTest {

  private static final String FILE = "tf2_test_numpy_constructors_1008.py";

  private static final Dimension<?> U = UnresolvedDim.INSTANCE;

  private static Dimension<?> n(int value) {
    return new NumericDim(value);
  }

  private static TensorType t(String dtype, Dimension<?>... dims) {
    return new TensorType(dtype, asList(dims));
  }

  private static final String F64 = FLOAT_64;

  private static final String F32 = FLOAT_32;

  private static final String I64 = INT_64;

  private static final String I32 = INT_32;

  /**
   * {@code np.append(...)} over two {@code np.arange} results, the flattened and joined arrays:
   * {@code float64}, since one bound is a quotient that may be a float.
   */
  @Test
  public void testAppend() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_append", 1, 1, Map.of(2, Set.of(t(F64, U))));
  }

  /** {@code np.outer(a, b)}, rank 2 whatever the inputs' ranks, with their promoted dtype. */
  @Test
  public void testOuter() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_outer", 1, 1, Map.of(2, Set.of(t(F64, U, U))));
  }

  /**
   * {@code np.concatenate} of two outer products, which no longer reads an unknown dtype once its
   * inputs are typed.
   */
  @Test
  public void testConcatenate() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_concatenate", 1, 1, Map.of(2, Set.of(t(F64, U, U))));
  }

  /**
   * {@code np.stack(..., axis=2)}, the arrays' count inserted at the axis, with their promoted
   * dtype.
   */
  @Test
  public void testStack() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_stack", 1, 1, Map.of(2, Set.of(t(F64, U, U, n(2)))));
  }

  /** An integer-array subscript {@code X[ii, :]} of the stacked array keeps its {@code float64}. */
  @Test
  public void testFancyIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_fancy", 1, 1, Map.of(2, Set.of(t(F64, U, U, n(2)))));
  }

  /** The array after the per-channel normalization loop is still {@code float64}. */
  @Test
  public void testNormalized() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_normalized", 1, 1, Map.of(2, Set.of(t(F64, U, U, n(2)))));
  }

  /** {@code np.random.choice(2, shape)} draws {@code int64}, with the shape {@code size} gives. */
  @Test
  public void testChoice() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_choice", 1, 1, Map.of(2, Set.of(t(I64, U, U, n(2)))));
  }

  /**
   * {@code np.random.choice(3)} without a {@code size} draws a Python {@code int}, encoded as
   * {@code int32}.
   */
  @Test
  public void testChoiceScalar() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_choice_scalar", 1, 1, Map.of(2, Set.of(t(I32))));
  }

  /** {@code np.repeat(-1, nt)} is a rank-1 {@code int64} array of {@code nt} elements. */
  @Test
  public void testRepeat() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_repeat", 1, 1, Map.of(2, Set.of(t(I64, n(8)))));
  }

  /** {@code np.diag} of a rank-1 array is a square matrix in its dtype. */
  @Test
  public void testDiag() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_diag", 1, 1, Map.of(2, Set.of(t(I64, n(8), n(8)))));
  }

  /**
   * {@code np.random.multivariate_normal(mean, cov, 4)} is {@code float64}, {@code size} with the
   * mean's length appended.
   */
  @Test
  public void testMultivariateNormal()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_multivariate_normal", 1, 1, Map.of(2, Set.of(t(F64, n(4), n(8)))));
  }

  /**
   * An array rescaled by its {@code np.mean} and {@code np.std} keeps its shape and {@code
   * float64}.
   */
  @Test
  public void testRescaled() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_rescaled", 1, 1, Map.of(2, Set.of(t(F64, n(8), n(8)))));
  }

  /**
   * {@code np.mean} of an {@code int64} array without an {@code axis} is a {@code float64} scalar.
   */
  @Test
  public void testMean() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean", 1, 1, Map.of(2, Set.of(t(F64))));
  }

  /** {@code np.mean(c, axis=0)} removes the axis. */
  @Test
  public void testMeanAxis() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_axis", 1, 1, Map.of(2, Set.of(t(F64, n(8)))));
  }

  /** {@code np.mean(b1, axis=1, keepdims=True)} keeps the axis as {@code 1}. */
  @Test
  public void testMeanKeepDims() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_keepdims", 1, 1, Map.of(2, Set.of(t(F64, n(4), n(1)))));
  }

  /**
   * {@code np.mean} with a {@code dtype} argument the analysis cannot read is a scalar of unknown
   * dtype, not the {@code float64} its {@code int64} input would average in (wala/ML#865).
   */
  @Test
  public void testMeanUnreadableDType()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_dtype", 1, 1, Map.of(2, Set.of(t(UNKNOWN))));
  }

  /**
   * {@code np.mean} over a value whose {@code int64} dtype only the sidecar supplies, with a {@code
   * dtype} argument the analysis cannot read: the result's dtype is unknown (wala/ML#865). The
   * input has no points-to members, so the generator's own read is unknown and the type feed is
   * what would map the input's dataflow dtype; with a {@code dtype} passed, it declares none.
   */
  @Test
  public void testMeanUnreadableDTypeOfDataflowInput()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        new String[] {"sidecar_mean_proj/loader.py", "sidecar_mean_proj/driver.py"},
        "driver.py",
        "consume_mean",
        "sidecar_mean_proj",
        1,
        1,
        Map.of(2, Set.of(t(UNKNOWN))));
  }

  /** {@code np.std} of a {@code float32} array is a {@code float32} scalar. */
  @Test
  public void testStd() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_std", 1, 1, Map.of(2, Set.of(t(F32))));
  }

  /**
   * {@code X[jj,]}, a one-element tuple index, selects as {@code X[jj]} does: the index's shape
   * replaces the leading axis rather than peeling it.
   */
  @Test
  public void testChoiceIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_choice_index", 1, 1, Map.of(2, Set.of(t(F64, n(4), U, n(2)))));
  }

  /**
   * {@code X[kk]} with {@code kk} from {@code np.random.permutation} replaces the leading axis with
   * the index's shape.
   */
  @Test
  public void testPermutationIndex() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_permutation_index", 1, 1, Map.of(2, Set.of(t(F64, n(10), U, n(2)))));
  }

  /** {@code np.diag(v, 1)} of a length-3 vector is a 4-by-4 matrix. */
  @Test
  public void testDiagOffset() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_diag_offset", 1, 1, Map.of(2, Set.of(t(F64, n(4), n(4)))));
  }

  /** {@code np.diag(v, -2)} of a length-3 vector is a 5-by-5 matrix. */
  @Test
  public void testDiagNegativeOffset()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_diag_negative_offset", 1, 1, Map.of(2, Set.of(t(F64, n(5), n(5)))));
  }

  /** {@code np.diag} of a 3-by-5 matrix is its main diagonal, of length 3. */
  @Test
  public void testDiagOfMatrix() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_diag_of_matrix", 1, 1, Map.of(2, Set.of(t(F64, n(3)))));
  }

  /** {@code np.diag(a, 1)} of a matrix is rank 1, of a length this generator leaves unresolved. */
  @Test
  public void testDiagOfMatrixOffset()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_diag_of_matrix_offset", 1, 1, Map.of(2, Set.of(t(F64, U))));
  }

  /**
   * {@code np.diag(v, k)} with an offset the analysis cannot read is a square matrix of unresolved
   * side.
   */
  @Test
  public void testDiagUnresolvedOffset()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_diag_unresolved_offset", 1, 1, Map.of(2, Set.of(t(F64, U, U))));
  }

  /** {@code np.arange(5)}, the one-argument form, is {@code int64} of length 5. */
  @Test
  public void testArangeStop() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arange_stop", 1, 1, Map.of(2, Set.of(t(I64, n(5)))));
  }

  /** {@code np.arange(0, 1, 0.25)}: a float step makes the result {@code float64}. */
  @Test
  public void testArangeFloatStep() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arange_float_step", 1, 1, Map.of(2, Set.of(t(F64, U))));
  }

  /** {@code np.arange(k * 2)} with {@code k = int(3)}: a product of integers is an integer. */
  @Test
  public void testArangeProduct() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arange_product", 1, 1, Map.of(2, Set.of(t(I64, n(6)))));
  }

  /** {@code np.arange(-k, 0)}: a negated integer is an integer. */
  @Test
  public void testArangeNegated() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arange_negated", 1, 1, Map.of(2, Set.of(t(I64, U))));
  }

  /** {@code np.arange(0, g)} reading a module-level float {@code g} is {@code float64}. */
  @Test
  public void testArangeGlobalFloat() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arange_global_float", 1, 1, Map.of(2, Set.of(t(F64, U))));
  }

  /**
   * {@code np.arange(n)} with a bound whose numeric kind the analysis cannot read has an unknown
   * dtype.
   */
  @Test
  public void testArangeUnclassified()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_arange_unclassified", 1, 1, Map.of(2, Set.of(t(UNKNOWN, U))));
  }

  /** {@code np.mean(c, axis=None)} reduces every axis, as no {@code axis} does. */
  @Test
  public void testMeanAxisNone() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_axis_none", 1, 1, Map.of(2, Set.of(t(F64))));
  }

  /**
   * {@code np.mean(c, axis=(0, 1))}: a tuple of axes is not read, so the shape is unknown and the
   * dtype {@code float64}.
   */
  @Test
  public void testMeanAxisTuple() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_axis_tuple", 1, 1, Map.of(2, Set.of(new TensorType(F64, null))));
  }

  /** {@code np.mean(c, keepdims=True)} keeps every axis as {@code 1}. */
  @Test
  public void testMeanKeepDimsAll() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_keepdims_all", 1, 1, Map.of(2, Set.of(t(F64, n(1), n(1)))));
  }

  /** {@code np.mean(b1, axis=-1)} removes the last axis. */
  @Test
  public void testMeanNegativeAxis() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_negative_axis", 1, 1, Map.of(2, Set.of(t(F64, n(4)))));
  }

  /** {@code np.mean} of a {@code float32} array keeps {@code float32}. */
  @Test
  public void testMeanFloat32() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_float32", 1, 1, Map.of(2, Set.of(t(F32, n(3)))));
  }

  /** {@code np.mean} of a {@code bool} array averages in {@code float64}. */
  @Test
  public void testMeanBool() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_mean_bool", 1, 1, Map.of(2, Set.of(t(F64))));
  }

  /**
   * {@code np.random.choice([1, 2, 3], 2)}: a Python sequence's dtype is NumPy's promotion over its
   * elements, which the generator leaves unknown.
   */
  @Test
  public void testChoiceList() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_choice_list", 1, 1, Map.of(2, Set.of(t(UNKNOWN, n(2)))));
  }

  /**
   * {@code np.random.multivariate_normal(mean, cov)} without a {@code size} is one vector of the
   * mean's length.
   */
  @Test
  public void testMultivariateNormalVector()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_multivariate_normal_vector", 1, 1, Map.of(2, Set.of(t(F64, n(8)))));
  }

  /**
   * {@code np.append(a, b, axis=0)} is a concatenation whose shape this generator does not compute.
   */
  @Test
  public void testAppendAxis() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_append_axis", 1, 1, Map.of(2, Set.of(new TensorType(I64, null))));
  }

  /**
   * {@code np.repeat(a, 2, axis=0)} keeps the input's rank, which this generator does not compute.
   */
  @Test
  public void testRepeatAxis() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_repeat_axis", 1, 1, Map.of(2, Set.of(new TensorType(I64, null))));
  }

  /**
   * {@code np.repeat(m, r)} with a count the analysis cannot read is rank 1 of unresolved length.
   */
  @Test
  public void testRepeatUnresolved() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_repeat_unresolved", 1, 1, Map.of(2, Set.of(t(I64, U))));
  }
}
