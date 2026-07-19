package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.COMPLEX_128;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.COMPLEX_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_0_0_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_5_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_30_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_7_5_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_32_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_NONE_NONE_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNRESOLVED_UNRESOLVED_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.UNKNOWN;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.client.Constant;
import com.ibm.wala.cast.python.ml.client.Linspace;
import com.ibm.wala.cast.python.ml.client.NpArray;
import com.ibm.wala.cast.python.ml.client.NpOnes;
import com.ibm.wala.cast.python.ml.client.NpZeros;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of tensor-constructor shape and dtype inference ({@code tf.ones}/{@code zeros}/{@code
 * eye}/{@code range}/{@code fill}/{@code one_hot}/{@code constant}/{@code convert_to_tensor}/{@code
 * tf.keras.Input}/{@code np.*}), carved from the {@code TestTensorflow2Model} monolith
 * (wala/ML#635); the assertions are verbatim.
 */
public class TestConstructors extends AbstractTensorTest {

  /**
   * Isolating regression guard for the {@code numpy → tf.constant} dtype/shape bridge (<a
   * href="https://github.com/wala/ML/issues/539">wala/ML#539</a>), surfaced from {@link
   * #testMultilayerPerceptron}. {@code consume(x)} where {@code x = tf.constant(np.ones((2, 3),
   * dtype=np.float32))} infers {@code (2, 3) float32}: {@code np.ones} is now modeled (see {@link
   * NpOnes}), so {@link Constant} recovers the numpy producer's shape and dtype through the value
   * argument rather than falling back to {@code ⊤ shape / UNKNOWN dtype}.
   */
  @Test
  public void testConstantFromNumpy()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_constant_from_numpy.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testConstantFromNumpy} for the {@code np.zeros → tf.constant} bridge:
   * {@code consume(x)} where {@code x = tf.constant(np.zeros((2, 3), dtype=np.int32))} infers
   * {@code (2, 3) int32}. Exercises {@link NpZeros} via the {@code createManualGenerator} recovery
   * path (symmetric to the {@code np.ones} bridge in {@link #testMultilayerPerceptron}, but with a
   * non-float dtype). Positive regression guard for wala/ML#539.
   */
  @Test
  public void testConstantFromNpZeros()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_constant_from_np_zeros.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testZerosLikeTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zeros_like_tensor.py", "func2", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testInputWithBatchSize()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t1 = TensorType.of(FLOAT_32, 16, 32);
    TensorType t2 = TensorType.of(FLOAT_32, 5, 10, 10);
    TensorType t3 = new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(5)));

    test(
        "tf2_test_input_batch_size.py",
        "check_input",
        3,
        3,
        Map.of(2, Set.of(t1), 3, Set.of(t2), 4, Set.of(t3)));
  }

  @Test
  public void testInputInt32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t1 = TensorType.of(INT_32, 32, 10);
    TensorType t2 = TensorType.of(INT_32, 8, 5, 5);

    test("tf2_test_input_int32.py", "check_input", 2, 2, Map.of(2, Set.of(t1), 3, Set.of(t2)));
  }

  @Test
  public void testInputMixedArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // input1: shape=(32, 10), dtype=int32
    TensorType t1 = TensorType.of(INT_32, 32, 10);
    // input2: shape=(16, 5, 5), dtype=float32
    TensorType t2 = TensorType.of(FLOAT_32, 16, 5, 5);
    // input3: shape=(None, 20), dtype=int32
    TensorType t3 = new TensorType(INT_32, asList(DynamicDim.INSTANCE, new NumericDim(20)));

    test(
        "tf2_test_input_mixed_args.py",
        "check_input",
        3,
        3,
        Map.of(2, Set.of(t1), 3, Set.of(t2), 4, Set.of(t3)));
  }

  /**
   * The `tensor` parameter wraps an existing tensor, so the result takes that tensor's shape and
   * dtype verbatim, with no batch dimension prepended (wala/ML#617).
   */
  @Test
  public void testInputTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(FLOAT_32, 2, 3);

    test("tf2_test_input_tensor_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
    test("tf2_test_input_tensor_pos.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * A ragged `Input` has the same tracked shape and dtype as a dense one, so the `ragged` parameter
   * is modeled by treating it as transparent to shape and dtype inference (wala/ML#617).
   */
  @Test
  public void testInputRagged()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(10)));

    test("tf2_test_input_ragged_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * The `type_spec` parameter supplies the full type, so the result takes the spec's shape and
   * dtype verbatim, with no batch dimension prepended (wala/ML#617).
   */
  @Test
  public void testInputTypeSpec()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = new TensorType(INT_32, asList(DynamicDim.INSTANCE, new NumericDim(4)));

    test("tf2_test_input_type_spec_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * A sparse `Input` has the same logical shape and dtype as a dense one, so the `sparse` parameter
   * is modeled by treating it as transparent to shape and dtype inference (wala/ML#616).
   */
  @Test
  public void testInputSparse()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = new TensorType(FLOAT_32, asList(DynamicDim.INSTANCE, new NumericDim(10)));

    test("tf2_test_input_sparse_kw.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
    test("tf2_test_input_sparse_pos.py", "check_input", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/598">wala/ML#598</a>: a bare
   * {@code numpy.array} value propagates its {@code TensorType} to a callee parameter. The issue's
   * reproducer; {@code f}'s parameter types to {@code (3,)} {@code float64}: {@code NpArray} infers
   * the list-literal shape, and the dtype from numpy's promotion of the Python float literals (<a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>). The runtime dtype is {@code
   * float64} (numpy promotes Python {@code float} to {@code float64}, not the {@code float32}
   * TF-literal convention).
   */
  @Test
  public void testNpArrayBareParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nparray_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_64, 3))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the integer-promotion path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: a bare {@code numpy.array} of
   * Python ints types to {@code (3,)} {@code int64}, because numpy promotes Python {@code int} to
   * {@code int64} (not the {@code int32} TF-literal convention).
   */
  @Test
  public void testNpArrayIntParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nparray_int_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 3))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the boolean path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: an all-boolean literal array
   * types to {@code (2,)} {@code bool}.
   */
  @Test
  public void testNpArrayBoolParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_nparray_bool_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(BOOL, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the string path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: an all-string literal array types
   * to {@code (2,)} {@code string} (a string leaf subsumes the array in numpy's promotion).
   */
  @Test
  public void testNpArrayStringParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_string_param.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(STRING, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the nested-literal promotion path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: a nested literal mixing ints and
   * a float promotes to {@code (2, 2)} {@code float64}, exercising the walk's descent through
   * nested lists and the float-over-int promotion.
   */
  @Test
  public void testNpArrayNestedParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_nested_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 2, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the complex path of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: an all-complex literal array
   * types to {@code (2,)} {@code complex128} (numpy promotes Python {@code complex} to {@code
   * complex128}).
   */
  @Test
  public void testNpArrayComplexParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_complex_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(COMPLEX_128, 2))));
  }

  /**
   * Companion to {@link #testNpArrayBareParam()} for the non-literal-source floor of <a
   * href="https://github.com/wala/ML/issues/626">wala/ML#626</a>: when {@code x} is itself an
   * {@code np.ndarray} rather than a Python literal, numpy preserves the source's dtype, which the
   * walk does not model, so the dtype floors to ⊤. The nested-array {@code (2,)} shape now
   * propagates through the outer {@code np.array} via the producer registration (wala/ML#625); the
   * dtype residue stays with wala/ML#626.
   */
  @Test
  public void testNpArrayArraySource()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_array_source.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(new TensorType(UNKNOWN, asList(new NumericDim(2))))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/598">wala/ML#598</a>: the
   * {@code tf.constant}-wrapped {@code numpy.array} form also propagates to the callee parameter.
   * It currently types to {@code ⊤} unknown, coarser than the bare form's {@code (3,)} because
   * {@code tf.constant} drops the array shape.
   *
   * <p>The {@code numpy.array} producer registration (wala/ML#625) delivers the concrete type: the
   * wrapped array's {@code (3,)} shape survives {@code tf.constant}, and the dtype is numpy's
   * {@code float64} default for float literals.
   */
  @Test
  public void testNpArrayWrappedParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_nparray_wrapped_param.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_64, 3))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/637">wala/ML#637</a>: a {@code
   * tf.constant} with an explicit {@code dtype=tf.complex64} types to {@code complex64} (now
   * modeled by the {@code DType} enum) rather than ⊤, so {@code consume}'s parameter is {@code
   * (2,)} complex64.
   */
  @Test
  public void testConstantComplex64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_complex64.py", "consume", 1, 1, Map.of(2, Set.of(TensorType.of(COMPLEX_64, 2))));
  }

  @Test
  public void testRange()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression test for wala/ML#451 (reproducer 3): each element {@code i} from {@code for i in
   * tf.range(...)} is a 0-D scalar int32 tensor. The receiving function {@code f}'s parameter must
   * be tensor-classified — never primitive-co-classified downstream. Differs from {@link
   * #testRange()} in being a stripped-down fixture that mirrors the issue body verbatim (no
   * intermediate {@code start}/{@code limit}/{@code delta} variables, no Python {@code assert}s
   * intervening between the {@code tf.range} call and the {@code for}-loop). Pre-fix, certain binop
   * sources at iteration time could shadow the element's tensor classification with spurious
   * primitive entries (the same {@link ElementWiseOperation} over-dispatch fixed by the binop
   * operand-tensor gate); the cleaner fixture here exercises that path directly.
   */
  @Test
  public void testRangeIterationElementType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_iter.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range2.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range3.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testRange4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_range4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRange5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  /**
   * Regression test for wala/ML#492. When an explicit {@code dtype=} keyword is supplied to {@code
   * tf.range}, the analyzer honors it instead of defaulting to {@code int32}. {@code tf.range(0, 5,
   * dtype=tf.float32)} infers {@code float32} via {@link Range#getDTypes}'s dtype-arg dispatch.
   */
  @Test
  public void testRangeDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testRangeDType()} — explicit {@code dtype=tf.int64} should be honored (not
   * collapsed to the {@code int32} default). Pinpoints that the dtype-arg path resolves arbitrary
   * {@link com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType} values, not just {@code
   * float32}.
   */
  @Test
  public void testRangeDTypeInt64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_dtype_int64.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT64)));
  }

  /**
   * Companion to {@link #testRangeDType()} — 1-positional form {@code tf.range(limit, dtype=...)}.
   * Verifies that the dtype-arg dispatch fires regardless of how many positional args are present
   * (the {@link Range} class's call-string-based shape resolution is independent of dtype lookup).
   */
  @Test
  public void testRangeDType1Arg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_dtype_1arg.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Regression test for the implicit-dtype path of {@link Range}: when no explicit {@code dtype} is
   * supplied but the {@code start}/{@code limit}/{@code delta} arguments are {@code float}-typed,
   * TF promotes the output to {@code float32} at runtime. {@link Range#getDefaultDTypes} now
   * derives its result from the numeric argument types, matching that promotion. Fix for <a
   * href="https://github.com/wala/ML/issues/492">wala/ML#492</a>.
   */
  @Test
  public void testRangeFloatArgs()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_float_args.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testConvertToTensor2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  /**
   * Pins the {@code tf.range} arm of the dimension-provenance split (wala/ML#721): a
   * configuration-sourced limit (an environment read the analysis cannot fold) types the rank-1
   * length as {@link UnresolvedDim} — a fixed runtime size of unknown value — not {@link
   * DynamicDim}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testRangeUnresolved()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_range_unresolved.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(UnresolvedDim.INSTANCE)))));
  }

  @Test
  public void testConvertToTensor4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testConvertToTensor6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testConvertToTensor10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  @Test
  public void testConvertToTensor12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_convert_to_tensor12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testOneHot()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot7.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot8.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot9.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot10.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testOneHot11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot11.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot12.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot13.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_3_INT32)));
  }

  @Test
  public void testOneHot14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot14.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot15.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot16.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_FLOAT32)));
  }

  @Test
  public void testOneHot17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot17.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testOneHot18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_one_hot18.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_5_3_FLOAT32)));
  }

  @Test
  public void testOneHot19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Fixed by handling `CONSTANT_OP_CONSTANT`.
    test("tf2_test_one_hot19.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
    test("tf2_test_one_hot19.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_5_3_FLOAT32)));
  }

  @Test
  public void testOneHot20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_one_hot20.py",
        "test",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_3_3_FLOAT32),
            3, Set.of(TENSOR_3_3_INT32),
            4, Set.of(TENSOR_3_3_FLOAT32),
            5, Set.of(TENSOR_3_3_FLOAT32)));
  }

  @Test
  public void testEye()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testEye2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testEye3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testEye4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_FLOAT32)));
  }

  @Test
  public void testEye5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_3_FLOAT32)));
  }

  @Test
  public void testEye6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_2_2_3_FLOAT32)));
  }

  @Test
  public void testFillKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_INT32)));
  }

  @Test
  public void testFillMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  @Test
  public void testRangeStartLimitKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_limit_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_4_INT32)));
  }

  @Test
  public void testRange1PosLimitDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_1_pos_limit_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRange1PosDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_range_1_pos_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_1_INT32)));
  }

  @Test
  public void testRangeStartDeltaKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_delta_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRangeStartKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_start_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_INT32)));
  }

  @Test
  public void testRangeKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testRangeMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_range_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_input.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_32_FLOAT32)));
  }

  @Test
  public void testInput2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_input2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_NONE_NONE_STRING)));
  }

  /**
   * Regression guard for wala/ML#355: when the {@code shape} argument to {@code tf.keras.Input} is
   * unresolvable from the static analyzer's perspective (here, sourced from {@code json.loads},
   * which Ariadne does not model), {@code Input.getDefaultShapes} must return {@code null} (⊤,
   * tensor with unknown shape) rather than {@code Collections.emptySet()} (⊥, not a tensor). The ⊥
   * return previously made the call's result silently disappear from the tensor analysis despite
   * being a tensor at runtime; the fix in ponder-lab/ML@078208f6 restores ⊤ propagation, so the
   * call site is recognized as a tensor with concrete dtype but unknown shape.
   */
  @Test
  public void testInputUnresolvableShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_input_unresolvable_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/604">wala/ML#604</a>: when an
   * allocator's {@code shape} argument is unresolvable (here {@code tf.zeros(json.loads(...))},
   * whose source Ariadne does not model), {@code TensorTypeAllocator.getDefaultShapes} must return
   * {@code null} (⊤, tensor with unknown shape) rather than throwing {@code
   * UnsupportedOperationException}, which previously aborted the whole analysis. Recovering the
   * content-dependent shape itself is the user-annotation problem tracked by wala/ML#370; this
   * guard only pins the non-crashing ⊤ floor.
   */
  @Test
  public void testZerosUnresolvableShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_zeros_unresolvable_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  @Test
  public void testConvertToTensor13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_convert_to_tensor13.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_3_FLOAT32),
            3, Set.of(TENSOR_2_2_INT32),
            4, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testEye7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye7.py",
        "test",
        3,
        3,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32),
            3, Set.of(TENSOR_3_3_FLOAT32),
            4, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Guards function-style {@code np.array(x, dtype)} shape-preservation. Mirrors {@link
   * #testAstype}'s shape/dtype contract but through the function-call path rather than the
   * method-call path: {@code np.array(x_train, np.float32)} should yield a tensor with {@code
   * x_train}'s shape {@code (60000, 28, 28)} and dtype {@code float32}.
   *
   * <p>Positive regression guard for wala/ML#404: {@code np.array} is modeled in {@code numpy.xml}
   * as a {@code Lnumpy/array} class whose {@code do} method returns a fresh {@code Lnumpy/ndarray},
   * and {@link NpArray} reads shape from arg 0 ({@code x}) and dtype from arg 1 ({@code dtype}).
   */
  @Test
  public void testNpArrayPreservesShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_array_preserves_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_60000_28_28_FLOAT32)));
  }

  /**
   * Pins {@link NpOnes} directly (isolated from the {@code tf.constant} bridge of {@link
   * #testConstantFromNumpy}): {@code consume_ones(x)} where {@code x = np.ones((2, 3),
   * dtype=np.float32)} should yield {@code (2, 3) float32} &mdash; shape from the shape-tuple
   * argument, dtype from the explicit {@code dtype} argument. Positive regression guard for
   * wala/ML#539.
   */
  @Test
  public void testNpOnes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_ones_zeros.py", "consume_ones", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins {@link NpZeros} directly: {@code consume_zeros(x)} where {@code x = np.zeros((4,),
   * dtype=np.int64)} should yield {@code (4,) int64}. Companion to {@link #testNpOnes}; positive
   * regression guard for wala/ML#539.
   */
  @Test
  public void testNpZeros()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_np_ones_zeros.py", "consume_zeros", 1, 1, Map.of(2, Set.of(TENSOR_4_INT64)));
  }

  /**
   * Pins {@link NpOnes}'s default dtype: {@code consume_ones_default(x)} where {@code x =
   * np.ones((2, 3))} (no {@code dtype} argument) should yield {@code (2, 3) float64}, since NumPy
   * defaults to {@code float64} (unlike {@code tf.ones}, which defaults to {@code float32}). Guards
   * the {@code float64} default override in {@link NpOnes#getDefaultDTypes} for wala/ML#539.
   */
  @Test
  public void testNpOnesDefaultDtype()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_np_ones_zeros.py",
        "consume_ones_default",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT64)));
  }

  /**
   * Guards {@code tf.fill}'s {@code .shape}-argument recovery (<a
   * href="https://github.com/wala/ML/issues/610">wala/ML#610</a>): {@code tf.fill(x.shape, 5.0)}
   * where {@code x} is {@code (2, 2)} yields {@code (2, 2) float32}. The {@code dims} argument is a
   * {@code .shape} property read with an empty points-to set, so resolution falls to {@link
   * com.ibm.wala.cast.python.ml.client.Fill#getDefaultShapes}, which recovers the source tensor's
   * shape rather than dropping to ⊤.
   */
  @Test
  public void testFillShapeArg()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_fill_shape_arg.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/606">wala/ML#606</a>: when
   * {@code tf.fill}'s {@code dims} argument is unresolvable (here from {@code json.loads}), {@link
   * com.ibm.wala.cast.python.ml.client.Fill#getDefaultShapes} must return ⊤ rather than throwing
   * {@code UnsupportedOperationException}, which previously aborted the whole analysis. {@code
   * Fill} extends {@code Constant}, so the base allocator floor (wala/ML#604) doesn't cover it. The
   * result is ⊤-shape {@code int32} (the fill value's dtype).
   */
  @Test
  public void testFillUnresolvableDims()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fill_unresolvable_dims.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Guards allocator-shape recovery from a {@code .shape} argument (<a
   * href="https://github.com/wala/ML/issues/604">wala/ML#604</a>): {@code tf.ones(x.shape)} where
   * {@code x} is {@code (2, 2)} yields {@code (2, 2) float32}. The shape argument is a {@code
   * .shape} property read with an empty points-to set, so resolution falls to {@link
   * com.ibm.wala.cast.python.ml.client.TensorTypeAllocator#getDefaultShapes}, which recovers the
   * source tensor's shape via {@code getShapeFromShapeAttributeArgument} rather than dropping to ⊤.
   */
  @Test
  public void testOnesTensorShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ones_tensor_shape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Guards the {@code tf.eye} unresolvable-{@code num_rows} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when {@code num_rows} is
   * unresolvable (here from {@code json.loads}), the result is still a rank-2 square matrix, so it
   * floors to {@code (Dynamic, Dynamic)} float32 rather than throwing "num_rows parameter is
   * required" (which previously aborted the whole analysis).
   */
  @Test
  public void testEyeUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNRESOLVED_UNRESOLVED_FLOAT32)));
  }

  /**
   * Complements {@link #testEyeUnresolvableBatchShape()}: when {@code batch_shape} is a list
   * literal whose length (and hence the output rank) is statically known but whose element is
   * unresolvable (here from {@code json.loads}), precision is preserved rather than floored to ⊤.
   * The single leading batch dimension is a fixed runtime size the analysis cannot compute ({@link
   * UnresolvedDim}, wala/ML#721) and the {@code (num_rows, num_columns)} suffix stays exact, so
   * {@code tf.eye(3, batch_shape=[<unknown>])} types to {@code (Unresolved, 3, 3)} float32. See <a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>.
   */
  @Test
  public void testEyeDynamicBatch()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t =
        new TensorType(
            FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(3), new NumericDim(3)));
    test("tf2_test_eye_dynamic_batch.py", "consume", 1, 1, Map.of(2, Set.of(t)));
  }

  /**
   * Guards the {@code tf.eye} unresolvable-{@code batch_shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when {@code batch_shape} is
   * present but unresolvable (here from {@code json.loads}), the number of leading batch dimensions
   * is unknown, so the overall rank can't be known and the result floors to ⊤ rather than throwing
   * "Batch shape argument for tf.eye() should be a list of dimensions." (which previously aborted
   * the whole analysis). The dtype stays float32.
   *
   * <p>TODO: the {@code batch_shape} value here is content-dependent (it comes from {@code
   * json.loads}), so recovering it is the user-annotation problem tracked by <a
   * href="https://github.com/wala/ML/issues/370">wala/ML#370</a>, the same recovery path the
   * allocator shape floor points at. Orthogonally, a structurally-inferable tensor {@code
   * batch_shape} (e.g. {@code tf.shape(x)}, whose rank, and often values, are statically known) can
   * be recovered without an annotation; that is tracked by <a
   * href="https://github.com/wala/ML/issues/619">wala/ML#619</a>. The already-recoverable
   * known-rank case is guarded by {@link #testEyeDynamicBatch()}.
   */
  @Test
  public void testEyeUnresolvableBatchShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_eye_unresolvable_batch_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Regression for the {@code getIntValueFromInstanceKey} non-numeric-constant degradation (<a
   * href="https://github.com/wala/ML/issues/590">wala/ML#590</a>): {@code tf.eye(True)} models the
   * Python {@code bool} as a {@code Boolean} constant, which degrades to {@code int(True) == 1} (a
   * {@code (1, 1)} identity) rather than throwing a {@code ClassCastException} on the {@code
   * Number} cast.
   */
  @Test
  public void testEyeBoolDim()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye_bool_dim.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_1_1_FLOAT32)));
  }

  /**
   * Companion to {@link #testEyeBoolDim} covering the {@code int(False) == 0} branch of {@code
   * getIntValueFromInstanceKey} (<a href="https://github.com/wala/ML/issues/590">wala/ML#590</a>):
   * {@code tf.eye(False)} degrades the {@code Boolean} to {@code 0}, yielding a {@code (0, 0)}
   * identity.
   */
  @Test
  public void testEyeBoolDimFalse()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye_bool_dim_false.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_0_0_FLOAT32)));
  }

  /**
   * Covers {@code tf.eye} with a {@code batch_shape}, which prepends the batch dimensions to the
   * identity shape (<a href="https://github.com/wala/ML/issues/591">wala/ML#591</a>): a {@code (3,
   * 3)} identity with {@code batch_shape=[2]} is {@code (2, 3, 3)}. Exercises the fresh-list
   * construction that replaced the shared-list mutation.
   */
  @Test
  public void testEyeBatchShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_eye_batch_shape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_3_FLOAT32)));
  }

  /**
   * Regression guard for wala/ML#449's closing fix: {@code tf.random.truncated_normal(shape)} now
   * dispatches to the {@code TruncatedNormal} generator (via {@code PROPERTY_NAME_GENERATORS}) and
   * resolves to precise {@code (2, 3) float32}. Pre-fix this fell through to {@code
   * ReadDataFallback} and emitted {@code ⊤ shape / UNKNOWN dtype}.
   */
  @Test
  public void testTruncatedNormal()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_truncated_normal.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Verifies that {@code tf.linspace(0.0, 10.0, 5)} routes through the dedicated {@link Linspace}
   * generator and emits the precise rank-1 shape {@code (5,)} with {@code float32} dtype (derived
   * from the float-typed {@code start} argument).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLinspace()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT32)));
  }

  /**
   * Companion to {@link #testLinspace()} that exercises the integer-promotion branch in {@link
   * com.ibm.wala.cast.python.ml.client.Linspace#getDefaultDTypes}. {@code tf.linspace} with integer
   * {@code start}/{@code stop} promotes the output to {@code float64} (verified on TF 2.9), not
   * {@code float32}. The float-input case is covered by {@link #testLinspace()}.
   */
  @Test
  public void testLinspaceInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace_int.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_FLOAT64)));
  }

  /**
   * Drives the {@code axis}-passed branch in {@link
   * com.ibm.wala.cast.python.ml.client.Linspace#getDefaultShapes}. With {@code axis=1} and vector
   * {@code start}/{@code stop}, {@code tf.linspace} interpolates along axis 1, producing a
   * higher-rank result whose runtime shape is {@code (2, 5)} with dtype {@code float32}.
   *
   * <p>The static analysis currently returns ⊤ (unknown shape) for any axis-passed call — combining
   * {@code start}'s rank with {@code num} is not yet implemented; the generator trades precision
   * for soundness. The assertion encodes the observed result with a TODO pointing at the precision
   * improvement that would narrow it.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/475">wala/ML#475</a> is fixed and
   * {@code Linspace.getDefaultShapes} computes the precise output shape from {@code
   * start.shape[:axis] + (num,) + start.shape[axis:]}, narrow the assertion to {@code
   * Set.of(TENSOR_2_5_FLOAT32)} (a new constant).
   *
   * <p>Companion to {@link #testLinspace()} (covering the absent-axis branch).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLinspaceAxis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_linspace_axis.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  @Test
  public void testGamma()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testGamma2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testGamma3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testGamma4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGamma6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testGammaMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gamma_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_30_3_2_FLOAT32)));
  }

  @Test
  public void testPoisson()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT32)));
  }

  @Test
  public void testPoisson3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_10_2_FLOAT64)));
  }

  @Test
  public void testPoisson4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_poisson4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_7_5_2_FLOAT32)));
  }

  @Test
  public void testGamma7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  @Test
  public void testPoisson5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson5.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_FLOAT64),
            3, Set.of(TENSOR_2_2_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.gamma} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): when the {@code shape} argument
   * is unresolvable (here from {@code json.loads}) the output rank can't be known, so the result
   * floors to ⊤ rather than throwing (which previously aborted the whole analysis). The dtype stays
   * float32.
   */
  @Test
  public void testGammaUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_gamma_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the {@code tf.random.poisson} unresolvable-{@code shape} floor (<a
   * href="https://github.com/wala/ML/issues/611">wala/ML#611</a>): same as {@link
   * #testGammaUnresolvable()}, the output rank rides on the unresolvable {@code shape}, so the
   * result floors to ⊤ rather than throwing.
   */
  @Test
  public void testPoissonUnresolvable()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_poisson_unresolvable.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Regression test for `wala/ML#447`: scalar `tf.constant(<bool>)` exercises the {@link
   * java.lang.Boolean} arm of {@code TensorGenerator.getDTypesOfValue}. Without it, dtype inference
   * threw {@code IllegalStateException: Unknown constant type: class java.lang.Boolean}.
   */
  @Test
  public void testBoolConstant()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bool_constant.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_BOOL)));
  }

  /**
   * List-of-bool form of {@link #testBoolConstant} — exercises the recursive {@code
   * getDTypesOfValue} call (line 1625) on a list whose elements are `Boolean` constants.
   */
  @Test
  public void testBoolConstant2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_bool_constant.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_BOOL)));
  }

  @Test
  public void testVariablePositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_positional.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  @Test
  public void testVariableKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_keyword.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
    test("tf2_test_variable_keyword.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testVariableMixed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_variable_mixed.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
  }

  @Test
  public void testVariablePositionalComplex()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_variable_positional_complex.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testVariablePositionalShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test(
        "tf2_test_variable_positional_shape.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_NONE_2_FLOAT32)));
  }

  @Test
  public void testVariablePositionalDType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {

    test("tf2_test_variable_positional_dtype.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT64)));
  }
}
