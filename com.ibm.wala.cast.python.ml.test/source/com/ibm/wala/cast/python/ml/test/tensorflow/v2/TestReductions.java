package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_BOOL;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_1_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_INT32_UNKNOWN_SHAPE;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of reduction and per-element rounding ops ({@code reduce_*}, {@code argmax}/{@code argmin},
 * {@code top_k}, {@code ceil}), carved from the {@code TestTensorflow2Model} monolith
 * (wala/ML#635); the assertions are verbatim.
 */
public class TestReductions extends AbstractTensorTest {

  @Test
  public void testReduceMean()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceMax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_max.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.reduce_min(input_tensor)}. Mirrors {@link
   * #testReduceMax()} — the {@link com.ibm.wala.cast.python.ml.client.ReduceMin} generator extends
   * {@link com.ibm.wala.cast.python.ml.client.ReduceMax}, sharing the axis-collapse / keepdims
   * shape inference and input-dtype passthrough.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testReduceMin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_min.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceProd()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_prod.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceLogSumExp()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_logsumexp.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceAll()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_all.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_BOOL)));
  }

  @Test
  public void testReduceMean2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceMean6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  @Test
  public void testReduceMean7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "i", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testReduceMean8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_mean_kwargs.py", "j", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  @Test
  public void testReduceSum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Guards that {@code tf.reduce_sum} preserves an integer input's dtype rather than promoting it
   * to {@code float32}. Unlike {@code tf.reduce_mean}, summing integers yields an integer;
   * previously the {@code Reduce*} ops inherited {@code ReduceMean}'s mean-specific promotion via
   * {@code extends ReduceMean}, so {@code reduce_sum} of an {@code int32} was mistyped {@code
   * float32}. Fixed by the {@code Reduction}-base hierarchy (wala/ML#514).
   */
  @Test
  public void testReduceSumInt()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_int.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Verifies that {@code tf.math.argmax(x, axis=0)} routes through the dedicated {@link
   * com.ibm.wala.cast.python.ml.client.Argmax} generator and emits the precise {@code int64} dtype
   * (the TF default for argmax indices) and the precise {@code (3,)} shape ({@code (2, 3)} with
   * {@code axis=0} removed). Shape precision was unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_argmax.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT64)));
  }

  /**
   * Counterpart of {@link #testArgmax()} for {@code tf.math.argmin}. Same semantics: dtype defaults
   * to {@code int64} (overridable via {@code output_type}, see {@link #testArgminOutputType()}),
   * and shape is the precise {@code (3,)} ({@code (2, 3)} with {@code axis=0} removed), unblocked
   * by wala/ML#530. See {@link com.ibm.wala.cast.python.ml.client.Argmin}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmin()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_argmin.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT64)));
  }

  /**
   * Verifies that {@code tf.math.argmax(x, axis=0, output_type=tf.int32)} honors the explicit
   * {@code output_type} override and emits an {@code int32}-dtype tensor instead of the {@code
   * int64} default. Exercises the dtype-arg dispatch path on {@link
   * com.ibm.wala.cast.python.ml.client.Argmax} after the wala/ML#463 fix. The fixture's sink {@code
   * f(x, y)} has two parameters so that each tensor's inferred type can be checked independently.
   * The result {@code y} (vn=3) has the precise {@code (3,) int32} shape ({@code (2, 3)} with
   * {@code axis=0} removed), unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxOutputType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_output_type.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Like {@link #testArgmaxOutputType()} but passes {@code output_type} positionally ({@code
   * tf.math.argmax(x, 0, tf.int32)}). Shape inference must not misread the positional {@code
   * output_type} argument as {@link com.ibm.wala.cast.python.ml.client.ReduceMean}'s {@code
   * keepdims} (they share positional index 2); the result {@code y} (vn=3) is the precise {@code
   * (3,) int32}, not a {@code keepdims=true} union (e.g. {@code (1, 3)}). Regression guard for the
   * {@code getKeepDimsValues} override on {@link com.ibm.wala.cast.python.ml.client.Argmax}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxOutputTypePositional()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_output_type_positional.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Tests that {@code tf.math.argmax} resolves its input when passed by keyword ({@code
   * tf.math.argmax(input=x, axis=0)}). {@link com.ibm.wala.cast.python.ml.client.Argmax} delegates
   * shape inference to {@link com.ibm.wala.cast.python.ml.client.ReduceMean}, whose input parameter
   * is named {@code input_tensor}; argmax's is named {@code input}, so without overriding the
   * input-parameter name the keyword lookup fails and shape inference throws {@code
   * IllegalStateException}. The result {@code y} (vn=3) is the precise {@code (3,) int64}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxInputKeyword()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_input_keyword.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT64)));
  }

  /**
   * Counterpart of {@link #testArgmaxOutputType()} for {@code tf.math.argmin}. Same dispatch path
   * via the inherited {@link com.ibm.wala.cast.python.ml.client.Argmin} extends {@link
   * com.ibm.wala.cast.python.ml.client.Argmax} relationship; the result {@code y} (vn=3) has the
   * precise {@code (3,) int32} shape, unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgminOutputType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmin_output_type.py",
        "f",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_FLOAT32),
            3, Set.of(TENSOR_3_INT32)));
  }

  /**
   * Companion to {@link #testArgmaxOutputType()} that exercises the *single-parameter sink, two
   * call sites* shape: {@code def f(a): ...; f(x); f(y)}. Parameter {@code a} should union {@code
   * x}'s and {@code y}'s tensor types across the two call sites &mdash; verifies that the {@code
   * output_type=tf.int32} override on {@code y} survives the second sink call rather than being
   * clobbered by the {@code int64} default. The {@code y} contribution to the union is the precise
   * {@code (3,) int32} shape, unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgmaxOutputTypeDoubleSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmax_output_type_double_sink.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_3_INT32)));
  }

  /**
   * Counterpart of {@link #testArgmaxOutputTypeDoubleSink()} for {@code tf.math.argmin}; the {@code
   * y} contribution to the union is the precise {@code (3,) int32} shape, unblocked by wala/ML#530.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testArgminOutputTypeDoubleSink()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_argmin_output_type_double_sink.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32, TENSOR_3_INT32)));
  }

  @Test
  public void testReduceSum2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  @Test
  public void testReduceSum6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "h", 1, 1, Map.of(2, Set.of(TENSOR_1_2_FLOAT32)));
  }

  /**
   * Exercises the {@code tf.math.reduce_sum} (ref="math") binding via {@code
   * tf.math.reduce_sum(input_tensor=x, axis=1, keepdims=True)}.
   */
  @Test
  public void testReduceSum7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "i", 1, 1, Map.of(2, Set.of(TENSOR_2_1_FLOAT32)));
  }

  @Test
  public void testReduceSum8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_reduce_sum_kwargs.py", "j", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Pure passthrough on {@code x}. See {@link com.ibm.wala.cast.python.ml.client.Ceil}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeil()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testCeil} that exercises the {@code x} side of the fixture (the input to
   * {@code tf.math.ceil}, asserted at the second sink {@code g(x)}). Combined with {@link
   * #testCeil} (the {@code y} side, the {@code ceil} output), this covers both ends of the
   * passthrough.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Keyword-argument variant of {@link #testCeil}: {@code tf.math.ceil(x=x)}. Exercises the keyword
   * arg-resolution path on the {@code PassThroughUnaryTensorGenerator} base.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilKw()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil_kw.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Companion to {@link #testCeilKw} that exercises the {@code x} side of the keyword-arg fixture.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilKwInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_ceil_kw.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * 2-arg-sink variant of {@link #testCeil}: same input and op, but with one combined sink {@code
   * f(y, x)} instead of two separate single-arg sinks {@code f(y); g(x)}. This is the same shape as
   * the wala/ML#495 multi-tensor-sink pattern; the difference is that #495 is specifically about
   * dataset-loader outputs (`fashion_mnist`/`cifar100`/etc.) flowing through {@link
   * com.ibm.wala.cast.python.ml.client.TensorGenerator#shapesFromSSAChain}'s fallback path. For
   * {@code ceil} on {@code tf.constant}, no fallback is involved, so the pattern works precisely
   * today &mdash; this test asserts the lattice-correct {@code (3,) float32} on both params and
   * stands as a canary: if #495 ever generalizes beyond dataset loaders to per-op generators, this
   * test will start failing.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCeilPair()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_ceil_pair.py",
        "f",
        2,
        2,
        Map.of(2, Set.of(TENSOR_3_FLOAT32), 3, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Guards the {@code k}-default path of the top_k composer (<a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>): with {@code k} omitted it
   * defaults to {@code 1}, so {@code values} of a {@code (4,)} input is {@code (1,)} float32.
   */
  @Test
  public void testTopkDefaultK()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_topk_default_k.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_1_FLOAT32)));
  }

  /**
   * Guards the non-constant-{@code k} path of the top_k composer (<a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>): when {@code k} is not a
   * resolvable integer constant (here from {@code json.loads}), the shape can't be composed and
   * degrades to ⊤ rather than guessing. The dtype stays precise (float32).
   */
  @Test
  public void testTopkNonConstantK()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_topk_nonconstant_k.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the unknown-input-shape path of the top_k composer (<a
   * href="https://github.com/wala/ML/issues/609">wala/ML#609</a>): when the input tensor's shape is
   * ⊤ (here {@code tf.ones(json.loads(...))}), {@code input.shape[:-1] + (k,)} can't be composed
   * and the result degrades to ⊤. The dtype stays precise (float32).
   */
  @Test
  public void testTopkUnknownInput()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_topk_unknown_input.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Guards the top_k output-shape composer (<a href="https://github.com/wala/ML/issues/609">
   * wala/ML#609</a>): {@code values, indices = tf.math.top_k(x, k=2)} on a {@code (5,)} input
   * yields {@code values} of shape {@code (2,)} float32, composed as {@code input.shape[:-1] +
   * (k,)} by {@link com.ibm.wala.cast.python.ml.client.TopK} rather than left at ⊤. Destructuring
   * (not the wala/ML#480 attribute-access path) gives the precise per-element type.
   */
  @Test
  public void testTopkShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_topk_shape.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/603">wala/ML#603</a>: slicing a
   * NamedTuple result ({@code tf.math.top_k}) walks an object catalog whose keys include the string
   * field aliases {@code values}/{@code indices} alongside the integer element indices. Those
   * non-integer keys must be filtered rather than crashing {@code getFieldIndex}; the slice then
   * recovers the element dtypes ({@code float32} values, {@code int32} indices). No {@code
   * read_data} is involved, so this is a case wala/ML#380 would not fix.
   *
   * <p>The composed {@code (k,) = (2,)} shape now appears for both elements (the wala/ML#609
   * composer), so the asserted set is a union of the precise {@code (2,)} shapes and the residual ⊤
   * ones. The ⊤ components come from the wala/ML#480 attribute/slice path, which doesn't carry the
   * composed per-element shape through; once wala/ML#480 lands they drop, narrowing this to {@code
   * Set.of(TENSOR_2_FLOAT32, TENSOR_2_INT32)}.
   */
  @Test
  public void testTopkSliceCatalog()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_topk_slice_catalog.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TENSOR_UNKNOWN_SHAPE_FLOAT32,
                TENSOR_2_FLOAT32,
                TENSOR_INT32_UNKNOWN_SHAPE,
                TENSOR_2_INT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_sum} (wala/ML#570). The output dtype
   * inherits from the {@code data} input ({@code float32}); the shape is the concrete {@code (2,
   * 3)} — {@code [num_segments] ++ data.shape[segment_ids.ndim:]} with the static {@code
   * num_segments = 2}, rank-1 {@code segment_ids}, and {@code (3, 3)} {@code data} (wala/ML#582).
   * Verified via a {@code consume_sum} sink on the aggregation result.
   */
  @Test
  public void testUnsortedSegmentSum()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py", "consume_sum", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_max} (wala/ML#570). Same dtype-from-
   * {@code data} and static-{@code num_segments} {@code (2, 3)} shape recovery as {@link
   * #testUnsortedSegmentSum} (wala/ML#582).
   */
  @Test
  public void testUnsortedSegmentMax()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py", "consume_max", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Pins the output type of {@code tf.math.unsorted_segment_mean} (wala/ML#570). Same dtype-from-
   * {@code data} and static-{@code num_segments} {@code (2, 3)} shape recovery as {@link
   * #testUnsortedSegmentSum} (wala/ML#582).
   */
  @Test
  public void testUnsortedSegmentMean()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_unsorted_segment.py",
        "consume_mean",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * Tier-5 generator (wala/ML#449): {@code tf.math.top_k(input, k)}. Returns a {@code (values,
   * indices)} 2-tuple. The dedicated {@link com.ibm.wala.cast.python.ml.client.TopK} generator
   * implements {@link com.ibm.wala.cast.python.ml.client.TupleElementProvider} with per-index dtype
   * precision ({@code values} inherits input dtype; {@code indices} is fixed at {@code int32}), but
   * the wrap-on-property-read dispatch in {@link
   * com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory} doesn't fire for NamedTuple-style
   * attribute access ({@code result.values} / {@code result.indices}). Until <a
   * href="https://github.com/wala/ML/issues/480">wala/ML#480</a> is fixed, both destructured
   * elements receive the aggregate union of per-element types.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow the
   * assertion to {@code Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)} (precise {@code values} dtype).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTopKValues()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_top_k.py",
        "f_values",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /**
   * Counterpart of {@link #testTopKValues()} for the {@code indices} element of {@code
   * tf.math.top_k}'s tuple result. Same wala/ML#480-driven imprecision: the assertion captures the
   * observed aggregate union with a TODO to narrow once the per-index dispatch is fixed.
   *
   * <p>TODO: Once <a href="https://github.com/wala/ML/issues/480">wala/ML#480</a> lands, narrow the
   * assertion to {@code Set.of(TENSOR_INT32_UNKNOWN_SHAPE)} (precise {@code indices} dtype).
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTopKIndices()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_top_k.py",
        "f_indices",
        1,
        1,
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32, TENSOR_INT32_UNKNOWN_SHAPE)));
  }
}
