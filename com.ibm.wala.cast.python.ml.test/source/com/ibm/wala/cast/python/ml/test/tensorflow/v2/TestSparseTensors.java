package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_4_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_2_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_5_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_5_5_INT32;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of sparse-tensor shape and dtype inference, carved from the {@code TestTensorflow2Model}
 * monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestSparseTensors extends AbstractTensorTest {

  /**
   * Regression guard for wala/ML#618's data-pipeline fix: {@code tf.sparse.to_dense} of a {@code
   * SparseTensor} types its dense result from the operand's {@code dense_shape} field (shape) and
   * {@code values} field (dtype), so {@code consume}'s parameter types to {@code (2,2)} int32.
   * Modeling this is what un-strands {@code get_loss}'s {@code real} (the dataset-sourced {@code
   * targets}) in {@link #testGpt2GetLossVendored()}.
   */
  @Test
  public void testSparseToDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_to_dense.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 2))));
  }

  @Test
  public void testSparseAdd()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseAdd2()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseAdd3()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32.asSparse())));
  }

  @Test
  public void testSparseAdd4()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32.asSparse())));
  }

  @Test
  public void testSparseAdd5()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseAdd6()
      throws ClassHierarchyException, IllegalArgumentException, IOException, CancelException {
    test("tf2_test_sparse_add6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_2_FLOAT32.asSparse())));
  }

  /**
   * {@code tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)} returns a fresh loss
   * tensor of shape {@code logits.shape[:-1]} and dtype {@code float32}. For this test, logits is
   * {@code (3, 4) float32} so the loss is {@code (3,) float32} (verified by out-of-band TF runtime
   * probe).
   *
   * <p>Expectation evolution:
   *
   * <ul>
   *   <li>Master: {@code MNIST_INPUT} &mdash; a generic tensor sentinel from before the analyzer
   *       was specific enough to narrow to a rank-1 shape for this sink.
   *   <li>Earlier on branch 267 ({@code 13c7ec0a}): narrowed to {@code TENSOR_3_INT32}, matching
   *       the pass-through bug's behaviour &mdash; the XML {@code <return value="labels"/>} on
   *       {@code sparse_softmax_cross_entropy_with_logits.do()} made the call's result share {@code
   *       labels}' shape and dtype ({@code (3,) int32}). Specific but semantically wrong.
   *   <li>Now (post-wala/ML#412, {@code 0cfdadc4}): {@code TENSOR_3_FLOAT32}. Runtime-correct.
   * </ul>
   */
  @Test
  public void testSparseSoftmaxCrossEntropyWithLogits()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_softmax_cross_entropy_with_logits.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_FLOAT32)));
  }

  /**
   * Generator-dispatch test for {@code tf.sparse.from_dense(tensor, name=None)}. The fixture uses
   * the keyword form {@code from_dense(tensor=x)} so that arg-resolution drives through {@link
   * com.ibm.wala.cast.python.ml.client.SparseFromDense}'s {@code Parameters.TENSOR.getName()}
   * keyword-name lookup, exercising the {@code Locale.ROOT} line that wala/ML#510 flagged as
   * uncovered. Output shape and dtype both inherit from {@code tensor}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSparseFromDense()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_from_dense.py",
        "f",
        1,
        1,
        Map.of(2, Set.of(TENSOR_2_3_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye2() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye2.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye3() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye3.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_5_INT32.asSparse())));
  }

  @Test
  public void testSparseEye4() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye4.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye5() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye5.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_FLOAT32.asSparse())));
  }

  @Test
  public void testSparseEye6() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_eye6.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_5_2_INT32.asSparse())));
  }

  @Test
  public void testSparseTensor() throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_sparse_tensor.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_4_INT32.asSparse())));
  }

  @Test
  public void testSparseAdd7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_add7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_2_INT32.asSparse()),
            3, Set.of(TENSOR_2_2_INT32.asSparse())));
  }

  @Test
  public void testSparseEye7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_sparse_eye7.py",
        "test",
        2,
        2,
        Map.of(
            2, Set.of(TENSOR_2_3_INT32.asSparse()),
            3, Set.of(TENSOR_3_3_FLOAT32.asSparse())));
  }

  /**
   * Regression guard for wala/ML#646: a SparseTensor flows through a dict subscript ({@code
   * features["t"]}) and keeps its type. {@code tf.sparse.SparseTensor} allocates directly in {@code
   * do()} (the former {@code read_data} call was inlined), so the result carries a live points-to
   * set that survives the dict {@code putfield}/{@code getfield}; the earlier empty PTS dropped it.
   */
  @Test
  public void testSparseTensorThroughDict()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dict_subscript.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 2, 2).asSparse())));
  }
}
