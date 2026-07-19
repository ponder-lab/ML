package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.FLOAT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.SCALAR_TENSOR_OF_STRING;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_1_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_32_32_3_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_10000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_102_13_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_102_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_1_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2246_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2246_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_25000_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_25000_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_2_NONE_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_32_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_3_INT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_404_13_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_404_FLOAT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_4_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_4_8_FLOAT32;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_50000_1_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_50000_32_32_3_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_28_28_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_60000_UINT8;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_8982_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_8982_OBJECT;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_DYNAMIC_INT64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.TENSOR_UNKNOWN_SHAPE_FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyMap;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests of {@code tf.data} datasets, Python-collection probes, and generator iteration, carved from
 * the {@code TestTensorflow2Model} monolith (wala/ML#635); the assertions are verbatim.
 */
public class TestDatasets extends AbstractTensorTest {

  @Test
  public void testDataset()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset2.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** This is not a legal case. */
  @Test
  public void testDataset3()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset3.py", "add", 0, 0, emptyMap());
  }

  /** This is not a legal case. */
  @Test
  public void testDataset4()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset4.py", "add", 0, 0, emptyMap());
  }

  @Test
  public void testDataset5()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset5.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset6()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset6.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset7()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset7.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  @Test
  public void testDataset8()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset8.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_32_28_28_UINT8), 3, Set.of(TENSOR_32_UINT8)));
  }

  @Test
  public void testDataset9()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset9.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(TENSOR_32_28_28_UINT8), 3, Set.of(TENSOR_32_UINT8)));
  }

  @Test
  public void testDataset10()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset10.py",
        "add",
        2,
        3,
        Map.of(
            2,
            Set.of(SCALAR_TENSOR_OF_INT32, TENSOR_2_NONE_INT32),
            3,
            Set.of(SCALAR_TENSOR_OF_INT32, TENSOR_2_NONE_INT32)));
  }

  @Test
  public void testDataset10a()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset10a.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(SCALAR_TENSOR_OF_INT32),
            3, Set.of(TENSOR_2_NONE_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset11()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset11.py", "f", 0, 0);
    test("tf2_test_dataset11.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset12()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset12.py", "f", 0, 0);
    test("tf2_test_dataset12.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset13()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset13.py", "f", 0, 0);
    test("tf2_test_dataset13.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset14()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset14.py", "f", 0, 0);
    test("tf2_test_dataset14.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test enumerating a dataset (https://github.com/wala/ML/issues/140). The first element of the
   * tuple returned isn't a tensor.
   */
  @Test
  public void testDataset15()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset15.py", "f", 0, 0);
    test("tf2_test_dataset15.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset16()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset16.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset17()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset17.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset17.py", "f", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset18()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset18.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset18.py", "f", 1, 2, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset18.py", "g", 0, 2);
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset19()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType images = TensorType.of(FLOAT_32, 512, 112, 112, 3);
    TensorType labels =
        new TensorType(FLOAT_32, asList(new NumericDim(512), UnresolvedDim.INSTANCE));

    test(
        "tf2_test_dataset19.py", "distributed_train_step", 1, 1, Map.of(2, Set.of(images, labels)));
  }

  @Test
  public void testDataset20()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset20.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset21()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset21.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset22()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset22.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset23()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset23.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset23.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset24()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset24.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset24.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset25()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset25.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset25.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/165 is fixed.
    test(
        "tf2_test_dataset25.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset26()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset26.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset26.py", "g1", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset26.py", "g2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset26.py", "g3", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/165 is fixed.
    test(
        "tf2_test_dataset26.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, TENSOR_2_INT32)));
  }

  @Test
  public void testDataset27()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset27.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
    test("tf2_test_dataset27.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset27.py", "h", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset27.py", "i", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  @Test
  public void testDataset28()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset28.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_3_INT32)));
    test("tf2_test_dataset28.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    // TODO: Change to 0, 0 when https://github.com/wala/ML/issues/164 is fixed:
    test(
        "tf2_test_dataset28.py",
        "h",
        1,
        1,
        Map.of(2, Set.of(TENSOR_3_INT32, SCALAR_TENSOR_OF_STRING)));
  }

  @Test
  public void testDataset29()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset29.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  @Test
  public void testDataset30()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset30.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset31()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "g1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "g2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "h", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "i1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "i2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test(
        "tf2_test_dataset31.py",
        "j",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64, TENSOR_2_INT64)));
    test("tf2_test_dataset31.py", "k1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "k2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "k3", 1, 1, Map.of(2, Set.of(TENSOR_2_INT64)));
    // TODO: Change to 0, 0 once https://github.com/wala/ML/issues/166 is fixed.
    test("tf2_test_dataset31.py", "l", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "m1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset31.py", "m2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset32()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset32.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  @Test
  public void testDataset33()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset33.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset34()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset34.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset35()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset35.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset36()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset36.py", "id1", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
    //    test("tf2_test_dataset36.py", "id2", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /** Test a dataset that uses an iterator. */
  @Test
  public void testDataset37()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset37.py",
        "add",
        2,
        3,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32), 3, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using a shared
   * function to retrieve the first element from two different datasets.
   */
  @Test
  public void testDataset38()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset38.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset38.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Control test for {@link #testDataset38()}, utilizing only a single dataset to ensure the 1-CFA
   * precision issue is specifically due to the merging of multiple datasets.
   */
  @Test
  public void testDataset39()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset39.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using a shared
   * function to retrieve the first element from two different datasets. Uses tf.constant to bypass
   * unrelated type inference issues with Python lists in from_tensor_slices.
   */
  @Test
  public void testDataset40()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset40.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset40.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /**
   * Control test for {@link #testDataset40()}, utilizing only a single dataset to ensure the 1-CFA
   * precision issue is specifically due to the merging of multiple datasets.
   */
  @Test
  public void testDataset41()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset41.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset38()}, iterating over the dataset directly
   * instead of through a shared function, avoiding the 1-CFA precision issue.
   */
  @Test
  public void testDataset42()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset42.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset42.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset42()}, utilizing only a single dataset. */
  @Test
  public void testDataset43()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset43.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset40()}, using tf.constant and iterating directly.
   */
  @Test
  public void testDataset44()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset44.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset44.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset44()}, utilizing only a single dataset. */
  @Test
  public void testDataset45()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset45.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case based on {@link #testDataset38()}, exercising potential 1-CFA precision problem with
   * Datasets by varying the shape (e.g., shape (2,) vs (2, 2)) instead of the dtype.
   */
  @Test
  public void testDataset46()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset46.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset46.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset46()}, utilizing only a single dataset. */
  @Test
  public void testDataset47()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset47.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case based on {@link #testDataset40()}, exercising potential 1-CFA precision problem with
   * Datasets by varying the shape using tf.constant.
   */
  @Test
  public void testDataset48()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset48.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset48.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset48()}, utilizing only a single dataset. */
  @Test
  public void testDataset49()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset49.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset46()}, iterating over the dataset directly
   * instead of through a shared function.
   */
  @Test
  public void testDataset50()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset50.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset50.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset50()}, utilizing only a single dataset. */
  @Test
  public void testDataset51()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset51.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Direct access test based on {@link #testDataset48()}, using tf.constant and iterating directly.
   */
  @Test
  public void testDataset52()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset52.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset52.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset52()}, utilizing only a single dataset. */
  @Test
  public void testDataset53()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset53.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with Datasets by using an explicit
   * iterator inside a shared function to retrieve the first element from two different datasets.
   */
  @Test
  public void testDataset54()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset54.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset54.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset54()}, utilizing only a single dataset. */
  @Test
  public void testDataset55()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset55.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case that exercises a potential 1-CFA precision problem with chained Datasets by sharing a
   * helper function that calls shuffle() on two different datasets.
   */
  @Test
  public void testDataset56()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset56.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset56.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset56()}, utilizing only a single dataset. */
  @Test
  public void testDataset57()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset57.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access version of {@link #testDataset56()}, utilizing chained Datasets (shuffle)
   * directly in the script without a shared helper function. This should NOT suffer from 1-CFA
   * precision merging.
   */
  @Test
  public void testDataset58()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset58.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset58.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset58()}, utilizing only a single dataset. */
  @Test
  public void testDataset59()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset59.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Direct access version of {@link #testDataset54()}, utilizing an explicit iterator directly in
   * the script without a shared helper function.
   */
  @Test
  public void testDataset60()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset60.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
    test("tf2_test_dataset60.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_FLOAT32)));
  }

  /** Control test for {@link #testDataset60()}, utilizing only a single dataset. */
  @Test
  public void testDataset61()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset61.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.from_tensors} inside a
   * shared wrapper function.
   */
  @Test
  public void testDataset62()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset62.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
    test("tf2_test_dataset62.py", "g", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }

  /** Control test for {@link #testDataset62()}, utilizing only a single dataset. */
  @Test
  public void testDataset63()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset63.py", "f", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 3))));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.range} inside a shared
   * wrapper function.
   */
  @Test
  public void testDataset64()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // The range generator always produces int64, but the 1-CFA merge occurs anyway,
    // though the type sets will both be {int64}. We verify it runs.
    test("tf2_test_dataset64.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
    test("tf2_test_dataset64.py", "g", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /** Control test for {@link #testDataset64()}, utilizing only a single dataset. */
  @Test
  public void testDataset65()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset65.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT64)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code shuffle} inside a shared wrapper
   * function, varying shapes instead of dtypes.
   */
  @Test
  public void testDataset66()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset66.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset66.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset66()}, utilizing only a single dataset. */
  @Test
  public void testDataset67()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset67.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test case exposing 1-CFA precision problem using {@code tf.data.Dataset.from_tensors} inside a
   * shared wrapper function, varying shapes instead of dtypes.
   */
  @Test
  public void testDataset68()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset68.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
    test("tf2_test_dataset68.py", "g", 1, 1, Map.of(2, Set.of(TENSOR_2_2_INT32)));
  }

  /** Control test for {@link #testDataset68()}, utilizing only a single dataset. */
  @Test
  public void testDataset69()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset69.py", "f", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * Test {@code tf.data.Dataset.from_tensor_slices} with a tuple of two Python list literals,
   * followed by the same {@code .shuffle(...).batch(...)} chain as {@link #testDataset5()}. The
   * iterator yields a tuple {@code (element_a, element_b)} where each element has the same batched
   * shape as {@code testDataset5}'s single element. This isolates the "tuple-structured argument"
   * path from the {@link #testDataset5()} "single-list" path; every test that feeds a tuple to
   * {@code from_tensor_slices} today does so via {@code mnist.load_data()}'s ndarray split, and no
   * existing test exercises the tuple-of-literals case in isolation.
   *
   * <p>The root-cause tuple walk was fixed on the feature branch for wala/ML#366 via a call-site
   * helper in {@link com.ibm.wala.cast.python.ml.client.DatasetFromTensorSlicesGenerator}, but this
   * test still fails because of a separate "chain-preservation gap": the {@code .shuffle().batch()}
   * chain wraps the dataset in generators that don't implement {@code DelegatingTensorGenerator},
   * so the factory's {@code PythonPropertyRead} dispatch for the {@code for element_a, element_b in
   * dataset:} unpack cannot peel back to find the tuple structure, and instead wraps in {@code
   * TensorElementGenerator} which over-peels one dim — collapsing the batched shape {@code (2,)}
   * back to {@code ()}.
   */
  @Test
  public void testDataset70()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset70.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  /**
   * Like {@link #testDataset70()} but the two elements of the tuple are {@code tf.constant(...)}
   * calls rather than raw Python list literals. Used as a comparison point to isolate whether the
   * tuple-structured-argument bug observed in {@link #testDataset70()} is specific to raw literals
   * or also applies when the tuple elements are already typed tensors. Both fail with identical
   * wrong output; see {@link #testDataset70()}'s Javadoc for the current root-cause status
   * (wala/ML#366 root-cause walk fixed, chain-preservation gap remains).
   */
  @Test
  public void testDataset71()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset71.py",
        "add",
        2,
        3,
        Map.of(
            2, Set.of(TENSOR_2_INT32, TENSOR_1_INT32), 3, Set.of(TENSOR_2_INT32, TENSOR_1_INT32)));
  }

  /**
   * Test a dataset created with {@code tf.data.Dataset.from_generator} using a <em>dict</em>
   * -structured legacy {@code output_types} argument (https://github.com/wala/ML/issues/615). The
   * dtype specification is a mapping, not a scalar {@code tf.DType} or a tuple/list of them, so the
   * dtype helper must recurse into the dict's values rather than asserting a single {@code DType}.
   */
  @Test
  public void testDataset72()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset72.py", "consume", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression guard for wala/ML#506: {@code tf.data.Dataset.map(map_func)} types its elements from
   * {@code map_func}'s return, not the receiver's elements. {@code map_func} here ({@code double})
   * consumes its argument ({@code tf.cast(x, tf.int64)}), exercising both halves: the callback's
   * parameter resolves to the upstream element type ({@code (4,)} float32, so the cast's shape is
   * {@code (4,)}) and its return ({@code (4,)} int64) becomes the mapped element type, recovered at
   * {@code consume}.
   */
  @Test
  public void testDatasetMap()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset_map.py", "consume", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Companion to {@link #testDatasetMap()} for a tuple-returning {@code map_func}: {@code split}
   * returns {@code (a, b)}, and iterating {@code for x, y in ds} resolves {@code y} (the second
   * component) to {@code (4,)} int64 via the mapped dataset's per-index tuple typing. wala/ML#506.
   */
  @Test
  public void testDatasetMapTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_tuple.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Like {@link #testDatasetMapTuple()} but iterated with {@code enumerate} and nested unpacking
   * ({@code for i, (x, y) in enumerate(ds)}) — the gpt-2 {@code fit}-loop shape (wala/ML#618). The
   * mapped tuple element's second component {@code y} still resolves to {@code (4,)} int64.
   * wala/ML#506.
   */
  @Test
  public void testDatasetMapEnumerate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_enumerate.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Regression guard for wala/ML#648: a {@code tf.data.Dataset} stored in a list and read back
   * ({@code [ds, ds][0]}) keeps its element type when iterated. The element lowers to {@code
   * d[iterator]}, a property read whose receiver came from a list getfield, so the def-chain alone
   * resolves the list, not the dataset; the fix recovers the dataset from the element at the
   * constant index and seeds the read off the receiver's points-to set. {@code consume}'s parameter
   * types to the sliced element {@code (4,)} float32.
   */
  @Test
  public void testDatasetInList()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_in_list.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * Regression guard for wala/ML#649: a pass-through transform after {@code map} keeps the mapped
   * element type. {@code map(split).repeat(2)} yields the same {@code (int32, int64)} tuple
   * elements as {@code map(split)}, so {@code y} types to {@code (4,)} int64. Before the fix,
   * {@code repeat}'s receiver-inheritance resolved the {@code map} receiver to a plain {@code
   * DatasetGenerator} (inheriting the upstream base), dropping {@code map_func}'s return; it now
   * resolves to a {@code DatasetMapGenerator} reading the {@code element} field off the receiver
   * instance.
   */
  @Test
  public void testDatasetMapRepeat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_map_repeat.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 4))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a>: a {@code
   * tf.data} dataset element passed to a function types its parameter. {@code target_convert}'s
   * {@code targets} receives a dataset element, so it types to {@code (2,)} int32 rather than being
   * missed.
   */
  @Test
  public void testDatasetElementParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_dataset_element_param.py",
        "target_convert",
        1,
        2,
        Map.of(2, Set.of(TensorType.of(INT_32, 2))));
  }

  /**
   * Verifies dtype recovery for {@code tf.linalg.matmul} on a {@code NamedTuple} field threaded in
   * as a parameter (<a href="https://github.com/wala/ML/issues/570">wala/ML#570</a>). {@code Inp}
   * is constructed in the caller and passed into {@code layer}, which reads {@code inp.x} (a {@code
   * NamedTuple} field) and feeds it to {@code matmul}; {@code consume(h)} pins the result. The
   * field read has no points-to set at the read site (the {@code NamedTuple} was built in the
   * caller), so the matmul input's dtype is recovered by reading the field off the object's
   * instance in the heap &mdash; the minimal form of the {@code gcn_proj} {@code
   * GraphConvolution.call} inner chain.
   *
   * <p>Before the fix, the field read's empty points-to set left the matmul input ⊤ on both axes;
   * now the dtype is recovered from the instance field and, for this single-call shape, the
   * existing shape machinery resolves the rest, so {@code consume}'s parameter is the concrete
   * {@code (4, 4) float32}. (The harder multi-call/chained case &mdash; {@code gcn_proj}'s layer
   * outputs &mdash; still needs forward-result propagation; see wala/ML#570.)
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeTupleReturnUnpack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_tuple_return_unpack.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  @Test
  public void testCollectionProbeLayerTupleReturn()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_layer_tuple_return.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  @Test
  public void testCollectionProbeListAppendIterate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_list_append_iterate.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Probe for the bare generator/next transit: a tensor yielded by a generator function, obtained
   * via {@code next()} with tuple unpacking, flows directly to {@code consume} with no model in
   * between; a regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>, where the generator transit
   * dropped the typing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextUnpack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_unpack.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Narrowing probe for the generator transit: the generator yields a single tensor (no tuple),
   * retrieved via {@code next()} with no unpacking. The minimal failing shape of <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>: neither tuple unpacking nor a
   * model forward is involved. The minimal shape of <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>, where the generator transit
   * dropped the typing; now a regression guard for it.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextSingle()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_single.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the transit through an explicit {@code iter()} call: {@code next(iter(gen()))}.
   * {@code iter()} was modeled as a fresh, empty {@code iterator} allocation, so {@code next}'s
   * read of the generator content field landed on the empty iterator instead of the generator
   * object and the yielded tensor's type was dropped. A regression guard for <a
   * href="https://github.com/wala/ML/issues/698">wala/ML#698</a>, where {@code iter} is modeled as
   * a pass-through of its argument.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextIter()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_iter.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probe for the two-argument {@code next(it, default)} form: the iterator is an empty generator,
   * so at runtime {@code next} returns the default and its type must reach the result. The
   * default's flow was dropped because the summary read only the iterator's generator content
   * field. A regression guard for <a href="https://github.com/wala/ML/issues/699">wala/ML#699</a>,
   * where the default (arg 3) is unioned into the result through a reachable join.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenNextDefault()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_next_default.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Companion probe for the generator transit: the same yielded pair consumed by for-loop
   * destructuring over the generator instead of {@code next()}. Also dropped (<a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>); distinct from the {@code
   * tf.data} destructuring shape of <a
   * href="https://github.com/wala/ML/issues/396">wala/ML#396</a>, where the producer is modeled and
   * the symptom is swapped element types. A regression guard for <a
   * href="https://github.com/wala/ML/issues/696">wala/ML#696</a>, where the generator transit
   * dropped the typing.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testGenForUnpack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_gen_for_unpack.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/668">wala/ML#668</a>: appending
   * a constant (an invariant-contents value whose pointer key the invoke's own argument processing
   * records as implicitly represented) must not crash call-graph construction with {@code
   * UnimplementedError}. The tensor appended alongside still types through the iteration.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeListAppendConstant()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_append_constant.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Probes the wala/ML#570 residual: a list accumulated with {@code append} in a loop feeds {@code
   * tf.concat} (mirroring {@code MessagePassing._calculate_messages_all_type} feeding {@code
   * _aggregate_function}). The appended values' shapes and dtype survive the list: the result keeps
   * the rank and non-axis dims with a dynamic axis dim (the element count is not statically known)
   * and the float32 dtype.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeListAppendConcat()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_append_concat.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(8))))));
  }

  @Test
  public void testCollectionProbeZipIterate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zip_iterate.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  @Test
  public void testCollectionProbeListElemSlice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_elem_slice.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_32, 4))));
  }

  @Test
  public void testCollectionProbeListLiteralIterate()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_list_literal_iterate.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  @Test
  public void testCollectionProbeZipDiag()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_zip_diag.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_8_FLOAT32)));
  }

  /**
   * Pins the gpt-2 decoder-stack shape in miniature (wala/ML#618): layers built by a list
   * comprehension, iterated with {@code zip} against a {@code [None] * n} list, each call's tuple
   * result destructured and the hidden state carried through the loop. The runtime {@code (4, 4)}
   * resolves; the ⊤-shape member is the loop-carried hidden state's unknown remainder, which the
   * exact operand reads surface instead of silently dropping (wala/ML#716, wala/ML#718) — the
   * loop's later iterations consume the stack's own unresolved output, so a fully resolved union
   * cannot be claimed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeLayerListComprehension()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_layer_list_comprehension.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32, TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code self.add_weight(...)} (wala/ML#618): the Keras weight-creation API, called from the
   * lazily-invoked {@code build} (wala/ML#595), creates a tensor whose shape and dtype come from
   * the call's {@code shape} list and {@code dtype} string arguments (wala/ML#667), so the matmul
   * against it composes to {@code (4, 4)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeAddWeight()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_add_weight.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored gpt-2 {@code EmbeddingLayer} forward result (wala/ML#618): the {@code
   * add_weight}-created weight dispatches and both {@code mode} branches contribute to the result
   * union. With {@code add_weight} consuming its {@code shape}/{@code dtype} arguments
   * (wala/ML#667), the embedding-mode member is fully concrete: {@code (2, 3, 8)} float32.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeVendoredEmbedding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    // Count-only: the parameter's type is a union across the `mode` branches whose exact
    // members shift with modeling precision.
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py",
          "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/probe_embedding.py"
        },
        "probe_embedding.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3, 8),
                // The one-hot member's leading dims fold to the runtime-true (2, 3) through the
                // tf.shape shape-vector arm (wala/ML#722).
                TensorType.of(INT_32, 2, 3, 10))));
  }

  /**
   * Pins the model self-call (wala/ML#618): a method calling {@code self(...)} and destructuring
   * the tuple result, mirroring gpt-2's {@code predictions, _ = self(inputs, training=True)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeModelSelfCall()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_model_self_call.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored gpt-2 forward output (the wala/ML#618 {@code pred} source): with wala/ML#665
   * forwarding wildcard import bindings, the full decoder stack types and the model output is a
   * rank-3-dominated tensor union. Receiver-keyed trampoline contexts (wala/ML#679) removed the
   * spurious {@code (?, ?, 4)}/{@code (?, ?, 12)} constructor-collapse members, and the wala/ML#739
   * operand-walk repairs with parameter defaults materializing (wala/ML#743) recover the
   * runtime-true logits member fully concrete: {@code (2, 3, 10)} float32, alongside the {@code (?,
   * ?, 10)} partial from the fit-path contexts. The wala/ML#680 {@code unknown}-dtype phantom is
   * gone: with the decoder-stack output resolving, {@code OutputLayer.call}'s dead {@code
   * self.porj_weights} arm no longer contributes a member.
   *
   * <p>TODO: Drop the {@code (2, 3, 8, 8)} member once <a
   * href="https://github.com/wala/ML/issues/746">wala/ML#746</a> filters constant-decidable branch
   * arms per call site; it is the {@code mode="projection"} call's rank-3 input crossing into the
   * embedding-mode lookup, runtime-infeasible at that site.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeVendoredForward()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_forward.py"
        },
        "probe_forward.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3, 10),
                TensorType.of(FLOAT_32, 2, 3, 8, 8),
                new TensorType(
                    FLOAT_32,
                    asList(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE, new NumericDim(10))))));
  }

  /**
   * Pins a tensor computed inside a {@code with tf.name_scope(...)} block (wala/ML#618): the
   * unresolved context manager does not perturb the body's dataflow.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeWithNameScope()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_with_name_scope.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Pins the vendored gpt-2 {@code Conv1d} forward result: {@code tf} arrives through the wildcard
   * import (wala/ML#665) and the {@code add_weight}-built kernel dispatches, so the result is
   * tensor-classified.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeVendoredConv1d()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "gpt2_vendored/layers/__init__.py", "gpt2_vendored/layers/embedding_layer.py",
          "gpt2_vendored/layers/feed_forward.py", "gpt2_vendored/layers/layer_norm.py",
          "gpt2_vendored/layers/attention_layer.py", "gpt2_vendored/utils/__init__.py",
          "gpt2_vendored/utils/tf_utils.py", "gpt2_vendored/scripts/__init__.py",
          "gpt2_vendored/scripts/utils.py", "gpt2_vendored/data_pipeline.py",
          "gpt2_vendored/A.py", "gpt2_vendored/probe_conv1d.py"
        },
        "probe_conv1d.py",
        "consume",
        "gpt2_vendored",
        1,
        1,
        // The computed output shape is opaque (a runtime-built list), so the reshape result is
        // pinned at unknown rank (wala/ML#703); receiver-keyed contexts (wala/ML#679) recover the
        // `add_weight` float32 dtype.
        Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins {@code tf.reshape} with a literal shape list (companion to {@link
   * #testCollectionProbeReshapeComputedShape()}).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeReshapeLiteralShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_computed_shape.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT32, 6, 8))));
  }

  /**
   * Pins {@code tf.reshape} with a shape list built at runtime by list concatenation ({@code
   * [tf.shape(x)[0], tf.shape(x)[1]] + [n]}, the vendored {@code Conv1d} idiom, wala/ML#618): the
   * interpreter evaluates the expression and the output shape is concrete.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeReshapeComputedShape()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reshape_computed_shape.py",
        "consume2",
        1,
        1,
        // The interpreter evaluates the concatenated shape expression to the concrete
        // (2, 3, 16); the opaque-shape-operand unknown-rank pin (wala/ML#703) rides along in the
        // union. TODO(https://github.com/wala/ML/issues/703): drop the ⊤ member once the pin
        // defers to interpreter-resolved shape operands.
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 3, 16), TENSOR_UNKNOWN_SHAPE_FLOAT32)));
  }

  /**
   * Pins wala/ML#665: {@code tf} reached through {@code from helpers import *} binds, matching
   * Python's wildcard semantics (every public module-level name is exported, including modules the
   * source module imported).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeWildcardTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {"wildcard_proj/helpers.py", "wildcard_proj/tf2_test_wildcard_tf.py"},
        "tf2_test_wildcard_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Companion to {@link #testCollectionProbeWildcardTf()} with a package-qualified wildcard source
   * ({@code from pkg.helpers2 import *}), the vendored {@code feed_forward.py} form (wala/ML#665).
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testCollectionProbeWildcardPkgTf()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "wildcard_proj/pkg/__init__.py",
          "wildcard_proj/pkg/helpers2.py",
          "wildcard_proj/tf2_test_wildcard_pkg_tf.py"
        },
        "tf2_test_wildcard_pkg_tf.py",
        "consume",
        "wildcard_proj",
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  /**
   * Lock-in test for wala/ML#456: {@code tf.data.Dataset.reduce(initial_state, reduce_func)}
   * returns a tensor with the same shape/dtype as {@code initial_state}. Pre-fix the {@code
   * tensorflow/data/reduce.do()} XML called {@code read_data} virtually on its receiver, but the
   * {@code reduce} class didn't define {@code read_data} — the call was unresolved and {@code
   * def="xx"} bound to nothing. Pre/post-fix this test produces the same observable type ({@code
   * [{[] of int32}]}) for {@code f}'s parameter at {@code vn=2} because the initial-state's tensor
   * type still propagates via PA assignment edges; the test serves as a regression lock so a future
   * XML/PA refactor that breaks the propagation surfaces here instead of regressing silently. The
   * XML cleanup itself is hygiene — it removes the unresolved virtual call and aligns the model
   * with TF runtime semantics ({@code reduce(...)} returns {@code initial_state}'s shape/dtype).
   */
  @Test
  public void testDatasetReduce()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_dataset_reduce.py", "f", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * Regression test for wala/ML#452: iterating a {@code tf.data.TextLineDataset} via {@code for
   * element in dataset:} should classify each element as a 0-D string tensor. The receiving
   * function {@code func}'s parameter at {@code vn=2} must therefore have type {@code
   * SCALAR_TENSOR_OF_STRING}. Pre-fix, the analyzer's {@code TextLineDataset} model didn't preserve
   * the per-element tensor type through the iteration substrate, leaving {@code func} with no
   * tensor classification at all (downstream {@code Function.getHasTensorParameter()} reported
   * false).
   */
  @Test
  public void testTextLineDatasetIterationElementType()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_textlinedataset_iter.py",
        "func",
        1,
        1,
        Map.of(2, Set.of(SCALAR_TENSOR_OF_STRING)));
  }

  /**
   * Regression guard for wala/ML#655: a {@code tf.io.FixedLenFeature} value, parsed by {@code
   * tf.io.parse_single_example} and read back through a dict subscript, types as a dense tensor
   * whose shape comes from the feature's {@code dims} argument and whose dtype comes from its
   * {@code type} argument. Previously {@code FixedLenFeature.do} allocated an {@code
   * Ltensorflow/objects/ feature} that the manual tensor walker ignores, so the parsed value never
   * typed.
   */
  @Test
  public void testFixedLenFeature()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fixed_len_feature.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Regression guard for wala/ML#655, end to end: the NLPGNN {@code TFLoader.load_valid} input
   * pipeline — a {@code TFRecordDataset} mapped by a {@code parse_single_example} decoder returning
   * a 4-tuple of {@code FixedLenFeature} dict-subscripts, then {@code prefetch}ed, then iterated
   * with a 4-way tuple unpack — types its first parsed field {@code X} ({@code input_ids}) to
   * {@code (128,)} int64. Before the {@code FixedLenFeature} fix, the mapped element was
   * non-tensor, so {@code X} (and any model parameter fed from it) came back non-tensor.
   */
  @Test
  public void testTfrecordLoader()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfrecord_loader.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(INT_64, 128))));
  }

  /**
   * Probe for <a href="https://github.com/wala/ML/issues/688">wala/ML#688</a>: a {@code map} stage
   * returning a tuple, batched with a tuple {@code padded_shapes}, iterated with destructuring —
   * the vendored gpt-2 {@code input_fn} element shape in miniature. Both halves type the
   * runtime-true {@code (4, 3) int64} (batch 4, padded to the longest sequence): the computed
   * second member resolves through the wala/ML#688 SSA fallback and the batch stage applies.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPaddedBatchPair()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_padded_batch_pair.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(DType.INT64, 4, 3))));
  }

  /**
   * Sibling half of {@link #testPaddedBatchPair()} (wala/ML#688): the second tuple member.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testPaddedBatchPair2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_padded_batch_pair.py",
        "consume2",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(DType.INT64, 4, 3))));
  }

  /**
   * Pins the <a href="https://github.com/wala/ML/issues/618">wala/ML#618</a> {@code
   * TuckERLoader.target_convert} row on the vendored subject: the loader is vendored verbatim from
   * {@code kyzhouhzau/NLPGNN} ({@code nlpgnn/datas/graphloader.py}); the driver, the tiny {@code
   * data/} triple files, and the {@code nlpgnn/gnn/utils.py} reachable slice are bespoke. The
   * {@code targets} parameter (a {@code padded_batch} dict-element field) types {@code (2, ?)}
   * int32 — the declared {@code padded_shapes} dims under the batch dimension (<a
   * href="https://github.com/wala/ML/issues/673">wala/ML#673</a>) — unioned with the standard
   * partial-batch sibling {@code (?, ?)}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   *     <p>The local count excludes the {@code enumerate(targets)} iterator object, pinned
   *     non-tensor since wala/ML#732; the element and its TensorFlow uses stay typed.
   */
  @Test
  public void testTuckerTargetConvert()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        new String[] {
          "tucker_proj/nlpgnn/__init__.py",
          "tucker_proj/nlpgnn/gnn/__init__.py",
          "tucker_proj/nlpgnn/gnn/utils.py",
          "tucker_proj/nlpgnn/datas/__init__.py",
          "tucker_proj/nlpgnn/datas/graphloader.py",
          "tucker_proj/tf2_test_tucker_target_convert.py"
        },
        "nlpgnn/datas/graphloader.py",
        "TuckERLoader.target_convert",
        "tucker_proj",
        1,
        6,
        Map.of(
            3,
            Set.of(
                new TensorType(INT_32, asList(new NumericDim(2), DynamicDim.INSTANCE)),
                new TensorType(INT_32, asList(new SymbolicDim("?"), DynamicDim.INSTANCE)))));
  }

  /**
   * Regression guard for <a href="https://github.com/wala/ML/issues/623">wala/ML#623</a>: a {@code
   * padded_batch} element threaded through a custom {@code fit} into {@code train_step} types the
   * parameters. {@code padded_batch} was unmodeled, so the per-element tensor type was dropped
   * before reaching the callee; modeling it like {@code batch} recovers it. The two parameters type
   * to {@code (2, 2)} int32 (the batch dimension prepended to the {@code (2,)} element).
   */
  @Test
  public void testPaddedBatchParam()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    TensorType t = TensorType.of(INT_32, 2, 2);

    test(
        "tf2_test_padded_batch_param.py",
        "Model.train_step",
        2,
        3,
        Map.of(3, Set.of(t), 4, Set.of(t)));
  }

  /**
   * Regression guard for wala/ML#645: {@code tf.io.VarLenFeature(dtype)} models the SparseTensor a
   * variable-length feature parses to, so {@code tf.sparse.to_dense} of it types from the feature's
   * dtype ({@code int64}) and the API-contract shape (rank-1 with a dynamic length, {@code (?,)}).
   * The {@code io}-registration fix makes {@code tf.io.*} resolve at all (they were registered
   * under {@code tf}, not the {@code io} object). The rank-1 dynamic shape is the contract-model
   * refinement of wala/ML#647 (formerly ⊤).
   */
  @Test
  public void testVarLenFeature()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_var_len_feature.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_DYNAMIC_INT64)));
  }

  /**
   * The realistic gpt-2 shape for wala/ML#645: a {@code VarLenFeature} in a feature dict, parsed by
   * {@code tf.io.parse_single_example}, subscripted, and densified. {@code consume}'s parameter
   * types to {@code (?,)} int64: the VarLenFeature SparseTensor now keeps a live points-to set
   * through the dict {@code putfield}/{@code getfield} (wala/ML#646), and {@code
   * tf.sparse.to_dense} resolves the dict-routed operand by dispatching the {@code VarLenFeature}
   * generator at the SparseTensor's allocation site, recovering the feature's dtype and the rank-1
   * dynamic (contract) shape. The static shape is {@code (?,)}, not the concrete {@code (2,)} the
   * Python runtime produces, because the example's length is lost across the serialize/parse
   * round-trip. This is the dict-routed companion to the direct {@link #testVarLenFeature()};
   * together they un-strand {@code get_loss}'s {@code real} in {@link #testGpt2GetLossVendored()}.
   */
  @Test
  public void testParseSingleExample()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_single_example.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TENSOR_DYNAMIC_INT64)));
  }

  /**
   * Mirrors gpt-2's {@code parse_example} for wala/ML#618: a tuple return over two {@code
   * tf.cast(tf.sparse.to_dense(parsed[k]), tf.int32)} values, each parsed from a {@code
   * VarLenFeature} in a feature dict, but called directly (no {@code dataset.map}). The recovered
   * {@code (?,)} int64 propagates through {@code to_dense}, the int32 cast, the tuple return, and
   * the destructuring, so {@code consume}'s parameter ({@code targets}) types to {@code (?,)}
   * int32. Together with {@link #testParseSingleExample()} this isolates the {@code dataset.map}
   * element-type layer as the sole remaining gap for the full {@link #testGpt2GetLossVendored()}
   * subject.
   */
  @Test
  public void testParseExampleTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_parse_example_tuple.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Regression guard for wala/ML#618: {@code tf.data.TFRecordDataset(...)} is a chainable dataset,
   * so {@code .map(parse_example)} resolves and the VarLenFeature-parsed {@code targets} types to
   * {@code (?,)} int32. Previously {@code TFRecordDataset} was a bare {@code Dataset} field with no
   * {@code do()}, so the chain did not resolve.
   */
  @Test
  public void testTfrecordMap()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfrecord_map.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, asList(DynamicDim.INSTANCE)))));
  }

  /**
   * Generator test for {@code tf.keras.datasets.fashion_mnist.load_data()}. Shapes and dtype are
   * identical to {@code mnist.load_data()}. The fixture passes all four unpacked arrays ({@code
   * x_train}, {@code y_train}, {@code x_test}, {@code y_test}) into the 4-arg sink, so the
   * assertion pins types at {@code vn=2..5}.
   */
  @Test
  public void testFashionMnistLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_fashion_mnist_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_60000_28_28_UINT8),
            3, Set.of(TENSOR_60000_UINT8),
            4, Set.of(TENSOR_10000_28_28_UINT8),
            5, Set.of(TENSOR_10000_UINT8)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.cifar100.load_data()}. Shapes are identical to
   * {@code cifar10.load_data()}, but the {@code y_train} / {@code y_test} dtype is {@code int64}
   * (cifar100's class indices) rather than {@code uint8} (cifar10's class indices). The dispatch
   * routes through the dedicated {@link com.ibm.wala.cast.python.ml.client.Cifar100InputData}
   * generator (closes wala/ML#487's mis-routing through {@code Cifar10InputData}). Asserts on all
   * four unpacked arrays at {@code vn=2..5}.
   */
  @Test
  public void testCifar100LoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_cifar100_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_50000_32_32_3_UINT8),
            3, Set.of(TENSOR_50000_1_INT64),
            4, Set.of(TENSOR_10000_32_32_3_UINT8),
            5, Set.of(TENSOR_10000_1_INT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.reuters.load_data()}. Asserts on all four unpacked
   * arrays at {@code vn=2..5}: {@code x_train} ({@code (8982,)} {@code object} &mdash; newswires
   * are variable-length integer-encoded sequences, so numpy stores them in an {@code object}
   * array), {@code y_train} ({@code (8982,)} {@code int64}), {@code x_test} ({@code (2246,)} {@code
   * object}), {@code y_test} ({@code (2246,)} {@code int64}). The {@code object} dtype matches the
   * runtime truth the Python fixture asserts (wala/ML#488).
   */
  @Test
  public void testReutersLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_reuters_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_8982_OBJECT),
            3, Set.of(TENSOR_8982_INT64),
            4, Set.of(TENSOR_2246_OBJECT),
            5, Set.of(TENSOR_2246_INT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.boston_housing.load_data()}. Asserts on all four
   * unpacked arrays at {@code vn=2..5}: {@code x_train} ({@code (404, 13)} {@code float64}
   * features), {@code y_train} ({@code (404,)} {@code float64} regression targets), {@code x_test}
   * ({@code (102, 13)} {@code float64}), {@code y_test} ({@code (102,)} {@code float64}).
   */
  @Test
  public void testBostonHousingLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_boston_housing_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_404_13_FLOAT64),
            3, Set.of(TENSOR_404_FLOAT64),
            4, Set.of(TENSOR_102_13_FLOAT64),
            5, Set.of(TENSOR_102_FLOAT64)));
  }

  /**
   * Generator test for {@code tf.keras.datasets.imdb.load_data()}. The four returned arrays each
   * have shape {@code (25000,)}: the {@code x_train} / {@code x_test} arrays carry numpy {@code
   * object} dtype (variable-length integer-encoded sequences, so numpy stores them in an {@code
   * object} array); the {@code y_train} / {@code y_test} arrays have dtype {@code int64} (binary
   * labels). Asserts on all four unpacked arrays at {@code vn=2..5}. The {@code object} dtype
   * matches the runtime truth the Python fixture asserts (wala/ML#488).
   */
  @Test
  public void testImdbLoadData()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_imdb_load_data.py",
        "f",
        4,
        4,
        Map.of(
            2, Set.of(TENSOR_25000_OBJECT),
            3, Set.of(TENSOR_25000_INT64),
            4, Set.of(TENSOR_25000_OBJECT),
            5, Set.of(TENSOR_25000_INT64)));
  }
}
