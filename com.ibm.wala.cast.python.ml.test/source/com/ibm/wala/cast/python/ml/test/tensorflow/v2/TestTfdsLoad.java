package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The elements of a dataset that TensorFlow Datasets' {@code tfds.load} returns are tensors, a pair
 * of them under {@code as_supervised=True} (<a
 * href="https://github.com/wala/ML/issues/1155">wala/ML#1155</a>). Their shapes and dtypes depend
 * on the named dataset's features, so each is a tensor of unknown shape and dtype. Each sink in
 * {@code tf2_test_tfds_load.py} receives a pair unpacked in a loop over such a dataset.
 */
public class TestTfdsLoad extends AbstractTensorTest {

  /** A tensor of unknown shape and dtype. */
  private static final TensorType UNKNOWN_TENSOR = new TensorType(UNKNOWN, null);

  /**
   * With no {@code split}, {@code tfds.load} returns a dictionary from split names to datasets,
   * here subscripted by {@code tfds.Split.TRAIN} and iterated through {@code enumerate}.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSupervised()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfds_load.py",
        "consume_supervised",
        2,
        2,
        Map.of(2, Set.of(UNKNOWN_TENSOR), 3, Set.of(UNKNOWN_TENSOR)));
  }

  /**
   * With a {@code split}, {@code tfds.load} returns the dataset itself.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testSplit()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfds_load.py",
        "consume_split",
        2,
        2,
        Map.of(2, Set.of(UNKNOWN_TENSOR), 3, Set.of(UNKNOWN_TENSOR)));
  }

  /**
   * With {@code with_info=True}, {@code tfds.load} returns a pair of its result and the dataset's
   * information.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testWithInfo()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfds_load.py",
        "consume_info",
        2,
        2,
        Map.of(2, Set.of(UNKNOWN_TENSOR), 3, Set.of(UNKNOWN_TENSOR)));
  }

  /**
   * Without {@code as_supervised}, an element is a dictionary of features, each a tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testFeatures()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfds_load.py",
        "consume_features",
        2,
        2,
        Map.of(2, Set.of(UNKNOWN_TENSOR), 3, Set.of(UNKNOWN_TENSOR)));
  }

  /**
   * A list {@code split} with {@code with_info=True}, unpacked as {@code (train, test), info}: the
   * first dataset.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testListedTrain()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfds_load.py",
        "consume_listed_train",
        2,
        2,
        Map.of(2, Set.of(UNKNOWN_TENSOR), 3, Set.of(UNKNOWN_TENSOR)));
  }

  /**
   * A list {@code split} with {@code with_info=True}, unpacked as {@code (train, test), info}: the
   * second dataset, which the information must not stand in for.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testListedTest()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "tf2_test_tfds_load.py",
        "consume_listed_test",
        2,
        2,
        Map.of(2, Set.of(UNKNOWN_TENSOR), 3, Set.of(UNKNOWN_TENSOR)));
  }
}
