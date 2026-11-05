package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The element of a dataset that may be a tuple or a tensor, and a constant position of it, in a
 * loop over a dataset whose element is not copied (wala/ML#997): where the element is the tensor,
 * it is that tensor, and its position is a row of it.
 */
public class TestDatasetMaybeTupleMember extends AbstractTensorTest {

  private static final String MAP_FILE = "tf2_test_dataset_map_maybe_tuple_member.py";

  private static final String EITHER_FILE = "tf2_test_dataset_either_tuple_member.py";

  /**
   * The element of a map whose function returns {@code (x, y)} on one call and a tensor of shape
   * {@code (3,)} on another: the tensor, beside the pair's members, which are imprecision, since
   * the analysis cannot tell which call returns the pair.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMapElement() throws ClassHierarchyException, CancelException, IOException {
    test(
        MAP_FILE,
        "consume_element",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 3), TensorType.of(FLOAT_32, 4), TensorType.of(FLOAT_32))));
  }

  /**
   * Position 0 of the element of that map: a row of the tensor, a scalar, beside the pair's first
   * member, which is imprecision.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testMapMember() throws ClassHierarchyException, CancelException, IOException {
    test(
        MAP_FILE,
        "consume_member",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32), TensorType.of(FLOAT_32, 4))));
  }

  /**
   * Position 0 of the element of one of two datasets, of pairs and of tensors of shape {@code
   * (3,)}: the pair's first member or a row of the tensor.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testEitherDatasetMember()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        EITHER_FILE,
        "consume_member",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32), TensorType.of(FLOAT_32, 4))));
  }
}
