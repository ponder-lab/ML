package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * The members of a dataset's tuple element of five components, unpacked in a loop, keep their own
 * types, the fifth too: the element's copy holds every position, though the components an iterator
 * allocates for an element that is not copied stop at four (<a
 * href="https://github.com/wala/ML/issues/1221">wala/ML#1221</a>).
 */
public class TestDatasetWideTupleElement extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_wide_tuple_element.py";

  /**
   * The fifth member of a sliced five-tuple.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSlicedFifth() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_sliced_fifth", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 5))));
  }

  /**
   * The fifth member of a {@code from_generator} five-tuple declared by {@code output_types} alone:
   * the declared dtype, and the rank the generator's yields give.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testGeneratedFifth() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_generated_fifth",
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_32, List.of(UnresolvedDim.INSTANCE)))));
  }
}
