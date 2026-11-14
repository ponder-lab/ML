package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests subscripts of a tensor that a function returns in a tuple and its caller unpacks (<a
 * href="https://github.com/wala/ML/issues/1261">wala/ML#1261</a>). A subscript has an empty
 * points-to set, so an unpacked element is typed by the returned literal's element write, read in
 * the callee that allocates it. Each element is divided in its own function and the quotient passed
 * to its own consumer.
 */
public class TestReturnedTupleSubscripts extends AbstractTensorTest {

  private static final String FILE = "tf2_test_tuple_return_subscript_unpack.py";

  /** The first element unpacked: the parameter and its quotient are both typed. */
  @Test
  public void testUnpackedFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "normalize_first", 1, 2, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The quotient of the first element unpacked. */
  @Test
  public void testUnpackedFirstQuotient()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The second element unpacked, read at index {@code 1}. */
  @Test
  public void testUnpackedSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "normalize_second", 1, 2, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The quotient of the second element unpacked. */
  @Test
  public void testUnpackedSecondQuotient()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_second", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** A single subscript returned, not in a tuple: typed before wala/ML#1261 too. */
  @Test
  public void testReturnedOne() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "normalize_one", 1, 2, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** The quotient of a single subscript returned. */
  @Test
  public void testReturnedOneQuotient()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_one", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }
}
