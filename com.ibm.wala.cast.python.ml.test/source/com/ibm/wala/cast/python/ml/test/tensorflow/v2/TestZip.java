package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests tensors carried through Python's builtin {@code zip} inside functions: unpacked in a {@code
 * for} loop, subscripted off {@code list(zip(...))}, and transposed by {@code zip(*rows)} before a
 * concatenation. In the loop and subscript forms each position holds a tensor of its own dtype, so
 * a field read off the wrong position shows as an extra member.
 */
public class TestZip extends AbstractTensorTest {

  private static final String FILE = "tf2_test_zip.py";

  private static final TensorType TENSOR_1_60_2_FLOAT32 = TensorType.of(FLOAT32, 1, 60, 2);

  private static final TensorType TENSOR_1_60_2_INT32 = TensorType.of(INT32, 1, 60, 2);

  /**
   * A concatenation of {@code (1, 60, 2)} tensors along axis 1 over a list whose length the
   * analysis does not count: the runtime extent is a fixed {@code 120}, so the axis is {@link
   * UnresolvedDim} (wala/ML#721).
   */
  private static final TensorType TENSOR_1_UNRESOLVED_2_FLOAT32 =
      new TensorType(
          FLOAT_32, asList(new NumericDim(1), UnresolvedDim.INSTANCE, new NumericDim(2)));

  /** The first loop variable of {@code for p, q in zip(a, b)} is an element of {@code a}. */
  @Test
  public void testLoopFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_first", 1, 1, Map.of(2, Set.of(TENSOR_1_60_2_FLOAT32)));
  }

  /** The second loop variable of {@code for p, q in zip(a, b)} is an element of {@code b}. */
  @Test
  public void testLoopSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop_second", 1, 1, Map.of(2, Set.of(TENSOR_1_60_2_INT32)));
  }

  /** Field 0 of {@code list(zip(a, b))[0]} is an element of {@code a}. */
  @Test
  public void testIndexFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_index_first", 1, 1, Map.of(2, Set.of(TENSOR_1_60_2_FLOAT32)));
  }

  /** Field 1 of {@code list(zip(a, b))[0]} is an element of {@code b}. */
  @Test
  public void testIndexSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_index_second", 1, 1, Map.of(2, Set.of(TENSOR_1_60_2_INT32)));
  }

  /**
   * The first concatenation over {@code list(zip(*rows))} joins the rows' first fields along axis
   * 1. The rows are unpacked into {@code zip}'s arguments, and {@code list} copies a tuple {@code
   * zip} built, a list whose length the analysis does not count, so the extent along the axis is
   * unresolved where the runtime has {@code 120}.
   */
  @Test
  public void testStarFirst() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_star_first", 1, 1, Map.of(2, Set.of(TENSOR_1_UNRESOLVED_2_FLOAT32)));
  }

  /**
   * The second concatenation over {@code list(zip(*rows))} joins the rows' second fields along axis
   * 1.
   */
  @Test
  public void testStarSecond() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_star_second", 1, 1, Map.of(2, Set.of(TENSOR_1_UNRESOLVED_2_FLOAT32)));
  }

  /**
   * A loop over {@code list(y)} for a {@code (3, 2)} tensor {@code y} binds its rows, as a loop
   * over {@code y} does. The {@code list} summary copies the elements the pointer analysis sees,
   * but a tensor's rows are typed by the iteration hooks of {@code TensorGeneratorFactory}, which
   * see through {@code reversed(y)} alone, so the loop variable reaches no tensor.
   *
   * <p>TODO: Blocked on <a href="https://github.com/wala/ML/issues/1102">wala/ML#1102</a>.
   */
  @Test(expected = AssertionError.class)
  public void testListOfTensor() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_list_element", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT32, 2))));
  }

  /**
   * A loop over {@code list(d)} for a dict {@code d} binds its keys, strings here, so the consumer
   * is reached with no tensor, although the dict's values are tensors.
   */
  @Test
  public void testListOfDictKeys() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keys", 0, 0);
  }

  /** The second component of a loop over {@code list(d.items())} is one of the dict's values. */
  @Test
  public void testListOfDictItems() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_item_values",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT32, 2), TensorType.of(FLOAT32, 3))));
  }

  /** A loop over {@code list(s)} for a string {@code s} binds no tensor. */
  @Test
  public void testListOfString() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_string", 0, 0);
  }

  /** A loop over {@code list(range(n))} binds no tensor. */
  @Test
  public void testListOfRange() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_range", 0, 0);
  }
}
