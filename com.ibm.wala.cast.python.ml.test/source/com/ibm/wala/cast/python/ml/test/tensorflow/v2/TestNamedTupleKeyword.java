package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a {@code collections.namedtuple} instance's field reads give the values its
 * constructor was passed, whether by keyword or by position, including a field read off a loop
 * variable that carries the instance through {@code tf.while_loop}.
 */
public class TestNamedTupleKeyword extends AbstractTensorTest {

  private static final String FILE = "tf2_test_namedtuple_keyword.py";

  /**
   * A field of an instance built by keyword.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testKeyword() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * A field of an instance built by position.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testPositional() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_positional", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * A field read off a loop variable holding an instance, reshaped to {@code (1, 1)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testLoopField() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_loop", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 1, 1))));
  }

  /**
   * A field of a type whose names are one string separated by spaces, its instance built by
   * position and keyword together.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testStringNames() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_string_x", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /**
   * A field read by its position, {@code point[1]}, bound by keyword.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testIndexed() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_indexed", 1, 1, Map.of(2, Set.of(SCALAR_TENSOR_OF_INT32)));
  }

  /**
   * A field unpacked from an instance, {@code first, _ = point}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnpacked() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_unpacked", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32))));
  }

  /**
   * A field at a position a starred argument spreads, {@code Point(x, *[y])}: the positions from
   * the starred argument on are not bound, so the field converts to a tensor of unknown type, never
   * the list the argument spreads, which would convert to a {@code (1, 3)} tensor where the
   * program's field is the {@code (3,)} element.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testStarred() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_starred", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A field given a list literal, {@code Point([t], y=0)}: the literal is bound to the field as
   * allocated, so subscripting the field reads the list's element.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testListLiteral() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_listed", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }
}
