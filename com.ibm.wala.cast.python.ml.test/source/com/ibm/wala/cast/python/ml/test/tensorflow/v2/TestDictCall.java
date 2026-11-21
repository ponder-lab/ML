package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a dictionary built by a call of {@code dict} holds the call's values (wala/ML#1307).
 * Before, the call had no target and its result pointed to nothing, so a value read off the
 * dictionary was no tensor.
 */
public class TestDictCall extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dict_call.py";

  /** A keyword argument is held under its name. */
  @Test
  public void testKeywords() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_keywords", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_keywords_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** The control: a display's values are held, as they were before. */
  @Test
  public void testDisplay() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_display", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A pair of a literal list of pairs is held under its first member. */
  @Test
  public void testPairs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_pairs", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_pairs_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A list of pairs is read through the name it is bound to. */
  @Test
  public void testNamedPairs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_named_pairs", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A tuple of pairs is read as a list of pairs is. */
  @Test
  public void testTupleOfPairs() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_tuple_of_pairs", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** An entry of a dictionary argument is held under its own key. */
  @Test
  public void testCopy() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_copy", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_copy_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** An entry of an unpacked dictionary is held under its own key. */
  @Test
  public void testUnpacking() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_unpacking", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A dictionary argument and a keyword argument beside it are both held. */
  @Test
  public void testCopyAndKeyword() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_copy_and_keyword", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_copy_and_keyword_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A value read with {@code get} is the one held. */
  @Test
  public void testGet() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_keywords_get", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * Two calls build two dictionaries, so a value held under a name by one is not read off the other
   * under that name.
   */
  @Test
  public void testEachCallBuildsItsOwn()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_first_call", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_second_call", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A dictionary a function builds from its parameters and returns holds its arguments. */
  @Test
  public void testBuiltInFunction() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_function", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A dictionary built by a call with no argument holds what is stored in it afterwards. */
  @Test
  public void testEmptyThenStore() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_empty_then_store", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A pair's first member that is a constant passed through a parameter names the entry. */
  @Test
  public void testParameterKey() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_parameter_key", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** The pairs another dictionary's {@code items} gives are held under their keys. */
  @Test
  public void testItems() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_items", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A call of {@code dict} through another name builds a dictionary all the same. */
  @Test
  public void testAlias() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_alias", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A dictionary held as a keyword argument's value is read through the one that holds it. */
  @Test
  public void testNestedCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_nested_call", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /**
   * A dictionary built from one a call built holds its entries, whether that one took them from
   * keywords or from pairs.
   */
  @Test
  public void testCopyOfCall() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_copy_of_call", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_copy_of_pairs", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A display written as a keyword argument's value is read through the dictionary. */
  @Test
  public void testKeywordDisplay() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_keyword_display", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** A constant passed as a keyword argument's value is held, so a read keyed by it finds it. */
  @Test
  public void testConstantValue() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_constant_value", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
  }

  /** Each of two dictionaries unpacked with {@code **} gives its own entries. */
  @Test
  public void testTwoUnpackings() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "by_two_unpackings", 1, 1, Map.of(2, Set.of(TENSOR_2_4_FLOAT32)));
    test(FILE, "by_two_unpackings_other", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }
}
