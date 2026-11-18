package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the tensors made from the {@code element_spec} of a {@code map} over a file's lines that
 * several functions reach, on {@code tf2_test_element_spec_places.py} (<a
 * href="https://github.com/wala/ML/issues/1300">wala/ML#1300</a>). A read describes the dataset's
 * element once per place, whatever number of functions return a value there, and a tensor made from
 * a spec's shape or dtype still has the shape or dtype of the dataset's element at the spec's
 * place. {@link com.ibm.wala.cast.python.ml.test.TestSpecPlaceProducers} pins the descriptions.
 */
public class TestElementSpecPlaces extends AbstractTensorTest {

  private static final String FILE = "tf2_test_element_spec_places.py";

  /** A dict of specs and a spec are not tensors. */
  @Test
  public void testSpecIsNotATensor()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "consume_several_spec", 0, 0);
    test(FILE, "consume_several_leaf_spec", 0, 0);
  }

  /** The values of the dicts three functions return, read by key. */
  @Test
  public void testSeveralFunctions()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_several_shape", 1, 1, Map.of(2, Set.of(TENSOR_4_2_FLOAT32)));
    test(FILE, "use_several_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** The fields of the namedtuple instances two functions return, read by name. */
  @Test
  public void testNamedTuple()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_paired_shape", 1, 1, Map.of(2, Set.of(TENSOR_6_FLOAT32)));
    test(FILE, "use_paired_dtype", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /** A value three levels down the dicts two functions return. */
  @Test
  public void testNested()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_nested_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 9, 2))));
  }

  /** A value nine levels down, past the places a description records. */
  @Test
  public void testPastTheRecordedPlaces()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_deep_shape", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 7))));
  }

  /** One function. */
  @Test
  public void testOneFunction()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FILE, "use_alone_shape", 1, 1, Map.of(2, Set.of(TENSOR_3_5_FLOAT32)));
  }
}
