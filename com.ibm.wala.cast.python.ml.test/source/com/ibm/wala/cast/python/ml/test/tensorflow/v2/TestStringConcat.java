package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a {@code +} whose operands are Python strings (<a
 * href="https://github.com/wala/ML/issues/1103">wala/ML#1103</a>): it is a string concatenation,
 * not a tensor operation, so its result has no tensor type. A path built from a directory literal
 * and a file name read from a file reaches the function it is passed to, through a generator, as no
 * tensor.
 */
public class TestStringConcat extends AbstractTensorTest {

  private static final String FILE = "tf2_test_string_concat.py";

  /**
   * The function a concatenated path is passed to receives no tensor: its only tensors are the two
   * arrays it builds. Before, the path read as a scalar string tensor.
   */
  @Test
  public void testConcatenatedPath() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "process_image_label", 0, 2);
  }

  /**
   * A control: a string tensor plus a Python string is a tensor operation, whose result is a scalar
   * string tensor.
   */
  @Test
  public void testTensorPlusString() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_suffixed", 1, 1, Map.of(2, Set.of(new TensorType(STRING, List.of()))));
  }
}
