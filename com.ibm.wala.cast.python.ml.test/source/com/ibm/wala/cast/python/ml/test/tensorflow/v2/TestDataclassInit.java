package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the initializer {@code @dataclass} generates (<a
 * href="https://github.com/wala/ML/issues/205">wala/ML#205</a>): a constructor call binds its
 * arguments to the class's annotated fields, positional ones in declaration order and keyword ones
 * by name, so a method reading a field through {@code self} reads the argument passed for it. Each
 * field is read into its own consumer.
 */
public class TestDataclassInit extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataclass_init.py";

  /** The first positional argument binds the first annotated field. */
  @Test
  public void testFirstPositional() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_first", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }

  /**
   * The second positional argument binds the second annotated field, not the first, so the two
   * fields keep their own shapes.
   */
  @Test
  public void testSecondPositional() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_second", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }

  /** A keyword argument binds the annotated field of its name, under {@code dataclass(...)}. */
  @Test
  public void testKeyword() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_keyword", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4))));
  }

  /**
   * The decorator imported under another name: the class is recognized by the attribute the
   * decorator sets, not by the decorator's name.
   */
  @Test
  public void testAliased() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_aliased", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 5))));
  }
}
