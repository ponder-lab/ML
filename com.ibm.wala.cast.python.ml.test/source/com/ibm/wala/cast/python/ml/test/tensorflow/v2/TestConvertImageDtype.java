package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code tf.image.convert_image_dtype} returns the image's shape in the requested dtype,
 * as {@code tf.cast} types its result. Unmodeled, its result was no tensor.
 */
public class TestConvertImageDtype extends AbstractTensorTest {

  private static final String FILE = "tf2_test_convert_image_dtype.py";

  /**
   * A uint8 image converted to float32.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testToFloat() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_float", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 8, 3))));
  }

  /**
   * The float32 image converted back to uint8, with {@code saturate} passed by keyword.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testBackToUint8() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_back", 1, 1, Map.of(2, Set.of(TensorType.of(UINT_8, 8, 8, 3))));
  }
}
