package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a Python scalar computed from Python numbers ({@code n ** 0.5}) imposes no dtype on
 * the tensor it is combined with (<a href="https://github.com/wala/ML/issues/992">wala/ML#992</a>).
 * TensorFlow converts a Python number to the tensor's dtype, so the parameter-dtype coercion of
 * wala/ML#828 must leave the parameter at its fed dtype, {@code float32} here, where it reported
 * the scalar's own promoted {@code float64}. The test method is the entrypoint, so the fixture is
 * named as pytest finds it.
 */
public class TestScalarCoercion extends AbstractTensorTest {

  private static final String FILE = "scalar_coercion_test.py";

  private static final TensorType FED = TensorType.of(FLOAT_32, 4, 5, 10);

  /** The binary form, {@code inputs * n ** 0.5}: the parameter keeps its fed dtype. */
  @Test
  public void testScalarPowerKeepsFedDType()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale", 1, 3, Map.of(2, Set.of(FED)));
  }

  /** The augmented form, {@code inputs *= n ** 0.5}: the parameter keeps its fed dtype. */
  @Test
  public void testScalarPowerInPlaceKeepsFedDType()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale_in_place", 1, 2, Map.of(2, Set.of(FED)));
  }

  /** The product itself is {@code float32}, as it was before. */
  @Test
  public void testProductDType() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "sink", 1, 1, Map.of(2, Set.of(FED)));
  }

  /** Control: a float literal beside the parameter left it alone before as well. */
  @Test
  public void testFloatLiteralKeepsFedDType()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale_by_literal", 1, 2, Map.of(2, Set.of(FED)));
  }
}
