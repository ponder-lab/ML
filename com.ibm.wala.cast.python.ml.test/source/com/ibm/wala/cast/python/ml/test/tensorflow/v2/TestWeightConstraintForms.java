package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the other forms a weight constraint takes (<a
 * href="https://github.com/wala/ML/issues/996">wala/ML#996</a>): every {@code add_weight} argument
 * supplied positionally, and a function or an object constraint on {@code tf.Variable}.
 */
public class TestWeightConstraintForms extends AbstractTensorTest {

  private static final String FILE = "tf2_test_weight_constraint_forms.py";

  /** A function constraint passed positionally to {@code add_weight} receives the weight. */
  @Test
  public void testPositionalFunction()
      throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "norm_positional", 1, 2, Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /** A function constraint on {@code tf.Variable} receives the variable. */
  @Test
  public void testVariableFunction() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "norm_variable", 1, 2, Map.of(2, Set.of(TensorType.of(FLOAT_32, 6, 1))));
  }

  /** An object constraint on {@code tf.Variable} receives the variable in its {@code __call__}. */
  @Test
  public void testVariableObject() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "ObjNorm.__call__", 1, 2, Map.of(3, Set.of(TensorType.of(FLOAT_32, 7, 1))));
  }
}
