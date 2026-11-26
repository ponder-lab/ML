package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a configuration read with {@code pop} or {@code get} and a default, from a dictionary that
 * {@code yaml.safe_load} or {@code json.loads} returns (<a
 * href="https://github.com/wala/ML/issues/1339">wala/ML#1339</a>). The analysis does not read the
 * document, so a value read from it is unknown, not the default alone.
 */
public class TestUntrackedDict extends AbstractTensorTest {

  private static final String YAML_FIXTURE = "tf2_test_config_pop.py";

  private static final String JSON_FIXTURE = "tf2_test_config_json.py";

  /**
   * A branch on a flag the document may set is not decided by the flag's default, so the tensor the
   * branch passes reaches its consumer. Before this, the flag read as its default {@code False}
   * alone, and the consumer received nothing typed.
   */
  @Test
  public void testFlagFromDocument() throws ClassHierarchyException, CancelException, IOException {
    test(YAML_FIXTURE, "consume_flagged", 1, 1, Map.of(2, Set.of(TENSOR_4_3_FLOAT32)));
  }

  /**
   * A batch size the document may set is unknown, so the batch axis is too. Before this, it read as
   * its default {@code 1} alone.
   */
  @Test
  public void testBatchSizeFromDocument()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        YAML_FIXTURE,
        "consume_batch",
        1,
        1,
        Map.of(
            2, Set.of(new TensorType(FLOAT_32, asList(new SymbolicDim("?"), new NumericDim(3))))));
  }

  /**
   * A {@code get} on the value another {@code get} reads off a JSON document is unknown too, so the
   * size it gives is: the default {@code 2} when the document has no such key, or a size the
   * analysis does not know. Before this, it read as its default alone.
   */
  @Test
  public void testNestedGetFromDocument()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        JSON_FIXTURE,
        "consume_rows",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 3),
                new TensorType(FLOAT_32, asList(UnresolvedDim.INSTANCE, new NumericDim(3))))));
  }
}
