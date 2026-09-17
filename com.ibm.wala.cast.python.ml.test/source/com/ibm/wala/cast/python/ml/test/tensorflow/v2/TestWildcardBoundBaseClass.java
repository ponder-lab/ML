package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A layer whose base class is written through a name a wildcard import binds (wala/ML#938), on
 * {@code addweight_proj}: {@code tf_utils.py} is {@code import tensorflow as tf} plus {@code from
 * tensorflow.keras.layers import Layer}; {@code block_wild.py} does {@code from tf_utils import *}
 * and defines {@code class BlockWild(tf.keras.layers.Layer)} whose {@code build} creates a weight
 * with {@code add_weight} and whose {@code call} passes it to a sink. Global reads through the
 * binding already resolved (wala/ML#665: the layer's {@code tf.matmul} dispatched), but the
 * class-definition-time base lookup did not, so the class never inherited the shell and its {@code
 * add_weight} had no target: the sink read no tensor parameter, where the identical layer with a
 * direct import in {@code block_direct.py} read {@code (4, 4) float32}.
 *
 * <p>{@code block_shadow.py} is the control against over-resolution: the same wildcard import also
 * binds the Keras {@code Layer}, and the module defines its own {@code class Layer} with a {@code
 * make} method. The local definition must win as the base of {@code BlockShadow}: {@code make}
 * resolves and reads {@code (3, 3) float32}, and {@code add_weight} does not exist on it.
 */
public class TestWildcardBoundBaseClass extends AbstractTensorTest {

  private static final String[] FILES = {
    "addweight_proj/tf_utils.py", "addweight_proj/block_wild.py",
    "addweight_proj/block_direct.py", "addweight_proj/block_shadow.py",
    "addweight_proj/driver.py"
  };

  private static final String PROJECT = "addweight_proj";

  /**
   * The same files with the wildcard source listed last, so the importer is parsed before the
   * module it imports from. The registry is filled as each module is parsed and read only when a
   * class is translated to IR, after every module has been parsed, so the parse order must not
   * change the answer; this arm is what would show it if it did.
   */
  private static final String[] SOURCE_LAST = {
    "addweight_proj/block_wild.py", "addweight_proj/block_direct.py",
    "addweight_proj/block_shadow.py", "addweight_proj/driver.py",
    "addweight_proj/tf_utils.py"
  };

  @Test
  public void testWildcardBoundBaseResolvesTheWeight()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "block_wild.py",
        "consume_wild",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  @Test
  public void testDirectImportBaseResolvesTheWeight()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "block_direct.py",
        "consume_plain",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
  }

  @Test
  public void testLocalDefinitionShadowsTheWildcardBinding()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "block_shadow.py",
        "consume_shadow",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
    test(FILES, "block_shadow.py", "consume_shadow_weight", PROJECT, 0, 0, Map.of());
  }

  @Test
  public void testWildcardBoundBaseResolvesWhateverTheParseOrder()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        SOURCE_LAST,
        "block_wild.py",
        "consume_wild",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TENSOR_4_4_FLOAT32)));
    test(
        SOURCE_LAST,
        "block_shadow.py",
        "consume_shadow",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 3))));
  }
}
