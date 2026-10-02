package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the model-level configuration round trip a model trained only through {@code fit} is
 * rebuilt by (<a href="https://github.com/wala/ML/issues/997">wala/ML#997</a>): {@code get_config}
 * stores the layers' serialized configurations under one key, deep-copied, and {@code from_config}
 * deep-copies the configuration, pops the layer configurations off it, rebuilds each layer in a
 * loop over their items, writes the rebuilt layers back with {@code update} and binds them with a
 * dict unpacking. Each fixture isolates one construct of that chain over a baseline written with
 * subscripts only, so each pins the construct's own modeling: {@code copy.deepcopy} and {@code
 * copy.copy} as aliases of their argument, and {@code pop}, {@code get}, {@code items} and {@code
 * update} as reads and writes of the fields a dictionary's constant keys name. The pinned value is
 * the weight the rebuilt model's constraint receives, which is absent when the chain is cut, since
 * the rebuilt model then has no layer to call.
 */
public class TestConfigChain extends AbstractTensorTest {
  /**
   * The rebuilt model's constraint sees the weight: the round trip written with subscripts only,
   * which the identities already carried.
   */
  @Test
  public void testBaseline() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_plain.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the stored layer configurations deep-copied, so
   * {@code copy.deepcopy} must alias its argument.
   */
  @Test
  public void testDeepCopiedStore() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_deepcopy_store.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the loaded configuration deep-copied before the
   * layers are rebuilt.
   */
  @Test
  public void testDeepCopiedLoad() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_deepcopy_load.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the layer configurations popped off the
   * configuration by a constant key.
   */
  @Test
  public void testPop() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_pop.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the layer configurations read off the
   * configuration by {@code get} with a constant key.
   */
  @Test
  public void testGet() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_get.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the layers rebuilt in a loop over the
   * configurations' items, so {@code items} must yield the values.
   */
  @Test
  public void testItems() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_items.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the layers rebuilt in a loop over items and
   * bound by a dict unpacking.
   */
  @Test
  public void testItemsStar() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_items_star.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the rebuilt layer written by {@code update} and
   * bound by a dict unpacking.
   */
  @Test
  public void testUpdateStar() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_update_star.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model's constraint sees the weight: the whole chain at once: deep copies on both
   * sides, {@code pop}, {@code items}, {@code update} and a dict unpacking.
   */
  @Test
  public void testFullChain() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_full.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * A receiver that is a dict on one path and a user object with its own {@code items} on the
   * other: the dict's fields are read and the user's method dispatches, so the rebuilt model sees
   * the layer from both paths.
   */
  @Test
  public void testUserItems() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_user_items.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /** The user's {@code items} method is reached beside the dict reading (present, no tensor). */
  @Test
  public void testUserItemsDispatches()
      throws ClassHierarchyException, CancelException, IOException {
    test("tf2_test_config_chain_user_items.py", "consume_registry", 0, 0);
  }

  /**
   * The chain with {@code get_config} and {@code from_config} inherited from a base model class,
   * the subclass's constructor forwarding to the base's through {@code super()} and the subclass
   * defining its own {@code call}: the base constructor's write of the kernel lands on the
   * instance, and the rebuilt model's constraint sees the weight.
   */
  @Test
  public void testInheritedChain() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_inherited.py",
        "consume_norm",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 3))));
  }

  /**
   * The rebuilt model is the subclass, not the base declaring {@code from_config}: {@code cls(...)}
   * in an inherited classmethod constructs the derived class, so the subclass's own {@code call}
   * runs on the rebuilt model's inputs.
   */
  @Test
  public void testInheritedChainRebuildsSubclass()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_config_chain_inherited.py",
        "consume_net_inputs",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 8, 4))));
  }
}
