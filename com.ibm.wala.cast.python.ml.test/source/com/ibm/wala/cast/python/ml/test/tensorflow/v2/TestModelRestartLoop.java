package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a model rebuilt from its own config in a loop, with the rebuilt model stored back
 * where the next round reads it, does not nest one receiver context per round (<a
 * href="https://github.com/wala/ML/issues/210">wala/ML#210</a>).
 */
public class TestModelRestartLoop extends AbstractTensorTest {

  /** The restarted model's output reaches the sink, and the analysis terminates. */
  @Test(timeout = 300_000)
  public void testRestartLoop() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_model_restart_loop.py",
        "consume",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 3, 2))));
  }
}
