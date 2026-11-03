package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A dataset whose element is fed back into a dataset built from it, a {@code batch} applied in a
 * loop to its own result or an element passed to {@code from_tensor_slices} again, still has a
 * finite copy of its element's structure, so the analysis finishes (<a
 * href="https://github.com/wala/ML/issues/1190">wala/ML#1190</a>).
 */
public class TestDatasetElementCycles extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_element_cycles.py";

  /**
   * The call graph and the tensor analysis finish on the fixture.
   *
   * @throws Exception if the fixture cannot be analyzed.
   */
  @Test(timeout = 120_000)
  public void testFinishes() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
  }

  /**
   * A member of the element of a {@code batch} applied three times in a loop to its own result.
   *
   * <p>TODO: Typed as batched once, {@code (2, 4)}, beside the unbatched {@code (4,)}, as on
   * master, until https://github.com/wala/ML/issues/1197 is fixed.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test(timeout = 120_000, expected = AssertionError.class)
  public void testRebatched() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_rebatched", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2, 2, 4))));
  }

  /**
   * An element of a dataset built from another dataset's element: a scalar, {@code ()}, at run
   * time. The loop variable also holds the dataset before the loop, whose element is {@code (4,)},
   * as on master.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test(timeout = 120_000)
  public void testResliced() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_resliced",
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32), TensorType.of(FLOAT_32, 4))));
  }

  /**
   * A leaf of a dataset built from a dict whose points-to set holds itself three ways: the copy of
   * its structure is bounded, and the leaf keeps its type.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test(timeout = 120_000)
  public void testSelfReferencing() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_self_referencing", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 3))));
  }
}
