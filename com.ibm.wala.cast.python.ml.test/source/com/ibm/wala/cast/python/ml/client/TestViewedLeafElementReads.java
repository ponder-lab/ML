package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import java.io.File;
import java.util.Collections;
import org.junit.Test;

/**
 * Tests that a shared read of a mirrored dataset element leaf's element ({@link
 * DatasetTupleElementGenerator#sharingReads}) is not kept when its computation views a dataset over
 * one of its receivers (<a href="https://github.com/wala/ML/issues/1273">wala/ML#1273</a>). The
 * class has one test, since it turns on the resolver's check mode for its JVM.
 */
public class TestViewedLeafElementReads extends AbstractTensorTest {

  /** Enumerations of datasets, among them ones whose receiver may be several datasets. */
  private static final String FILE = "tf2_test_dataset_enumerate_element.py";

  /**
   * The analysis passes the resolver's check mode, which computes a kept value again at every read
   * of it and fails when the two differ or read different queries. The element of an enumeration of
   * a receiver that may be several datasets is read through views of the receiver's transformations
   * ({@link DatasetGenerator#overReceiver}), made anew by each computation, whose queries are keyed
   * by identity; kept, such a read stood on queries its recomputation never reads.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testAnalysisPassesTheCheckMode() throws Exception {
    String before = System.getProperty(WorklistTypeResolver.CHECK_TRACKED_PROPERTY);
    System.setProperty(WorklistTypeResolver.CHECK_TRACKED_PROPERTY, "true");
    try {
      PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
      PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
      builder.makeCallGraph(builder.getOptions());
      engine.performAnalysis(builder);
      assertTrue(
          "The analysis reads a leaf's element.",
          DatasetTupleElementGenerator.readComputations(builder) > 0);
    } finally {
      if (before == null) System.clearProperty(WorklistTypeResolver.CHECK_TRACKED_PROPERTY);
      else System.setProperty(WorklistTypeResolver.CHECK_TRACKED_PROPERTY, before);
    }
  }
}
