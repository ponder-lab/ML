package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.ElementMirrorKey;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that the analysis reads each of a dataset element's mirrored leaves once, however many
 * values hold it (<a href="https://github.com/wala/ML/issues/1269">wala/ML#1269</a>). Every value
 * holding a leaf reads it the same way, and unmemoized, each read of a value spread the leaf's
 * datasets and read their structures at each path again.
 */
public class TestMirroredLeafReads extends AbstractTensorTest {

  private static final String FILE = "tf2_test_dataset_leaf_reads.py";

  /** The members of a mapped, batched pair, relayed through three functions, keep their types. */
  @Test
  public void testRelayedFeatures() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /** As {@link #testRelayedFeatures()}, for the pair's second member. */
  @Test
  public void testRelayedLabels() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_label", 1, 1, Map.of(2, Set.of(TENSOR_2_INT32)));
  }

  /**
   * The analysis reads no leaf more than once: the readings it computes are at most the mirrored
   * leaves the pointer analysis holds, though many more values hold them.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testEachLeafReadOnce() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    long leaves = 0;
    for (InstanceKey key : builder.getPointerAnalysis().getInstanceKeys())
      if (key instanceof ElementMirrorKey mirror && mirror.isLeaf()) leaves++;
    assertTrue("The fixture has mirrored leaves.", leaves > 0);
    engine.performAnalysis(builder);
    long reads = TensorGeneratorFactory.leafReads(builder);
    assertTrue("The analysis reads a leaf.", reads > 0);
    assertTrue(
        "The analysis read " + reads + " leaf readings for " + leaves + " leaves.",
        reads <= leaves);
  }

  /**
   * A second dispatch of a value scans nothing (<a
   * href="https://github.com/wala/ML/issues/1293">wala/ML#1293</a>): the scan of every value the
   * first reached is memoized while an analysis resolves for the builder. Unmemoized, the second
   * scans the value again before it finds its own memoized dispatch.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testRepeatedDispatchScansNothing() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    // A snapshot: dispatch creates points-to set variables for keys it reads.
    List<PointerKey> keys = new ArrayList<>();
    builder.getPointerAnalysis().getPointerKeys().forEach(keys::add);
    WorklistTypeResolver.install(builder);
    try {
      int values = 0;
      for (PointerKey key : keys) {
        if (builder.getPropagationSystem().isImplicit(key)) continue;
        boolean holdsLeaf = false;
        for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(key))
          holdsLeaf |= instance instanceof ElementMirrorKey mirror && mirror.isLeaf();
        if (!holdsLeaf) continue;
        values++;
        PointsToSetVariable variable = builder.getPropagationSystem().findOrCreatePointsToSet(key);
        TensorGeneratorFactory.getGenerator(variable, builder);
        long scans = TensorGeneratorFactory.leafScanCount(builder);
        assertTrue("The first dispatch scans the value.", scans > 0);
        TensorGeneratorFactory.getGenerator(variable, builder);
        assertEquals(key + "", scans, TensorGeneratorFactory.leafScanCount(builder));
      }
      assertTrue("The fixture has values holding mirrored leaves.", values > 1);
    } finally {
      WorklistTypeResolver.uninstall(builder, false);
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * The memoized readings type every value holding a mirrored leaf as the readings computed afresh
   * do, read while an analysis resolves for the builder, the only time they are memoized, and read
   * again once the memo holds them.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testMemoizedReadingsAgree() throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    // A snapshot: dispatch creates points-to set variables for keys it reads.
    List<PointerKey> keys = new ArrayList<>();
    builder.getPointerAnalysis().getPointerKeys().forEach(keys::add);
    WorklistTypeResolver.install(builder);
    try {
      int values = 0;
      for (PointerKey key : keys) {
        if (builder.getPropagationSystem().isImplicit(key)) continue;
        boolean holdsLeaf = false;
        for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(key))
          holdsLeaf |= instance instanceof ElementMirrorKey mirror && mirror.isLeaf();
        if (!holdsLeaf) continue;
        values++;
        PointsToSetVariable variable = builder.getPropagationSystem().findOrCreatePointsToSet(key);
        Set<TensorType> fresh =
            types(TensorGeneratorFactory.getUnmemoizedGenerator(variable, builder), builder);
        assertEquals(
            key + "",
            fresh,
            types(TensorGeneratorFactory.getGenerator(variable, builder), builder));
        assertEquals(
            key + " (replayed)",
            fresh,
            types(TensorGeneratorFactory.getGenerator(variable, builder), builder));
      }
      assertTrue("The fixture has values holding mirrored leaves.", values > 1);
      assertTrue("The readings were memoized.", TensorGeneratorFactory.hasLeafReadings(builder));
      assertTrue("The scans were memoized.", TensorGeneratorFactory.hasLeafScans(builder));
      assertTrue(
          "The mapped components were memoized.", DatasetMapGenerator.hasMappedComponents(builder));
    } finally {
      WorklistTypeResolver.uninstall(builder, false);
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * A generator's tensor types.
   *
   * @param generator The generator, or {@code null}.
   * @param builder The builder.
   * @return Its types, or {@code null} for no generator.
   */
  private static Set<TensorType> types(
      TensorGenerator generator, PythonSSAPropagationCallGraphBuilder builder) {
    return generator == null ? null : generator.getTensorTypes(builder);
  }
}
