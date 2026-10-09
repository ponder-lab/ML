package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A layer wrapper whose wrapped layer may be another wrapper of its class re-enters the wrapper's
 * {@code call} on another instance at every forwarding call. Keyed on its caller and the dispatched
 * receiver, each re-entry minted a fresh anchored context, so a cycle of wrappers copied everything
 * beneath it once per level up to the depth cap: the call graph of this fixture, three wrappers
 * whose {@code layer} fields each hold either of the other two, carried thousands of nodes of the
 * wrapper's {@code call}. The re-entered method is keyed on the dispatched receiver alone, so the
 * wrapper's {@code call} has a node per receiver and the anchor chain stays shallow. The census
 * hook of {@link AbstractTensorTest} is the instrument: the node list written beside the census
 * names every node with its context.
 */
public class TestWrapperReentryContexts extends AbstractTensorTest {

  private static final String FILE_PROPERTY = "wala.ml.callgraph.census.file";

  private static final String NODES_DIR_PROPERTY = FILE_PROPERTY + ".nodes.dir";

  private static final String FIXTURE = "tf2_test_wrapper_reentry_contexts.py";

  /**
   * The nodes of the wrapper's {@code call} once the re-entries are keyed on their receivers: a
   * trampoline and a body node per dispatched receiver, three, and the pair of the entry from the
   * fixture's script; keyed on their callers too, the cycle minted 5,352.
   */
  private static final int WRAPPER_CALL_NODE_CEILING = 8;

  /**
   * The deepest chain of anchored contexts on a node of the wrapper's {@code call}: the entry's,
   * anchored on the script, and one hop; keyed on their callers too, the chains reached the depth
   * cap of eight.
   */
  private static final int ANCHOR_DEPTH_BOUND = 2;

  @Test
  public void testWrapperCycleIsKeyedPerReceiver()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Path dir = Files.createTempDirectory("wrapper-reentry-census");
    Path file = dir.resolve("census.csv");
    Path nodesDir = dir.resolve("nodes");
    String oldFile = System.getProperty(FILE_PROPERTY);
    String oldNodesDir = System.getProperty(NODES_DIR_PROPERTY);
    System.setProperty(FILE_PROPERTY, file.toString());
    System.setProperty(NODES_DIR_PROPERTY, nodesDir.toString());
    try {
      test(FIXTURE, "consume", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5, 8))));
    } finally {
      restore(FILE_PROPERTY, oldFile);
      restore(NODES_DIR_PROPERTY, oldNodesDir);
    }
    List<String> lines = Files.readAllLines(file);
    assertEquals("one census line per analysis", 1, lines.size());
    List<String> nodes = Files.readAllLines(nodesDir.resolve(FIXTURE + "__consume.nodes"));
    long wrapperCalls = 0;
    int deepest = 0;
    for (String node : nodes) {
      if (!node.contains("LayerWrapper/call")) continue;
      wrapperCalls++;
      // An anchored context prints as "Anchored: <anchor> @ <site>", nesting its anchor's context.
      deepest = Math.max(deepest, node.split("Anchored: ", -1).length - 1);
    }
    assertTrue(
        "the wrapper's call has a node per receiver, not per re-entry: " + wrapperCalls,
        wrapperCalls <= WRAPPER_CALL_NODE_CEILING);
    assertTrue(
        "the wrapper's call is not re-anchored at every hop: deepest chain " + deepest,
        deepest <= ANCHOR_DEPTH_BOUND);
  }

  private static void restore(String property, String value) {
    if (value == null) System.clearProperty(property);
    else System.setProperty(property, value);
  }
}
