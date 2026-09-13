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
 * The call-graph census hook of {@link AbstractTensorTest} (wala/ML#916): with the census file
 * property set, every analysis appends one line of {@code
 * fixture,function,nodes,edges,sliceSites,sliceEmpty}, and with the node-list directory set too,
 * the sorted node list is written beside it. The census is the instrument that sees a
 * pointer-analysis change remove dispatch targets, which no value-level reading can, so its
 * contract is asserted here rather than trusted: the counts are positive, the slice call sites the
 * fixture's graph reaches all have a non-empty result, and the node list has exactly as many lines
 * as the node count.
 */
public class TestCallGraphCensus extends AbstractTensorTest {

  private static final String FILE_PROPERTY = "wala.ml.callgraph.census.file";

  private static final String NODES_DIR_PROPERTY = FILE_PROPERTY + ".nodes.dir";

  @Test
  public void testCensusRecordsOneLinePerAnalysis()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Path dir = Files.createTempDirectory("callgraph-census");
    Path file = dir.resolve("census.csv");
    Path nodesDir = dir.resolve("nodes");
    String fixture = "tf2_test_slice_result_allocation.py";
    String function = "consume_first_window";
    String oldFile = System.getProperty(FILE_PROPERTY);
    String oldNodesDir = System.getProperty(NODES_DIR_PROPERTY);
    System.setProperty(FILE_PROPERTY, file.toString());
    System.setProperty(NODES_DIR_PROPERTY, nodesDir.toString());
    try {
      test(fixture, function, 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2048, 8))));
    } finally {
      restore(FILE_PROPERTY, oldFile);
      restore(NODES_DIR_PROPERTY, oldNodesDir);
    }

    List<String> lines = Files.readAllLines(file);
    assertEquals("one census line per analysis", 1, lines.size());
    String[] fields = lines.get(0).split(",");
    assertEquals("fixture,function,nodes,edges,sliceSites,sliceEmpty", 6, fields.length);
    assertEquals(fixture, fields[0]);
    assertEquals(function, fields[1]);
    long nodes = Long.parseLong(fields[2]);
    assertTrue("the graph has nodes", nodes > 0);
    assertTrue("the graph has edges", Long.parseLong(fields[3]) > 0);
    assertTrue("the fixture's slice calls are reached", Long.parseLong(fields[4]) > 0);
    assertEquals("every reached slice call has a result", 0, Long.parseLong(fields[5]));

    Path nodeList = nodesDir.resolve(fixture + "__" + function + ".nodes");
    assertTrue("the node list is written beside the census", Files.exists(nodeList));
    assertEquals("one line per node", nodes, Files.readAllLines(nodeList).size());
  }

  private static void restore(String property, String value) {
    if (value == null) System.clearProperty(property);
    else System.setProperty(property, value);
  }
}
