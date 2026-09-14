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
 * fixture,function,nodes,edges,sliceSites,sliceEmpty,sliceCycles,caughtExceptions}, and with the
 * node-list directory set too, the sorted node list is written beside it. The census is the
 * instrument that sees a pointer-analysis change remove dispatch targets, which no value-level
 * reading can, so its contract is asserted here rather than trusted: the counts are positive, the
 * slice call sites the fixture's graph reaches all have a non-empty result, and the node list has
 * exactly as many lines as the node count.
 *
 * <p>The last column is the resolver's count of slice-result queries in a cycle of its query graph,
 * the engine's view of a loop-carried slice. It has a positive control, the fixture whose slice is
 * loop-carried reads at least one, and a negative, a fixture whose slices are straight-line reads
 * zero, so that a zero elsewhere is a reading and not an absence.
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
    assertEquals(
        "fixture,function,nodes,edges,sliceSites,sliceEmpty,sliceCycles,caughtExceptions",
        8,
        fields.length);
    assertEquals("no query evaluation threw on this fixture: " + fields[7], "0", fields[7]);
    assertEquals(fixture, fields[0]);
    assertEquals(function, fields[1]);
    long nodes = Long.parseLong(fields[2]);
    assertTrue("the graph has nodes", nodes > 0);
    assertTrue("the graph has edges", Long.parseLong(fields[3]) > 0);
    assertTrue("the fixture's slice calls are reached", Long.parseLong(fields[4]) > 0);
    assertEquals("every reached slice call has a result", 0, Long.parseLong(fields[5]));
    // The positive control: this fixture's `loop = loop[:, 1:]` puts the slice call's own result in
    // its receiver's set, so the slice query depends on itself and sits in a cycle.
    assertTrue(
        "the loop-carried slice is a slice query in a cycle: " + fields[6],
        Long.parseLong(fields[6]) >= 1);

    Path nodeList = nodesDir.resolve(fixture + "__" + function + ".nodes");
    assertTrue("the node list is written beside the census", Files.exists(nodeList));
    assertEquals("one line per node", nodes, Files.readAllLines(nodeList).size());
    // The sites of the slice generators in a cycle are written beside it, one line each, so the
    // count above names where to look; the loop-carried slice is the one line here.
    Path sliceCycles = nodesDir.resolve(fixture + "__" + function + ".slicecycles");
    assertTrue("the slice cycle list is written beside the census", Files.exists(sliceCycles));
    List<String> sites = Files.readAllLines(sliceCycles);
    assertEquals(
        "one line per slice generator in a cycle: " + sites,
        Long.parseLong(fields[6]),
        sites.size());
    assertTrue("the site names the fixture: " + sites, sites.get(0).contains(fixture));
  }

  @Test
  public void testCensusReadsNoSliceCycleOnStraightLineSlices()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Path dir = Files.createTempDirectory("callgraph-census");
    Path file = dir.resolve("census.csv");
    String fixture = "tf2_test_column_slice_rank.py";
    String function = "consume_const_col";
    String oldFile = System.getProperty(FILE_PROPERTY);
    System.setProperty(FILE_PROPERTY, file.toString());
    try {
      test(fixture, function, 1, 1, Map.of(2, Set.of(TENSOR_4_FLOAT32)));
    } finally {
      restore(FILE_PROPERTY, oldFile);
    }

    List<String> lines = Files.readAllLines(file);
    assertEquals("one census line per analysis", 1, lines.size());
    String[] fields = lines.get(0).split(",");
    assertEquals(8, fields.length);
    // The negative control: every slice in this fixture is straight-line, so no slice query depends
    // on itself. This is what makes a zero elsewhere a reading rather than an absence.
    assertEquals("no slice query in a cycle: " + fields[6], "0", fields[6]);
  }

  private static void restore(String property, String value) {
    if (value == null) System.clearProperty(property);
    else System.setProperty(property, value);
  }
}
