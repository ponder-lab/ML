package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.modref.PythonModRef;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a call of the {@code time}, {@code random} or {@code cv2} module, or a file opened
 * with {@code tf.io.gfile.GFile}, reads or writes state the program does not pass it, as a mod-ref
 * query over the call graph sees it (wala/ML#1080). Before, these modules had no summary, so the
 * module and its members had empty points-to sets and the calls resolved to nothing: a function
 * that reads the clock, draws a random number or reads a file read as touching no state.
 */
public class TestModuleIoEffects extends AbstractTensorTest {

  private static final String FILE = "tf2_test_module_io_effects.py";

  /** {@code time.time()} reads the clock. */
  @Test
  public void testReadsClock() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("reads_clock", false, "Ltime.clock");
  }

  /** {@code random.random()} advances the generator's state. */
  @Test
  public void testDrawsPythonRandom() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("draws_python_random", true, "Lrandom.state");
  }

  /** {@code random.uniform(a, b)} advances the generator's state. */
  @Test
  public void testScalesRandomly() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("scales_randomly", true, "Lrandom.state");
  }

  /** {@code cv2.imread(path)} reads the file system. */
  @Test
  public void testReadsImage() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("reads_image", false, "Lcv2.files");
  }

  /** {@code cv2.imwrite(path, image)} writes the file system. */
  @Test
  public void testWritesImage() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("writes_image", true, "Lcv2.files");
  }

  /**
   * Counting a file's lines through {@code tf.io.gfile.GFile}, in a {@code with} statement, reads
   * the file system.
   */
  @Test
  public void testCountLines() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("count_lines", false, "Ltensorflow/io/gfile.files");
  }

  /**
   * Iterating an open file, {@code for line in f}, reads the file system: the loop goes through the
   * file's {@code __iter__} and its iterator's {@code __next__}.
   */
  @Test
  public void testIteratesFile() throws ClassHierarchyException, CancelException, IOException {
    this.assertTouches("iterates_file", false, "Ltensorflow/io/gfile.files");
  }

  /**
   * A line a loop over an open file binds is no function object: the iteration does not read the
   * file object's method fields as its elements.
   */
  @Test
  public void testLineIsNoFunction() throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String signature = "script " + FILE + ".consume_line.do()LRoot;";
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      PointerKey line =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(node, node.getIR().getParameter(1));
      for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(line))
        assertFalse(
            "The line is a function object: " + instance,
            instance.concreteType().getName().toString().contains("/functions/"));
    }
    assertTrue("Function must exist in call graph: " + signature, reached);
  }

  /** A control: a function that calls none of these modules touches none of their state. */
  @Test
  public void testDoublesTouchesNoModuleState()
      throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), this.moduleState("doubles", false));
    assertEquals(Set.of(), this.moduleState("doubles", true));
  }

  /**
   * Asserts that a function reads, or modifies, a module-state field.
   *
   * @param function The function.
   * @param modified Whether the field is modified rather than read.
   * @param field The field, as {@code Type.field}.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private void assertTouches(String function, boolean modified, String field)
      throws ClassHierarchyException, CancelException, IOException {
    Set<String> state = this.moduleState(function, modified);
    assertTrue(
        function + (modified ? " modifies " : " reads ") + field + ": " + state,
        state.contains(field));
  }

  /** The module-state fields the summaries read and write. */
  private static final Set<String> STATE_FIELDS = Set.of("clock", "state", "files");

  /**
   * Runs the analysis on the fixture and collects the module-state fields a function reads or
   * modifies, transitively through its callees, as {@code Type.field}, where the type is the module
   * object's.
   *
   * @param function The function.
   * @param modified Whether to collect the modified fields rather than the read ones.
   * @return The fields.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<String> moduleState(String function, boolean modified)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    PythonModRef modRef = new PythonModRef();
    Map<CGNode, OrdinalSet<PointerKey>> keys =
        modified
            ? modRef.computeMod(cg, builder.getPointerAnalysis())
            : modRef.computeRef(cg, builder.getPointerAnalysis());

    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    Set<String> ret = HashSetFactory.make();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      for (PointerKey key : keys.get(node)) {
        if (!(key instanceof InstanceFieldKey field)) continue;
        String name = field.getField().getName().toString();
        if (!STATE_FIELDS.contains(name)) continue;
        ret.add(field.getInstanceKey().concreteType().getName() + "." + name);
      }
    }
    assertTrue("Function must exist in call graph: " + signature, reached);
    return ret;
  }
}
