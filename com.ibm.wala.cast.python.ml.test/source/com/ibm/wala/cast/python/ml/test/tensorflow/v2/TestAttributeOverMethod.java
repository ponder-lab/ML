package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.TrampolineReceiverContextSelector;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a call through an instance attribute that the program assigns over a method the model
 * installs on a framework object (<a
 * href="https://github.com/wala/ML/issues/1044">wala/ML#1044</a>). At run time the instance
 * attribute shadows the class's method, so a call through it reaches only the assigned value. The
 * analysis may drop the method only when no read can see it: the write is the first use of the
 * object after the call that creates it, and no attribute of that name is ever deleted.
 */
public class TestAttributeOverMethod extends AbstractTensorTest {

  /** The method trampoline {@code tf.keras.Model} installs as {@code predict}. */
  private static final String MODEL_PREDICT = "tensorflow/keras/models/Model/predict";

  /** The builtin {@code len}. */
  private static final String LEN = "len";

  /**
   * The call in {@code calls_replaced}, on a model whose {@code predict} was assigned {@code len}
   * before any other use, reaches {@code len} and not the model's method.
   */
  @Test
  public void testReplaced() throws ClassHierarchyException, CancelException, IOException {
    Set<String> targets = predictTargets("tf2_test_attribute_over_method.py", "calls_replaced");
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(LEN)));
    assertFalse(targets.toString(), targets.stream().anyMatch(t -> t.contains(MODEL_PREDICT)));
  }

  /**
   * The call in {@code calls_kept}, on a model whose {@code predict} is assigned only after the
   * call, reaches the model's method.
   */
  @Test
  public void testAssignedAfterUse() throws ClassHierarchyException, CancelException, IOException {
    Set<String> targets = predictTargets("tf2_test_attribute_over_method.py", "calls_kept");
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(MODEL_PREDICT)));
  }

  /**
   * A program that deletes the attribute exposes the method again, so the call reaches both the
   * assigned value and the model's method.
   */
  @Test
  public void testDeleted() throws ClassHierarchyException, CancelException, IOException {
    Set<String> targets =
        predictTargets("tf2_test_attribute_over_method_deleted.py", "calls_replaced");
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(LEN)));
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(MODEL_PREDICT)));
  }

  /**
   * A program that deletes the attribute through a local alias of {@code delattr} exposes the
   * method again, so the call reaches both the assigned value and the model's method.
   */
  @Test
  public void testDeletedThroughAlias()
      throws ClassHierarchyException, CancelException, IOException {
    Set<String> targets =
        predictTargets("tf2_test_attribute_over_method_deleted_alias.py", "calls_replaced");
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(LEN)));
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(MODEL_PREDICT)));
  }

  /**
   * The rule still applies in a program with a function whose SSA conversion fails: the scan for
   * attribute deletions reads each body's instructions before conversion, so that function neither
   * aborts the call graph's construction nor counts as a deletion.
   */
  @Test
  public void testUnconvertibleBody() throws ClassHierarchyException, CancelException, IOException {
    Set<String> targets =
        predictTargets("tf2_test_attribute_over_method_unconvertible.py", "calls_replaced");
    assertTrue(targets.toString(), targets.stream().anyMatch(t -> t.contains(LEN)));
    assertFalse(targets.toString(), targets.stream().anyMatch(t -> t.contains(MODEL_PREDICT)));
  }

  /**
   * A write before any other use of a name the model does not install as a method shadows nothing,
   * so the call that creates the object keeps its context.
   */
  @Test
  public void testNonMethodNotKeyed() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(0, keyedNodes("tf2_test_attribute_over_non_method.py"));
  }

  /** A write over an installed method keys the call that creates the object. */
  @Test
  public void testMethodKeyed() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(keyedNodes("tf2_test_attribute_over_method.py") > 0);
  }

  /**
   * Counts the call graph's nodes whose context carries {@link
   * TrampolineReceiverContextSelector#SHADOWED_ATTRIBUTES}.
   *
   * @param fixture The fixture file.
   * @return The number of keyed nodes.
   */
  private long keyedNodes(String fixture)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), new String[] {fixture});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    Util.addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    long keyed = 0;
    for (CGNode node : CG)
      if (node.getContext().get(TrampolineReceiverContextSelector.SHADOWED_ATTRIBUTES) != null)
        keyed++;
    return keyed;
  }

  /**
   * Returns the declaring classes of the targets of the calls in a function's nodes that are not
   * calls of TensorFlow functions, which in the fixtures are the {@code m.predict(t)} calls.
   *
   * @param fixture The fixture file.
   * @param function The function.
   * @return The targets' declaring class names.
   */
  private Set<String> predictTargets(String fixture, String function)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), new String[] {fixture});
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    Util.addPytestEntrypoints(builder);
    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);
    Set<String> targets = new HashSet<>();
    boolean found = false;
    for (CGNode node : CG) {
      if (!node.getMethod().getSignature().contains(fixture + "." + function + ".do")) continue;
      found = true;
      for (SSAInstruction instruction : node.getIR().getInstructions())
        if (instruction instanceof PythonInvokeInstruction invoke)
          for (CGNode target : CG.getPossibleTargets(node, invoke.getCallSite())) {
            String name = target.getMethod().getDeclaringClass().getName().toString();
            if (!name.startsWith("Ltensorflow/math") && !name.contains("reduce_sum"))
              targets.add(name);
          }
    }
    assertTrue("Function must exist in call graph: " + function, found);
    return targets;
  }
}
