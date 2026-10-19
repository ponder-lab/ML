package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashSetFactory;
import java.io.IOException;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a {@code Sequential} model's call passes each of its layers the {@code training} value
 * it is called with, by keyword, as Keras's {@code Sequential.call} does (wala/ML#1120). Before,
 * the summary called each layer with its input only, so a layer's {@code training} took its
 * default.
 */
public class TestSequentialTraining extends AbstractTensorTest {

  private static final String FILE = "tf2_test_sequential_training.py";

  /** A layer whose {@code call} takes {@code training} second receives the model's {@code True}. */
  @Test
  public void testTrainingReachesLayer()
      throws ClassHierarchyException, CancelException, IOException {
    Set<Object> training = this.parameterConstants("Block.call", 3);
    assertTrue(training.toString(), training.contains(true));
  }

  /**
   * A layer whose {@code call} declares {@code mask} before {@code training} receives {@code True}
   * in {@code training}, and not in {@code mask}: the value binds by name, not by position.
   */
  @Test
  public void testTrainingBindsByName()
      throws ClassHierarchyException, CancelException, IOException {
    Set<Object> training = this.parameterConstants("Masked.call", 4);
    assertTrue(training.toString(), training.contains(true));
    Set<Object> mask = this.parameterConstants("Masked.call", 3);
    assertFalse(mask.toString(), mask.contains(true));
  }

  /**
   * Runs the analysis on the fixture and collects the constants a function's parameter receives.
   *
   * @param function The function, as {@code Class.method}.
   * @param parameter The parameter's index, counting the function object and {@code self}.
   * @return The constants.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Set<Object> parameterConstants(String function, int parameter)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    Set<Object> ret = HashSetFactory.make();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      PointerKey key =
          builder
              .getPointerAnalysis()
              .getHeapModel()
              .getPointerKeyForLocal(node, node.getIR().getParameter(parameter));
      for (InstanceKey instance : builder.getPointerAnalysis().getPointsToSet(key))
        if (instance instanceof ConstantKey<?> constant) ret.add(constant.getValue());
    }
    assertTrue("Function must exist in call graph: " + signature, reached);
    return ret;
  }
}
