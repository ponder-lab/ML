package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that reading {@code trainable_variables} off an instance of a subclassed Keras model gives
 * a list (<a href="https://github.com/wala/ML/issues/1111">wala/ML#1111</a>), so {@code
 * tape.gradient(loss, model.trainable_variables)} returns a list of gradients by the rule of <a
 * href="https://github.com/wala/ML/issues/464">wala/ML#464</a>. Each value is handed to its own
 * consumer, whose parameter is read.
 */
public class TestTrainableVariables extends AbstractTensorTest {

  private static final String FILE = "tf2_test_trainable_variables.py";

  /** A subclass whose own {@code __init__} calls {@code super().__init__()} has a list. */
  @Test
  public void testModelTrainableVariables()
      throws ClassHierarchyException, CancelException, IOException {
    assertTrue(containers("take_sources").contains(PythonTypes.list));
  }

  /** The gradient over the model's variables is a list. */
  @Test
  public void testGradientOverModel() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(containers("take_grads").contains(PythonTypes.list));
  }

  /** A subclass that inherits {@code Model.__init__} has a list too. */
  @Test
  public void testInheritedInit() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(containers("take_plain_sources").contains(PythonTypes.list));
  }

  /** A model built by the functional constructor has a list too. */
  @Test
  public void testFunctionalModel() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(containers("take_functional_sources").contains(PythonTypes.list));
  }

  /**
   * The container types, among list, tuple and dict, that a function's first parameter may hold.
   *
   * @param function The function, in the fixture.
   * @return The container types of the parameter's allocations.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the fixture cannot be read.
   */
  private Set<TypeReference> containers(String function)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FILE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    CallGraph cg = builder.makeCallGraph(builder.getOptions());
    String signature = "script " + FILE + "." + function + ".do()LRoot;";
    Set<TypeReference> ret = new HashSet<>();
    boolean reached = false;
    for (CGNode node : cg) {
      if (!node.getMethod().getSignature().equals(signature)) continue;
      reached = true;
      PointerKey param = builder.getPointerKeyForLocal(node, node.getIR().getParameter(1));
      for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(param)) {
        TypeReference type = ik.concreteType().getReference();
        if (type.equals(PythonTypes.list)
            || type.equals(PythonTypes.tuple)
            || type.equals(PythonTypes.dict)) ret.add(type);
      }
    }
    assertTrue("`" + function + "` must be in the call graph.", reached);
    return ret;
  }
}
