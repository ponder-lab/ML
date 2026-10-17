package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
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
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that {@code tape.gradient(loss, sources)} returns a container when {@code sources} is one
 * (<a href="https://github.com/wala/ML/issues/464">wala/ML#464</a>): a list for a list, a tuple for
 * a tuple, and no container for a single source. Each result is handed to its own consumer, whose
 * parameter is read.
 */
public class TestGradientContainer extends AbstractTensorTest {

  private static final String FILE = "tf2_test_gradient_container.py";

  /** A list of sources gives a list. */
  @Test
  public void testList() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(containers("take_list").contains(PythonTypes.list));
  }

  /** A tuple of sources gives a tuple. */
  @Test
  public void testTuple() throws ClassHierarchyException, CancelException, IOException {
    assertTrue(containers("take_tuple").contains(PythonTypes.tuple));
  }

  /** A single source gives no container. */
  @Test
  public void testSingle() throws ClassHierarchyException, CancelException, IOException {
    assertEquals(Set.of(), containers("take_single"));
  }

  /** The elements of a loop over a list of gradients are the sources' gradients. */
  @Test
  public void testLoopElement() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "scale", 1, 2, Map.of(2, Set.of(TENSOR_2_FLOAT32, TENSOR_3_FLOAT32)));
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
