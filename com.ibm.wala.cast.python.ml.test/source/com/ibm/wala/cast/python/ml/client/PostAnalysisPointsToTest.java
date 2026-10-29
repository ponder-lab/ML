package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ssa.IR;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * A query made after the analysis only reads the points-to map (<a
 * href="https://github.com/wala/ML/issues/1173">wala/ML#1173</a>). Before the fix such a query
 * registered a points-to set for each key the solver never created, so a client querying on several
 * threads, or querying while walking the pointer keys, raced on the map.
 */
public class PostAnalysisPointsToTest extends TestPythonMLCallGraphShape {

  /**
   * The number of pointer keys the pointer analysis holds.
   *
   * @param builder The builder.
   * @return The count.
   */
  private static int pointerKeys(PythonSSAPropagationCallGraphBuilder builder) {
    return held(builder).size();
  }

  /**
   * The pointer keys the pointer analysis holds.
   *
   * @param builder The builder.
   * @return The keys.
   */
  private static Set<PointerKey> held(PythonSSAPropagationCallGraphBuilder builder) {
    Set<PointerKey> ret = new HashSet<>();
    for (PointerKey key : builder.getPointerAnalysis().getPointerKeys()) ret.add(key);
    return ret;
  }

  /**
   * The nodes of the program's own code, rather than of a summary.
   *
   * @param builder The builder.
   * @return The nodes whose IR the script defines.
   */
  private static List<CGNode> scriptNodes(PythonSSAPropagationCallGraphBuilder builder) {
    List<CGNode> ret = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getIR() != null
          && node.getMethod().getDeclaringClass().getName().toString().startsWith("Lscript "))
        ret.add(node);
    return ret;
  }

  /**
   * Container and API queries a client analyzing each function makes, on every value of the
   * script's code, after the analysis, leave the number of pointer keys unchanged. They include
   * values the solver made no set for, and element fields of containers that no write reached. The
   * points-to set of such a key is an empty one the map doesn't hold, the same each time it's asked
   * for.
   */
  @Test
  public void testPostAnalysisQueriesAddNoPointerKey() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_container_elements.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    try {
      Set<PointerKey> held = held(builder);
      int before = held.size();

      PointerKey unsolved = null;
      for (CGNode node : scriptNodes(builder)) {
        IR ir = node.getIR();
        for (SSAInstruction instruction : ir.getInstructions())
          if (instruction instanceof SSAAbstractInvokeInstruction call)
            TensorFlowApis.of(builder, node, call);

        for (int vn = 1; vn <= ir.getSymbolTable().getMaxValueNumber(); vn++) {
          PointerKey value =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
          if (builder.getPropagationSystem().isImplicit(value)) continue;
          if (unsolved == null && !held.contains(value)) unsolved = value;
          ContainerElements.ofValue(builder, value).elementTypes(analysis);
          for (InstanceKey container : builder.getPointerAnalysis().getPointsToSet(value))
            ContainerElements.of(builder, container).elementTypes(analysis);
        }
      }

      assertEquals(
          "A query made after the analysis adds no pointer key.", before, pointerKeys(builder));

      assertTrue("Expecting a script value the solver made no set for.", unsolved != null);
      PointsToSetVariable first = builder.getPropagationSystem().findOrCreatePointsToSet(unsolved);
      assertSame(
          "A key the solver never created reads as the same set each time.",
          first,
          builder.getPropagationSystem().findOrCreatePointsToSet(unsolved));
      assertTrue(
          "A key the solver never created reads as an empty set.",
          first.getValue() == null || first.getValue().isEmpty());
      assertEquals(
          "Reading a key the solver never created adds no pointer key.",
          before,
          pointerKeys(builder));
    } finally {
      TensorGenerator.clearCaches(builder);
    }
  }
}
