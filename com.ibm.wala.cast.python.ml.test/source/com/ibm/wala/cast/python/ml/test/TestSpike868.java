package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import java.io.File;
import java.util.Collections;
import java.util.Iterator;
import org.junit.Test;

/**
 * Spike probe for wala/ML#868 layer two (NOT for commit): measures, on the two-file reduction,
 * whether the decorated test method's class parameter receives anything and which nodes are
 * reachable, before and after the entrypoint-binding prototype.
 */
public class TestSpike868 extends TestPythonMLCallGraphShape {

  @Test
  public void probeParameterState() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            Collections.<File>emptyList(), "test_probe868_parameterized.py", "probe868mod.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    CallGraph cg = builder.makeCallGraph(builder.getOptions());

    System.err.println("SPIKE868 nodes: " + cg.getNumberOfNodes());

    for (CGNode node : cg) {
      String sig = node.getMethod().getSignature();
      if (!sig.contains("probe868")) continue;

      System.err.println("SPIKE868 NODE " + sig);

      if (sig.contains("test_pick.do")) {
        IR ir = node.getIR();
        for (int vn = 1; vn <= ir.getNumberOfParameters(); vn++) {
          PointerKey pk =
              builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
          int size = 0;
          for (InstanceKey ik : builder.getPointerAnalysis().getPointsToSet(pk)) {
            System.err.println("SPIKE868   PARAM v" + vn + " MEMBER " + ik);
            size++;
          }
          System.err.println("SPIKE868   PARAM v" + vn + " SIZE " + size);
        }

        for (Iterator<CGNode> succs = cg.getSuccNodes(node); succs.hasNext(); )
          System.err.println("SPIKE868   SUCC " + succs.next().getMethod().getSignature());
      }
    }
  }
}
