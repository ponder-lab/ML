package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.Collections;
import org.junit.Test;

/**
 * A probe, not a test: prints the points-to sets that decide how a constant guard's dead arm
 * reaches the pointer analysis (wala/ML#1187, wala/ML#1256). It asserts nothing; its output is
 * recorded at https://github.com/wala/ML/issues/1187#issuecomment-6892084051.
 */
public class ProbeConstantGuards extends TestPythonMLCallGraphShape {
  private static void dump(
      PythonSSAPropagationCallGraphBuilder b, String suffix, int vn, String label) {
    int nodes = 0;
    for (CGNode node : b.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith(suffix)) {
        nodes++;
        PointerKey p = b.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
        StringBuilder s = new StringBuilder();
        for (InstanceKey k : b.getPointerAnalysis().getPointsToSet(p)) {
          String t = k.toString();
          s.append(" {").append(t.length() > 140 ? t.substring(0, 140) : t).append("}");
        }
        System.out.println(
            "PROBE "
                + label
                + " node#"
                + nodes
                + " ctx="
                + node.getContext().toString().length()
                + ":"
                + s);
      }
    if (nodes == 0) System.out.println("PROBE " + label + " NO NODE");
  }

  @Test
  public void probe() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_probe_constant_guards.py");
    PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
    b.makeCallGraph(b.getOptions());
    engine.performAnalysis(b);
    dump(b, "/Net/call", 4, "Net.call training");
    dump(b, "/Net/call", 5, "Net.call eval");
    dump(b, "/consume", 2, "consume y_pred");
    dump(b, "/flag_full", 2, "Full self.concat");
    dump(b, "/flag_noassign", 2, "NoAssign self.concat");
    dump(b, "/message_full", 3, "index full (both routes)");
    dump(b, "/message_noassign", 3, "index noassign (call route)");
    dump(b, "/message_nosub", 3, "index nosub (merge route)");
    dump(b, "/message_bind", 3, "index bind only (call, no subscript)");
    dump(b, "/message_none", 3, "index no arm (control)");
    dump(b, "/add_loop", 2, "add_loop edges");
    java.util.List<CGNode> calls = new java.util.ArrayList<>();
    for (CGNode n : b.getCallGraph())
      if (n.getMethod().getDeclaringClass().getName().toString().endsWith("/Net/call"))
        calls.add(n);
    for (CGNode n : b.getCallGraph())
      if (n.getMethod().getDeclaringClass().getName().toString().endsWith("/custom_train_step"))
        for (java.util.Iterator<CGNode> it = b.getCallGraph().getSuccNodes(n); it.hasNext(); ) {
          CGNode t = it.next();
          System.out.println(
              "PROBE train_step -> "
                  + t.getMethod().getDeclaringClass().getName()
                  + " netcall#"
                  + (calls.indexOf(t) + 1));
        }
    for (CGNode n : calls)
      for (java.util.Iterator<CGNode> it = b.getCallGraph().getPredNodes(n); it.hasNext(); )
        System.out.println(
            "PROBE netcall#"
                + (calls.indexOf(n) + 1)
                + " <- "
                + it.next().getMethod().getDeclaringClass().getName());
  }
}
