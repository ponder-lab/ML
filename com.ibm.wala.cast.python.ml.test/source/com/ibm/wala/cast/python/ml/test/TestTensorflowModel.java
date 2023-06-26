package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import org.junit.Test;

public class TestTensorflowModel extends TestPythonMLCallGraphShape {

  private static final Logger logger = Logger.getLogger(TestTensorflowModel.class.getName());

  @Test
  public void testTf1()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonAnalysisEngine<TensorTypeAnalysis> E = makeEngine("tf1.py");
    PythonSSAPropagationCallGraphBuilder builder = E.defaultCallGraphBuilder();
    CallGraph CG = builder.makeCallGraph(builder.getOptions());

    // CAstCallGraphUtil.AVOID_DUMP = false;
    //
    // CAstCallGraphUtil.dumpCG(((SSAPropagationCallGraphBuilder)builder).getCFAContextInterpreter(),
    // builder.getPointerAnalysis(), CG);

    // System.err.println(CG);

    Collection<CGNode> nodes = getNodes(CG, "script tf1.py/model_fn");
    assert !nodes.isEmpty() : "model_fn should be called";
    check:
    {
      for (CGNode node : nodes) {
        for (Iterator<CGNode> ns = CG.getPredNodes(node); ns.hasNext(); ) {
          if (ns.next().getMethod().isWalaSynthetic()) {
            break check;
          }
        }

        assert false : node + " should have synthetic caller";
      }
    }
  }

  @Test
  public void testTf2()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    testTf2("tf2.py", "add", 2, 3, 2, 3);
    testTf2("tf2b.py", "add", 2, 3, 2, 3);
    testTf2("tf2c.py", "add", 2, 4, 2, 3);
    testTf2("tf2d.py", "add", 2, 3, 2, 3);
    testTf2("tf2d2.py", "add", 2, 3, 2, 3);
    testTf2("tf2d3.py", "add", 2, 3, 2, 3);
    testTf2("tf2d4.py", "add", 2, 4, 2, 3);
    testTf2("tf2e.py", "add", 2, 3, 2, 3);
    testTf2("tf2e2.py", "add", 2, 4, 2, 3);
    testTf2("tf2e3.py", "add", 2, 3, 2, 3);
    testTf2("tf2e4.py", "add", 2, 4, 2, 3);
    testTf2("tf2f.py", "add", 2, 3, 2, 3);
    testTf2("tf2f2.py", "add", 2, 4, 2, 3);
    testTf2("tf2f3.py", "add", 2, 4, 2, 3);
    testTf2("tf2g.py", "add", 2, 3, 2, 3);
    testTf2("tf2g2.py", "add", 2, 4, 2, 3);
    testTf2("tf2h.py", "add", 2, 3, 2, 3);
    testTf2("tf2h2.py", "add", 2, 4, 2, 3);
    testTf2("tf2i.py", "add", 2, 3, 2, 3);
    testTf2("tf2i2.py", "add", 2, 4, 2, 3);
    testTf2("tf2j.py", "add", 2, 3, 2, 3);
    testTf2("tf2j2.py", "add", 2, 4, 2, 3);
    testTf2("tf2k.py", "add", 2, 3, 2, 3);
    testTf2("tf2k2.py", "add", 2, 4, 2, 3);
    testTf2("tf2l.py", "add", 2, 3, 2, 3);
    testTf2("tf2l2.py", "add", 2, 4, 2, 3);
    testTf2("tf2m.py", "add", 2, 3, 2, 3);
    testTf2("tf2m2.py", "add", 2, 4, 2, 3);
    testTf2("tf2n.py", "func2", 1, 4, 2);
    testTf2("tf2n2.py", "func2", 1, 4, 2);
    testTf2("tf2n3.py", "func2", 1, 4, 2);
    testTf2("tf2o.py", "add", 2, 3, 2, 3);
    testTf2("tf2o2.py", "add", 2, 4, 2, 3);
    testTf2("tf2p.py", "value_index", 2, 4, 2, 3);
    testTf2("tf2p2.py", "value_index", 2, 4, 2, 3);
    testTf2("tf2q.py", "add", 2, 3, 2, 3);
    testTf2("tf2r.py", "add", 2, 3, 2, 3);
  }

  private void testTf2(
      String filename,
      String functionName,
      int expectedNumberOfTensorParameters,
      int expectedNumberOfTensorVariables,
      int... expectedTensorParameterValueNumbers)
      throws ClassHierarchyException, CancelException, IOException {
    PythonAnalysisEngine<TensorTypeAnalysis> E = makeEngine(filename);
    PythonSSAPropagationCallGraphBuilder builder = E.defaultCallGraphBuilder();

    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    // CAstCallGraphUtil.AVOID_DUMP = false;
    // CAstCallGraphUtil.dumpCG(((SSAPropagationCallGraphBuilder)builder).getCFAContextInterpreter(),
    // builder.getPointerAnalysis(), CG);
    // System.err.println(CG);

    TensorTypeAnalysis analysis = E.performAnalysis(builder);

    // Create a mapping from method signatures to pointer keys.
    Map<String, Set<LocalPointerKey>> methodSignatureToPointerKeys = new HashMap<>();

    // Create a mapping from method signatures to tensor variables.
    Map<String, Set<TensorVariable>> methodSignatureToTensorVariables = new HashMap<>();

    // for each pointer key, tensor variable pair.
    analysis.forEach(
        p -> {
          PointerKey pointerKey = p.fst;

          if (pointerKey instanceof LocalPointerKey) {
            LocalPointerKey localPointerKey = (LocalPointerKey) pointerKey;

            // get the call graph node associated with the
            CGNode node = localPointerKey.getNode();

            // get the method associated with the call graph node.
            IMethod method = node.getMethod();
            String methodSignature = method.getSignature();

            // associate the method to the pointer key.
            methodSignatureToPointerKeys.compute(
                methodSignature,
                (k, v) -> {
                  if (v == null) v = new HashSet<>();
                  v.add(localPointerKey);
                  return v;
                });

            TensorVariable tensorVariable = p.snd;

            // associate the method to the tensor variables.
            methodSignatureToTensorVariables.compute(
                methodSignature,
                (k, v) -> {
                  if (v == null) v = new HashSet<>();
                  v.add(tensorVariable);
                  return v;
                });
          } else logger.warning(() -> "Encountered: " + pointerKey.getClass());
        });

    // check the maps.
    assertEquals(expectedNumberOfTensorVariables, methodSignatureToPointerKeys.size());
    assertEquals(expectedNumberOfTensorVariables, methodSignatureToTensorVariables.size());

    final String functionSignature = "script " + filename + "." + functionName + ".do()LRoot;";

    // get the pointer keys for the function.
    Set<LocalPointerKey> functionPointerKeys = methodSignatureToPointerKeys.get(functionSignature);

    // check tensor parameters.
    assertEquals(expectedNumberOfTensorParameters, functionPointerKeys.size());

    // check value numbers.
    Set<Integer> actualValueNumberSet =
        functionPointerKeys.stream()
            .map(LocalPointerKey::getValueNumber)
            .collect(Collectors.toSet());

    assertEquals(expectedTensorParameterValueNumbers.length, actualValueNumberSet.size());
    Arrays.stream(expectedTensorParameterValueNumbers)
        .forEach(ev -> actualValueNumberSet.contains(ev));

    // get the tensor variables for the function.
    Set<TensorVariable> functionTensors = methodSignatureToTensorVariables.get(functionSignature);

    // check tensor parameters.
    assertEquals(expectedNumberOfTensorParameters, functionTensors.size());
  }
}
