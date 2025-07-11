package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.loader.AstMethod;
// import com.ibm.wala.cast.ipa.callgraph.CAstCallGraphUtil;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.SSAPropagationCallGraphBuilder;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import org.junit.Test;

public class TestTensorflowModelParsing extends TestPythonMLCallGraphShape {

  private static final Logger logger = Logger.getLogger(TestTensorflowModelParsing.class.getName());

  /** Test a parsing bug (see https://github.com/wala/ML/pull/46#issue-1743032519). */
  @Test
  public void testParsing()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    final String filename = "parsing_test.py";
    PythonAnalysisEngine<TensorTypeAnalysis> E = makeEngine(filename);
    PythonSSAPropagationCallGraphBuilder builder = E.defaultCallGraphBuilder();

    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    if (logger.isLoggable(Level.FINE)) {
      CAstCallGraphUtil.AVOID_DUMP.set(false);
      CAstCallGraphUtil.dumpCG(
          ((SSAPropagationCallGraphBuilder) builder).getCFAContextInterpreter(),
          builder.getPointerAnalysis(),
          CG);
      logger.fine("Call graph:\n" + CG);
    }

    TensorTypeAnalysis analysis = E.performAnalysis(builder);

    // Create a mapping from method signatures to pointer keys.
    Map<String, Set<LocalPointerKey>> methodSignatureToPointerKeys = new HashMap<>();

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
          } else
            logger.warning(() -> "Encountered pointer key type: " + pointerKey.getClass() + ".");
        });

    // we should have 3 methods.
    assertEquals(3, methodSignatureToPointerKeys.size());

    final String addFunctionSignature = "script " + filename + ".add.do()LRoot;";

    // get the pointer keys for the add() function.
    Set<LocalPointerKey> addFunctionPointerKeys =
        methodSignatureToPointerKeys.get(addFunctionSignature);

    // two tensor parameters, a and b.
    assertEquals(2, addFunctionPointerKeys.size());

    // should have value numbers of 2 and 3.
    Set<Integer> valueNumberSet =
        addFunctionPointerKeys.stream()
            .map(LocalPointerKey::getValueNumber)
            .collect(Collectors.toSet());
    assertEquals(2, valueNumberSet.size());
    assertTrue(valueNumberSet.contains(2));
    assertTrue(valueNumberSet.contains(3));

    // check the source positions of each function parameter.
    for (LocalPointerKey lpk : addFunctionPointerKeys) {
      AstMethod method = (AstMethod) lpk.getNode().getIR().getMethod();
      int paramIndex = lpk.getValueNumber() - 1;
      Position parameterPosition = method.getParameterPosition(paramIndex);

      // check the line.
      assertEquals(4, parameterPosition.getFirstLine());

      // check the columns.
      if (lpk.getValueNumber() == 2) {
        assertEquals(8, parameterPosition.getFirstCol());
        assertEquals(9, parameterPosition.getLastCol());
      } else if (lpk.getValueNumber() == 3) {
        assertEquals(11, parameterPosition.getFirstCol());
        assertEquals(12, parameterPosition.getLastCol());
      } else throw new IllegalStateException("Expecting value numbers 2 or 3.");
    }
  }
}
