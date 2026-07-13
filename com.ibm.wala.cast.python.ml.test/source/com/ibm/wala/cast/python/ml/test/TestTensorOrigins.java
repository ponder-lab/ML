package com.ibm.wala.cast.python.ml.test;

import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CallGraph;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for the numpy-origin vs TensorFlow-origin record on tensor-typed values (wala/ML#724).
 *
 * <p>Assertions target the first parameter (value number 2) of a sink function, per the
 * calling-context convention: the fixture routes each value of interest through {@code
 * consume_np(x)} or {@code consume_tf(x)}, and the parameter's origins are the union across calling
 * contexts, mirroring the type-assertion semantics of {@code TestTensorflow2Model.test}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestTensorOrigins extends TestPythonMLCallGraphShape {

  /**
   * The three instruction shapes from wala/ML#724 flow to {@code consume_np}: a binary operator
   * over numpy operands ({@code a + b}), an ndarray method ({@code m.reshape(...)}), and an
   * interprocedural return of {@code np.array(...)}. All are numpy-origin. The mixed binary
   * operator ({@code ndarray + Tensor}) flows to {@code consume_tf}: runtime dispatch makes it a
   * TensorFlow operation and its result a {@code tf.Tensor}.
   *
   * @throws ClassHierarchyException If the class hierarchy cannot be built.
   * @throws CancelException If the analysis is canceled.
   * @throws IOException If the test file cannot be read.
   */
  @Test
  public void testNumpyOrigin() throws ClassHierarchyException, CancelException, IOException {
    Map<String, Set<TensorOrigin>> sinkParameterOrigins =
        getSinkParameterOrigins("tf2_test_numpy_origin.py", "consume_np", "consume_tf");

    assertEquals(EnumSet.of(TensorOrigin.NUMPY), sinkParameterOrigins.get("consume_np"));
    assertEquals(EnumSet.of(TensorOrigin.TENSORFLOW), sinkParameterOrigins.get("consume_tf"));
  }

  /**
   * Runs the tensor analysis on the given file and collects the origins of each named sink
   * function's first parameter (value number 2), unioned across calling contexts.
   *
   * @param filename The Python test file to analyze.
   * @param sinkFunctionNames The sink functions whose first-parameter origins are collected.
   * @return A map from sink function name to the union of its first parameter's origins; a sink
   *     absent from the map received no tensor-typed state at all.
   * @throws ClassHierarchyException If the class hierarchy cannot be built.
   * @throws CancelException If the analysis is canceled.
   * @throws IOException If the test file cannot be read.
   */
  private Map<String, Set<TensorOrigin>> getSinkParameterOrigins(
      String filename, String... sinkFunctionNames)
      throws ClassHierarchyException, CancelException, IOException {
    PythonTensorAnalysisEngine engine = makeEngine(emptyList(), filename);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();

    CallGraph CG = builder.makeCallGraph(builder.getOptions());
    assertNotNull(CG);

    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    Map<String, Set<TensorOrigin>> ret = new HashMap<>();

    analysis.forEach(
        pt -> {
          if (!(pt.fst instanceof LocalPointerKey)) return;
          LocalPointerKey localPointerKey = (LocalPointerKey) pt.fst;

          // The first explicit parameter (value number 2; value number 1 is the function object).
          if (localPointerKey.getValueNumber() != 2) return;

          String signature = localPointerKey.getNode().getMethod().getSignature();

          for (String sink : sinkFunctionNames)
            if (signature.contains("/" + sink + ".do("))
              ret.computeIfAbsent(sink, k -> EnumSet.noneOf(TensorOrigin.class))
                  .addAll(pt.snd.getOrigins());
        });

    return ret;
  }
}
