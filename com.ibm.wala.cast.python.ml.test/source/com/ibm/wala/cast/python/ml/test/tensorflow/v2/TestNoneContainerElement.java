package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;

/**
 * Witnesses for {@link TensorGenerator#anyNullConstant} (wala/ML#867), asserted against a real
 * analysis run rather than a hand-built set so each can fail if the substrate moves. The predicate
 * exists for consumers making a universal claim over the values that may arrive at a container
 * element: a claim defeated by a single {@code None} member even when tensor may-evidence sits
 * beside it on the same {@link InstanceFieldKey}.
 *
 * <p>{@link #testNoneBesideTensorIsDetected} pins the dual-channel configuration wala/ML#867 is
 * about: a leading element that is {@code None} on every reachable runtime path while a statically
 * feasible branch behind a data-dependent guard assigns an ndarray, leaving one field key carrying
 * tensor state <em>and</em> a {@code ConstantKey(null)}. It fails if the pointer analysis stops
 * propagating the {@code None} constant into the element field's points-to set (the predicate's
 * reason to exist disappears) or if the tensor may-state vanishes (the configuration degenerates
 * into the plain no-evidence case a consumer already handles). Two siblings keep the predicate
 * falsifiable in the other direction: {@link #testAllTensorElementsAreNotFlagged} (the same
 * geometry with every element unconditionally an ndarray) and {@link
 * #testLiteralBesideTensorIsNotFlagged} (the dual-channel geometry with an ordinary literal in
 * place of {@code None}, so a predicate mistakenly flagging any constant fails it).
 */
public class TestNoneContainerElement extends AbstractTensorTest {

  @Test
  public void testNoneBesideTensorIsDetected() throws Exception {
    List<Pair<Boolean, Boolean>> keys =
        leadingElementFieldKeys("tf2_test_none_container_element.py");
    assertFalse("Expecting the read-side list's leading element field key.", keys.isEmpty());

    for (Pair<Boolean, Boolean> key : keys) {
      assertTrue("Expecting the None constant in the element field's points-to set.", key.fst);
      assertTrue("Expecting tensor may-state beside the None constant on the same key.", key.snd);
    }
  }

  @Test
  public void testLiteralBesideTensorIsNotFlagged() throws Exception {
    List<Pair<Boolean, Boolean>> keys =
        leadingElementFieldKeys("tf2_test_literal_container_element.py");
    assertFalse("Expecting the read-side list's leading element field key.", keys.isEmpty());

    for (Pair<Boolean, Boolean> key : keys) {
      assertFalse(
          "Expecting an ordinary literal beside tensor state not to be flagged: the predicate"
              + " tests None-possibility, not constant-ness.",
          key.fst);
      assertTrue("Expecting tensor may-state beside the literal on the same key.", key.snd);
    }
  }

  @Test
  public void testAllTensorElementsAreNotFlagged() throws Exception {
    List<Pair<Boolean, Boolean>> keys =
        leadingElementFieldKeys("tf2_test_tensor_container_element.py");
    assertFalse("Expecting the read-side list's leading element field key.", keys.isEmpty());

    for (Pair<Boolean, Boolean> key : keys) {
      assertFalse("Expecting no None constant in the element field's points-to set.", key.fst);
      assertTrue("Expecting tensor may-state on the element field key.", key.snd);
    }
  }

  /**
   * Runs the tensor analysis on the given fixture and returns, for each field-{@code 0} key of the
   * list allocated in the fixture's {@code read} function, whether {@link
   * TensorGenerator#anyNullConstant} flags its points-to set (first) and whether the analysis
   * carries non-empty tensor state on the same key (second).
   *
   * @param fixture The Python fixture file name.
   * @return One pair per matching field key; empty if the fixture's geometry no longer produces
   *     one.
   */
  private List<Pair<Boolean, Boolean>> leadingElementFieldKeys(String fixture) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<java.io.File>emptyList(), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    List<Pair<Boolean, Boolean>> result = new ArrayList<>();

    for (Pair<PointerKey, TensorVariable> pair : analysis) {
      if (!(pair.fst instanceof InstanceFieldKey)) continue;

      InstanceFieldKey fieldKey = (InstanceFieldKey) pair.fst;
      InstanceKey container = fieldKey.getInstanceKey();

      if (!(container instanceof AllocationSiteInNode)) continue;

      AllocationSiteInNode allocation = (AllocationSiteInNode) container;
      String allocatingMethod =
          allocation.getNode().getMethod().getDeclaringClass().getName().toString();

      if (allocatingMethod.endsWith("/read")
          && fieldKey.getField().getName().toString().equals("0"))
        result.add(
            Pair.make(
                TensorGenerator.anyNullConstant(
                    builder.getPointerAnalysis().getPointsToSet(fieldKey)),
                pair.snd != null && !pair.snd.getTypes().isEmpty()));
    }

    return result;
  }
}
