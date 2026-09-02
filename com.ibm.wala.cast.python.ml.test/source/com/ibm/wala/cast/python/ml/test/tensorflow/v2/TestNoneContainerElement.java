package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
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
 * place of {@code None}, so a predicate mistakenly flagging any constant fails it). Each witness
 * also pins the points-to facts it depends on: the literal case asserts the non-null {@link
 * ConstantKey} is actually IN the element field's points-to set, so the not-flagged assertion
 * cannot be satisfied vacuously by a field the substrate stopped propagating the literal into.
 */
public class TestNoneContainerElement extends AbstractTensorTest {

  /**
   * The facts this class asserts about one leading-element field key: whether {@link
   * TensorGenerator#anyNullConstant} flags its points-to set, whether the analysis carries
   * non-empty tensor state on the same key, and whether the points-to set actually contains a
   * non-null {@link ConstantKey} (the fact the literal witness's not-flagged assertion depends on
   * to be non-vacuous).
   */
  private record ElementKeyFacts(
      boolean flaggedAsPossiblyNone, boolean hasTensorState, boolean hasNonNullConstant) {}

  @Test
  public void testNoneBesideTensorIsDetected() throws Exception {
    List<ElementKeyFacts> keys = leadingElementFieldKeys("tf2_test_none_container_element.py");
    assertFalse("Expecting the read-side list's leading element field key.", keys.isEmpty());

    for (ElementKeyFacts key : keys) {
      assertTrue(
          "Expecting the None constant in the element field's points-to set.",
          key.flaggedAsPossiblyNone());
      assertTrue(
          "Expecting tensor may-state beside the None constant on the same key.",
          key.hasTensorState());
      assertFalse(
          "Expecting no non-null constant in this fixture's element field.",
          key.hasNonNullConstant());
    }
  }

  @Test
  public void testLiteralBesideTensorIsNotFlagged() throws Exception {
    List<ElementKeyFacts> keys = leadingElementFieldKeys("tf2_test_literal_container_element.py");
    assertFalse("Expecting the read-side list's leading element field key.", keys.isEmpty());

    for (ElementKeyFacts key : keys) {
      assertTrue(
          "Expecting the literal's non-null constant in the element field's points-to set: without"
              + " it the not-flagged assertion below is vacuous, satisfied by a field the"
              + " substrate stopped propagating the literal into.",
          key.hasNonNullConstant());
      assertFalse(
          "Expecting an ordinary literal beside tensor state not to be flagged: the predicate"
              + " tests None-possibility, not constant-ness.",
          key.flaggedAsPossiblyNone());
      assertTrue(
          "Expecting tensor may-state beside the literal on the same key.", key.hasTensorState());
    }
  }

  @Test
  public void testAllTensorElementsAreNotFlagged() throws Exception {
    List<ElementKeyFacts> keys = leadingElementFieldKeys("tf2_test_tensor_container_element.py");
    assertFalse("Expecting the read-side list's leading element field key.", keys.isEmpty());

    for (ElementKeyFacts key : keys) {
      assertFalse(
          "Expecting no None constant in the element field's points-to set.",
          key.flaggedAsPossiblyNone());
      assertTrue("Expecting tensor may-state on the element field key.", key.hasTensorState());
      assertFalse(
          "Expecting no constant of any kind in this fixture's element field.",
          key.hasNonNullConstant());
    }
  }

  /**
   * Runs the tensor analysis on the given fixture and returns the {@link ElementKeyFacts} for each
   * field-{@code 0} key of the list allocated in the fixture's {@code read} function.
   *
   * @param fixture The Python fixture file name.
   * @return One entry per matching field key; empty if the fixture's geometry no longer produces
   *     one.
   */
  private List<ElementKeyFacts> leadingElementFieldKeys(String fixture) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<java.io.File>emptyList(), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);
    List<ElementKeyFacts> result = new ArrayList<>();

    for (Pair<PointerKey, TensorVariable> pair : analysis) {
      if (!(pair.fst instanceof InstanceFieldKey)) continue;

      InstanceFieldKey fieldKey = (InstanceFieldKey) pair.fst;
      InstanceKey container = fieldKey.getInstanceKey();

      if (!(container instanceof AllocationSiteInNode)) continue;

      AllocationSiteInNode allocation = (AllocationSiteInNode) container;
      String allocatingMethod =
          allocation.getNode().getMethod().getDeclaringClass().getName().toString();

      if (allocatingMethod.endsWith("/read")
          && fieldKey.getField().getName().toString().equals("0")) {
        boolean nonNullConstant = false;
        for (InstanceKey member : builder.getPointerAnalysis().getPointsToSet(fieldKey))
          if (member instanceof ConstantKey && ((ConstantKey<?>) member).getValue() != null) {
            nonNullConstant = true;
            break;
          }

        result.add(
            new ElementKeyFacts(
                TensorGenerator.anyNullConstant(
                    builder.getPointerAnalysis().getPointsToSet(fieldKey)),
                pair.snd != null && !pair.snd.getTypes().isEmpty(),
                nonNullConstant));
      }
    }

    return result;
  }
}
