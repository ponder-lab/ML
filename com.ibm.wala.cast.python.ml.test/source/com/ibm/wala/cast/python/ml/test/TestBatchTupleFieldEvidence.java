package com.ibm.wala.cast.python.ml.test;

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Regression guard for the batch tuple's element evidence (<a
 * href="https://github.com/wala/ML/issues/834">wala/ML#834</a>): the {@code (x, y)} tuple that
 * {@code flow_from_directory}'s summary materializes must carry its per-position element types on
 * the tuple's own instance-field keys, fields {@code "0"} and {@code "1"}, since that is where a
 * structure-consuming client reads element evidence. Reader-local types alone present the two
 * positions as indistinguishable parameter-level alternatives, which a sound consumer joins into
 * one wildcard where two positions were available.
 *
 * <p>The mechanism is the helper-call pattern (the keras dataset-loader precedent): a
 * summary-internal call named {@code do} is seeded, so the helper results' types flow through the
 * summary's {@code putfield} edges onto the field keys, where a direct allocation is seeded
 * nowhere. The fixture is the reduced witness from the issue.
 */
public class TestBatchTupleFieldEvidence extends TestPythonMLCallGraphShape {

  private static final String FIXTURE = "tf2_test_dataset74.py";

  /**
   * Both positional fields of the materialized batch tuple carry exactly their position's type:
   * field {@code "0"} the rank-4 images batch, field {@code "1"} the categorical labels.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testTupleFieldsCarryPositionalTypes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    Map<String, Set<TensorType>> fieldTypes = HashMapFactory.make();
    for (Iterator<Pair<PointerKey, TensorVariable>> it = analysis.iterator(); it.hasNext(); ) {
      Pair<PointerKey, TensorVariable> pair = it.next();
      if (!(pair.fst instanceof InstanceFieldKey)) continue;
      InstanceFieldKey fieldKey = (InstanceFieldKey) pair.fst;
      if (!(fieldKey.getInstanceKey() instanceof AllocationSiteInNode)) continue;
      AllocationSiteInNode allocation = (AllocationSiteInNode) fieldKey.getInstanceKey();
      if (!allocation.concreteType().getReference().getName().toString().equals("Ltuple")) continue;
      Set<TensorType> types = pair.snd.getTypes();
      if (types == null || types.isEmpty()) continue;
      fieldTypes
          .computeIfAbsent(fieldKey.getField().getName().toString(), k -> new java.util.HashSet<>())
          .addAll(types);
    }

    TensorType images =
        new TensorType(
            "float32",
            asList(DynamicDim.INSTANCE, new NumericDim(64), new NumericDim(64), new NumericDim(3)));
    TensorType labels =
        new TensorType("float32", asList(DynamicDim.INSTANCE, UnresolvedDim.INSTANCE));

    assertTrue("The tuple's field \"0\" must carry element evidence.", fieldTypes.containsKey("0"));
    assertTrue("The tuple's field \"1\" must carry element evidence.", fieldTypes.containsKey("1"));
    assertEquals(
        "Field \"0\" carries exactly the images position's type.",
        Set.of(images),
        fieldTypes.get("0"));
    assertEquals(
        "Field \"1\" carries exactly the labels position's type.",
        Set.of(labels),
        fieldTypes.get("1"));
  }
}
