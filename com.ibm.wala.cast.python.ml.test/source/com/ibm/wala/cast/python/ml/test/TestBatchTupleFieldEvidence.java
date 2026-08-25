package com.ibm.wala.cast.python.ml.test;

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.analysis.TensorVariable;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
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
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.io.IOException;
import java.util.Iterator;
import java.util.Locale;
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

  /** The {@code flow_from_directory} summary class name, the batch tuple's allocating class. */
  private static final String BATCH_TUPLE_ALLOCATING_CLASS =
      TensorFlowTypes.IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE.getName().toString();

  /** The {@code np.unique} summary class name, the foreign tuple's allocating class. */
  private static final String UNIQUE_ALLOCATING_CLASS = "Lnumpy/unique";

  private static final String FLOAT32 = DType.FLOAT32.name().toLowerCase(Locale.ROOT);

  private static final String INT64 = DType.INT64.name().toLowerCase(Locale.ROOT);

  private static final TensorType IMAGES_64 =
      new TensorType(
          FLOAT32,
          asList(DynamicDim.INSTANCE, new NumericDim(64), new NumericDim(64), new NumericDim(3)));

  private static final TensorType IMAGES_96 =
      new TensorType(
          FLOAT32,
          asList(DynamicDim.INSTANCE, new NumericDim(96), new NumericDim(96), new NumericDim(3)));

  private static final TensorType LABELS =
      new TensorType(FLOAT32, asList(DynamicDim.INSTANCE, UnresolvedDim.INSTANCE));

  private static final TensorType UNIQUE_INDICES =
      new TensorType(INT64, asList(UnresolvedDim.INSTANCE));

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
    Map<String, Map<String, Set<TensorType>>> byClass = tupleFieldTypesByAllocatingClass(FIXTURE);
    Map<String, Set<TensorType>> fieldTypes = byClass.get(BATCH_TUPLE_ALLOCATING_CLASS);

    assertNotNull("The batch tuple must carry element evidence.", fieldTypes);
    assertEquals(
        "Field \"0\" carries exactly the images position's type.",
        Set.of(IMAGES_64),
        fieldTypes.get("0"));
    assertEquals(
        "Field \"1\" carries exactly the labels position's type.",
        Set.of(LABELS),
        fieldTypes.get("1"));
  }

  /**
   * The multi-call-site form: both sites' batch tuples union per position under the batch tuple's
   * allocating class, while the {@code np.unique} tuple in the same fixture, a foreign tuple whose
   * fields also carry tensor types, stays under its own class. The foreign evidence is asserted
   * PRESENT before it is asserted excluded, so the exclusion cannot pass vacuously, and per-SITE
   * attribution is deliberately out of scope here: this test checks the per-class union only, and
   * cross-site attribution is {@code TestDatasets#testDataset76}'s parameter-level job.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testFilterExcludesForeignTuples()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    Map<String, Map<String, Set<TensorType>>> byClass =
        tupleFieldTypesByAllocatingClass("tf2_test_dataset76.py");

    // The precondition: the foreign tuple's evidence exists to be excluded.
    Map<String, Set<TensorType>> unique = byClass.get(UNIQUE_ALLOCATING_CLASS);
    assertNotNull("The np.unique tuple must carry field evidence of its own.", unique);
    assertEquals(
        "The unique tuple's index position carries the int64 indices type.",
        Set.of(UNIQUE_INDICES),
        unique.get("1"));

    Map<String, Set<TensorType>> batch = byClass.get(BATCH_TUPLE_ALLOCATING_CLASS);
    assertNotNull("The batch tuples must carry element evidence.", batch);
    assertEquals(
        "Field \"0\" carries both sites' image types and nothing foreign.",
        Set.of(IMAGES_64, IMAGES_96),
        batch.get("0"));
    assertEquals(
        "Field \"1\" carries the labels type and nothing foreign.", Set.of(LABELS), batch.get("1"));
  }

  /**
   * Runs the analysis on a fixture and collects the tensor types on every tuple's instance-field
   * keys, grouped by the tuple's allocating summary class, so a test can assert both what a class's
   * tuples carry and that another class's evidence exists.
   *
   * @param fixture The Python fixture to analyze.
   * @return The types keyed by allocating class name, then by field name.
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws IllegalArgumentException if the input fixture is malformed.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  private Map<String, Map<String, Set<TensorType>>> tupleFieldTypesByAllocatingClass(String fixture)
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, emptyList(), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    TensorTypeAnalysis analysis = engine.performAnalysis(builder);

    Map<String, Map<String, Set<TensorType>>> ret = HashMapFactory.make();
    for (Iterator<Pair<PointerKey, TensorVariable>> it = analysis.iterator(); it.hasNext(); ) {
      Pair<PointerKey, TensorVariable> pair = it.next();
      if (!(pair.fst instanceof InstanceFieldKey)) continue;
      InstanceFieldKey fieldKey = (InstanceFieldKey) pair.fst;
      if (!(fieldKey.getInstanceKey() instanceof AllocationSiteInNode)) continue;
      AllocationSiteInNode allocation = (AllocationSiteInNode) fieldKey.getInstanceKey();
      if (!allocation.concreteType().getReference().getName().toString().equals("Ltuple")) continue;
      // Group by the allocating summary class. The name is matched EXACTLY: summary `do` nodes are
      // declared on the summary class itself (trampoline generation suffixes the method name on a
      // separately prefixed class, so no trampoline case reaches here), and a loose prefix would
      // silently admit a future sibling class whose name textually extends this one.
      String allocatingClass =
          allocation.getNode().getMethod().getDeclaringClass().getReference().getName().toString();
      Set<TensorType> types = pair.snd.getTypes();
      if (types == null || types.isEmpty()) continue;
      ret.computeIfAbsent(allocatingClass, k -> HashMapFactory.make())
          .computeIfAbsent(fieldKey.getField().getName().toString(), k -> HashSetFactory.make())
          .addAll(types);
    }
    return ret;
  }
}
