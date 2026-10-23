package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@link ContainerElements}: the length, element keys, per-position tensor types and {@code
 * None} elements the analysis reports for a container, on {@code tf2_test_container_elements.py},
 * where each sink receives one container. A length is reported only when it is the run-time one;
 * every form whose length or element order the analysis cannot vouch for reports none.
 */
public class TestContainerElements extends TestPythonMLCallGraphShape {

  private static final TensorType F32_2_3 = TensorType.of(FLOAT32, 2, 3);

  private static final TensorType I32_4 = TensorType.of(INT32, 4);

  /** The fixture most tests read. */
  private static final String FILE = "tf2_test_container_elements.py";

  /** The fixture of lists a summary or operator allocates, and of unrecorded mutations. */
  private static final String MUTATORS = "tf2_test_container_elements_mutators.py";

  /** The fixture of subscript {@code del} statements. */
  private static final String DELETES = "tf2_test_container_elements_del.py";

  /** The fixture of lists mutated only in callbacks no node runs, and through {@code getattr}. */
  private static final String GLOBALS = "tf2_test_container_elements_globals.py";

  /** The fixture of dicts keyed by floats, bools, {@code None} and an unresolved key. */
  private static final String DICT_KEYS = "tf2_test_container_elements_dict_keys.py";

  /** The fixture of sets grown by {@code add}, {@code |=} and {@code update}. */
  private static final String SETS = "tf2_test_container_elements_sets.py";

  /** The fixture of a {@code del} of a name only. */
  private static final String DELETES_NAME = "tf2_test_container_elements_del_name.py";

  /** Per fixture, its builder. */
  private static final Map<String, PythonSSAPropagationCallGraphBuilder> BUILDERS = new HashMap<>();

  /** Per fixture, its solved analysis. */
  private static final Map<String, TensorTypeAnalysis> ANALYSES = new HashMap<>();

  /**
   * Analyzes a fixture, once.
   *
   * @param file The fixture.
   * @return Its builder.
   */
  private static PythonSSAPropagationCallGraphBuilder builder(String file) {
    return BUILDERS.computeIfAbsent(
        file,
        f -> {
          try {
            PythonTensorAnalysisEngine engine =
                new TestContainerElements().makeEngine(Collections.<File>emptyList(), f);
            PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
            b.makeCallGraph(b.getOptions());
            ANALYSES.put(f, engine.performAnalysis(b));
            return b;
          } catch (Exception e) {
            throw new IllegalStateException(e);
          }
        });
  }

  /**
   * The solved analysis of the main fixture.
   *
   * @return The analysis.
   */
  private static TensorTypeAnalysis analysis() {
    return analysis(FILE);
  }

  /**
   * The solved analysis of a fixture.
   *
   * @param file The fixture.
   * @return The analysis.
   */
  private static TensorTypeAnalysis analysis(String file) {
    builder(file);
    return ANALYSES.get(file);
  }

  /**
   * The elements of the one container a sink of the main fixture receives.
   *
   * @param sink The sink's name.
   * @return Its parameter's container's elements.
   */
  private static ContainerElements elements(String sink) {
    return elements(FILE, sink);
  }

  /**
   * The elements of the one container a sink receives.
   *
   * @param file The fixture.
   * @param sink The sink's name.
   * @return Its parameter's container's elements.
   */
  private static ContainerElements elements(String file, String sink) {
    PythonSSAPropagationCallGraphBuilder builder = builder(file);
    List<InstanceKey> containers = new ArrayList<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) {
        PointerKey parameter =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
        for (InstanceKey container : builder.getPointerAnalysis().getPointsToSet(parameter))
          if (!containers.contains(container)) containers.add(container);
      }
    assertEquals("Expecting one container at " + sink + ".", 1, containers.size());
    return ContainerElements.of(builder, containers.get(0));
  }

  /**
   * Asserts the length a sink's container reports.
   *
   * @param file The fixture.
   * @param sink The sink's name.
   * @param expected The expected length: the run-time one, or empty.
   */
  private static void assertArity(String file, String sink, OptionalInt expected) {
    assertEquals(file + " " + sink, expected, elements(file, sink).arity());
  }

  /**
   * Asserts that no container reaches a sink: no length is reported for it.
   *
   * @param file The fixture.
   * @param sink The sink's name.
   */
  private static void assertNoContainer(String file, String sink) {
    PythonSSAPropagationCallGraphBuilder builder = builder(file);
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
        assertTrue(
            sink,
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2))
                .isEmpty());
  }

  /**
   * Asserts that a container reports no length, and so no positions, but still has element keys.
   *
   * @param sink The sink's name.
   */
  private static void assertNoArity(String sink) {
    ContainerElements elements = elements(sink);
    assertEquals(sink, OptionalInt.empty(), elements.arity());
    assertTrue(sink, elements.positions().isEmpty());
    assertTrue(sink, elements.positionTypes(analysis()).isEmpty());
    assertFalse(sink, elements.allElements().isEmpty());
  }

  /** A tuple literal of two tensors: two positions, each with its tensor's type. */
  @Test
  public void testTuple() {
    ContainerElements elements = elements("sink_tuple");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(2, elements.positions().size());
    assertEquals(List.of(Set.of(F32_2_3), Set.of(I32_4)), elements.positionTypes(analysis()));
    assertFalse(elements.mayHoldNone());
  }

  /** A list literal of three tensors. */
  @Test
  public void testList() {
    ContainerElements elements = elements("sink_list");
    assertEquals(OptionalInt.of(3), elements.arity());
    assertEquals(
        List.of(Set.of(F32_2_3), Set.of(I32_4), Set.of(F32_2_3)),
        elements.positionTypes(analysis()));
  }

  /** A list only read through a method that changes neither its length nor its order. */
  @Test
  public void testListReadThroughCopy() {
    assertEquals(OptionalInt.of(2), elements("sink_counted").arity());
  }

  /** A dict has no positions, but its values are elements. */
  @Test
  public void testDict() {
    ContainerElements elements = elements("sink_dict");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertEquals(2, elements.allElements().size());
  }

  /** A position holding a tuple has no tensor type of its own. */
  @Test
  public void testNested() {
    ContainerElements elements = elements("sink_nested");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(List.of(Set.of(F32_2_3), Set.of()), elements.positionTypes(analysis()));
  }

  /** A {@code None} element, against the same tuple without one. */
  @Test
  public void testNoneElement() {
    assertTrue(elements("sink_with_none").mayHoldNone());
    assertFalse(elements("sink_without_none").mayHoldNone());
  }

  /** An appended element leaves the length unknown. */
  @Test
  public void testAppended() {
    assertNoArity("sink_appended");
  }

  /** A comprehension's length is unknown. */
  @Test
  public void testComprehension() {
    assertNoArity("sink_comprehension");
  }

  /** A starred element, {@code [*a, x]}, leaves every position unknown. */
  @Test
  public void testStarred() {
    assertNoArity("sink_starred");
  }

  /** {@code l.extend(m)}, which the analysis does not record among the list's elements. */
  @Test
  public void testExtended() {
    assertNoArity("sink_extended");
  }

  /** A list built by concatenation, {@code [x] + [y, x]}. */
  @Test
  public void testConcatenated() {
    assertNoArity("sink_concatenated");
  }

  /** A list built by repetition, {@code [x, y] * 2}. */
  @Test
  public void testRepeated() {
    assertNoArity("sink_repeated");
  }

  /** An augmented assignment, {@code l += [y]}. */
  @Test
  public void testAugmented() {
    assertNoArity("sink_augmented");
  }

  /** A slice assignment, {@code l[0:1] = [y, x]}. */
  @Test
  public void testSliceAssigned() {
    assertNoArity("sink_slice_assigned");
  }

  /** {@code l.insert(...)} in a callee the list is passed to. */
  @Test
  public void testMutatedElsewhere() {
    assertNoArity("sink_mutated_elsewhere");
  }

  /** {@code list.append(l, y)}: the list is an argument of the unbound method. */
  @Test
  public void testUnboundAppend() {
    assertArity(MUTATORS, "sink_unbound_append", OptionalInt.empty());
  }

  /** {@code getattr(l, "append")(y)}. */
  @Test
  public void testGetattrAppend() {
    assertArity(MUTATORS, "sink_getattr_append", OptionalInt.empty());
  }

  /** {@code operator.iadd(l, [y])}, a call that resolves to no target. */
  @Test
  public void testOperatorIadd() {
    assertArity(MUTATORS, "sink_operator_iadd", OptionalInt.empty());
  }

  /** {@code heapq.heappush(l, 2)}, a call that resolves to no target. */
  @Test
  public void testHeappush() {
    assertArity(MUTATORS, "sink_heappush", OptionalInt.empty());
  }

  /** {@code bisect.insort(l, 2)}, a call that resolves to no target. */
  @Test
  public void testInsort() {
    assertArity(MUTATORS, "sink_insort", OptionalInt.empty());
  }

  /** An empty list after {@code heapq.heappush}. */
  @Test
  public void testEmptyHeappush() {
    assertArity(MUTATORS, "sink_empty_heappush", OptionalInt.empty());
  }

  /** A mutation in a {@code key} lambda of {@code sorted}, which no call-graph node runs. */
  @Test
  public void testSortedKeyLambda() {
    assertArity(MUTATORS, "sink_sorted_key_lambda", OptionalInt.empty());
  }

  /** A mutation in a lambda {@code map} applies, which no call-graph node runs. */
  @Test
  public void testMapLambda() {
    assertArity(MUTATORS, "sink_map_lambda", OptionalInt.empty());
  }

  /**
   * {@code tuple(l)} of an appended list: no container reaches the sink, so none reports a length.
   */
  @Test
  public void testTupleOfAppended() {
    assertNoContainer(MUTATORS, "sink_tuple_of_appended");
  }

  /** {@code list(l)} of an appended list: a summary's allocation. */
  @Test
  public void testListOfAppended() {
    assertArity(MUTATORS, "sink_list_of_appended", OptionalInt.empty());
  }

  /**
   * {@code l.copy()} of an appended list: no container reaches the sink, so none reports a length.
   */
  @Test
  public void testCopyOfAppended() {
    assertNoContainer(MUTATORS, "sink_copy_of_appended");
  }

  /** {@code l *= 2}. */
  @Test
  public void testImul() {
    assertArity(MUTATORS, "sink_imul", OptionalInt.empty());
  }

  /** A slice insertion, {@code l[1:1] = [x]}. */
  @Test
  public void testSliceInsert() {
    assertArity(MUTATORS, "sink_slice_insert", OptionalInt.empty());
  }

  /** {@code h.items.append(y)}, through an attribute. */
  @Test
  public void testAttributeAppend() {
    assertArity(MUTATORS, "sink_attr_append", OptionalInt.empty());
  }

  /** {@code add = l.append; add(y)}, through a bound method object. */
  @Test
  public void testBoundMethod() {
    assertArity(MUTATORS, "sink_bound_method", OptionalInt.empty());
  }

  /** An append in a comprehension. */
  @Test
  public void testComprehensionMutation() {
    assertArity(MUTATORS, "sink_comp_mutation", OptionalInt.empty());
  }

  /** {@code list.extend(l, ...)} in a callee. */
  @Test
  public void testUnboundExtendInCallee() {
    assertArity(MUTATORS, "sink_unbound_extend_callee", OptionalInt.empty());
  }

  /** {@code "a,b,c".split(",")}: a summary's allocation. */
  @Test
  public void testStringSplit() {
    assertArity(MUTATORS, "sink_split", OptionalInt.empty());
  }

  /** {@code list(range(3))}: a summary's allocation. */
  @Test
  public void testListOfRange() {
    assertArity(MUTATORS, "sink_list_range", OptionalInt.empty());
  }

  /** {@code sorted({3, 1, 2})}: a summary's allocation. */
  @Test
  public void testSortedSet() {
    assertArity(MUTATORS, "sink_sorted_set", OptionalInt.empty());
  }

  /** {@code tf.unstack(tf.ones([2, 3]))}: an operator model's allocation. */
  @Test
  public void testUnstack() {
    assertArity(MUTATORS, "sink_unstack", OptionalInt.empty());
  }

  /** A tuple {@code zip} yields: a summary's allocation. */
  @Test
  public void testZipTuple() {
    assertArity(MUTATORS, "sink_zip_tuple", OptionalInt.empty());
  }

  /** {@code range(3)}: a summary's allocation. */
  @Test
  public void testRange() {
    assertArity(MUTATORS, "sink_range_obj", OptionalInt.empty());
  }

  /** {@code tf.split(t, 4)}: an operator model's allocation. */
  @Test
  public void testTensorSplit() {
    assertArity(MUTATORS, "sink_split_tf", OptionalInt.empty());
  }

  /** {@code d.items()}: a summary's allocation. */
  @Test
  public void testDictItems() {
    assertArity(MUTATORS, "sink_items", OptionalInt.empty());
  }

  /** {@code list(enumerate(...))}: a summary's allocation. */
  @Test
  public void testListOfEnumerate() {
    assertArity(MUTATORS, "sink_enumerate_list", OptionalInt.empty());
  }

  /**
   * A list a TensorFlow function reads, {@code tf.image.random_crop(t, size)}, keeps its length,
   * though the call resolves to no target.
   */
  @Test
  public void testTensorFlowArgumentKeepsArity() {
    assertArity(MUTATORS, "sink_crop_size", OptionalInt.of(3));
  }

  /** {@code del l[0]} in a callee. */
  @Test
  public void testCalleeDelete() {
    assertArity(DELETES, "sink_callee_del", OptionalInt.empty());
  }

  /** {@code del G[0]} of a global in a function. */
  @Test
  public void testGlobalDelete() {
    assertArity(DELETES, "sink_global_del", OptionalInt.empty());
  }

  /** {@code del o.inner.items[0]}, two fields deep. */
  @Test
  public void testDeepDelete() {
    assertArity(DELETES, "sink_deep_del", OptionalInt.empty());
  }

  /** {@code del d["a"]["b"][0]}, a dict two deep. */
  @Test
  public void testDictDeepDelete() {
    assertArity(DELETES, "sink_dict_deep_del", OptionalInt.empty());
  }

  /** {@code del cl[0]} of a closure's list. */
  @Test
  public void testClosureDelete() {
    assertArity(DELETES, "sink_closure_del", OptionalInt.empty());
  }

  /** A list no {@code del} touches reports none either, since the deleted list is unknown. */
  @Test
  public void testUntouchedListBesideDelete() {
    assertArity(DELETES, "sink_untouched", OptionalInt.empty());
  }

  /** A tuple, which no {@code del} can shorten, keeps its length beside one. */
  @Test
  public void testTupleBesideDelete() {
    assertArity(DELETES, "sink_tuple", OptionalInt.of(2));
  }

  /** A {@code del} of a name only, {@code del v}, deletes no element: a list keeps its length. */
  @Test
  public void testNameDeleteKeepsArity() {
    assertArity(DELETES_NAME, "sink_list", OptionalInt.of(2));
  }

  /**
   * A module-level list mutated only in a lambda {@code sorted} never calls: the lambda reads it as
   * a variable of the module's frame.
   */
  @Test
  public void testLambdaReadsModuleVariable() {
    assertArity(GLOBALS, "sink_lambda_global", OptionalInt.empty());
  }

  /**
   * A module-level list mutated only in a function {@code sorted} never calls, which declares it
   * {@code global}: the function reads it as a global.
   */
  @Test
  public void testUnreachedFunctionReadsGlobal() {
    assertArity(GLOBALS, "sink_declared_global", OptionalInt.empty());
  }

  /**
   * {@code getattr(l, "append")(y)}: {@code getattr} is imported, not summarized, so unresolved.
   */
  @Test
  public void testGetattrCall() {
    assertArity(GLOBALS, "sink_getattr", OptionalInt.empty());
  }

  /**
   * An appended list of tensors has no length, so no position types, but its elements' types, those
   * of the appended element included.
   */
  @Test
  public void testElementTypesOfAppended() {
    ContainerElements elements = elements("sink_appended");
    assertTrue(elements.positionTypes(analysis()).isEmpty());
    assertEquals(Set.of(F32_2_3, I32_4), elements.elementTypes(analysis()));
  }

  /**
   * A comprehension of tensors has no length, but its elements' types: {@code tf.ones([i + 1])} is
   * a rank-1 float32 tensor whose extent the analysis does not fold.
   */
  @Test
  public void testElementTypesOfComprehension() {
    ContainerElements elements = elements("sink_comprehension");
    assertTrue(elements.positionTypes(analysis()).isEmpty());
    assertEquals(
        Set.of(new TensorType("float32", List.of(UnresolvedDim.INSTANCE))),
        elements.elementTypes(analysis()));
  }

  /** A list of ints has a length, but no element with tensor evidence. */
  @Test
  public void testElementTypesOfInts() {
    ContainerElements elements = elements("sink_ints");
    assertEquals(OptionalInt.of(3), elements.arity());
    assertEquals(Set.of(), elements.elementTypes(analysis()));
  }

  /** A tuple's element types are the union of its position types. */
  @Test
  public void testElementTypesOfTuple() {
    assertEquals(Set.of(F32_2_3, I32_4), elements("sink_tuple").elementTypes(analysis()));
  }

  /** A dict keyed by a float: its tensor value is an element. */
  @Test
  public void testFloatKeyElementTypes() {
    ContainerElements elements = elements(DICT_KEYS, "sink_float_tensor");
    assertTrue(elements.isComplete());
    assertEquals(Set.of(F32_2_3), elements.elementTypes(analysis(DICT_KEYS)));
    assertFalse(elements.mayHoldNone());
  }

  /** A dict keyed by a float whose value is {@code None}. */
  @Test
  public void testFloatKeyNone() {
    assertTrue(elements(DICT_KEYS, "sink_float_none").mayHoldNone());
  }

  /** A dict keyed by a bool: its tensor value is an element. */
  @Test
  public void testBoolKeyElementTypes() {
    ContainerElements elements = elements(DICT_KEYS, "sink_bool_tensor");
    assertTrue(elements.isComplete());
    assertEquals(Set.of(F32_2_3), elements.elementTypes(analysis(DICT_KEYS)));
    assertFalse(elements.mayHoldNone());
  }

  /** A dict keyed by a bool whose value is {@code None}. */
  @Test
  public void testBoolKeyNone() {
    assertTrue(elements(DICT_KEYS, "sink_bool_none").mayHoldNone());
  }

  /** A dict keyed by {@code None} whose value is {@code None}. */
  @Test
  public void testNoneKeyNone() {
    ContainerElements elements = elements(DICT_KEYS, "sink_none_key");
    assertTrue(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /**
   * A dict stored into under a key the analysis does not resolve, {@code d[k] = None}: its keys are
   * not all known, so it may hold {@code None}.
   */
  @Test
  public void testUnresolvedKey() {
    ContainerElements elements = elements(DICT_KEYS, "sink_unresolved_key");
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /** A dict of constant keys only has all its keys. */
  @Test
  public void testConstantKeysComplete() {
    assertTrue(elements("sink_dict").isComplete());
  }

  /**
   * A set grown by {@code add}, which the analysis does not record among its elements: its keys are
   * not all known, so it may hold {@code None}.
   */
  @Test
  public void testSetAdd() {
    ContainerElements elements = elements(SETS, "sink_added");
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /** {@code set()} allocates nothing, so no container reaches the sink. */
  @Test
  public void testEmptySetAdd() {
    assertNoContainer(SETS, "sink_empty_added");
  }

  /** {@code s |= {x}} makes the result of {@code |}, which allocates nothing. */
  @Test
  public void testSetInPlaceOr() {
    assertNoContainer(SETS, "sink_ior");
  }

  /**
   * An alias of a set {@code s |= {x}} grows in place: the left operand of {@code |}, its keys are
   * not all known.
   */
  @Test
  public void testSetInPlaceOrAlias() {
    ContainerElements elements = elements(SETS, "sink_ior_alias");
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /** A set grown by {@code update}. */
  @Test
  public void testSetUpdate() {
    ContainerElements elements = elements(SETS, "sink_updated");
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }
}
