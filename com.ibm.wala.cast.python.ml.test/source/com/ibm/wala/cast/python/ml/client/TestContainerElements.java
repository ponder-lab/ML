package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
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

  private static final String DELETES_KEY = "tf2_test_container_elements_del_key.py";

  private static final String DELETES_UNREACHED = "tf2_test_container_elements_del_unreached.py";

  private static final String UNMODELED_LIBRARY =
      "tf2_test_container_elements_unmodeled_library.py";

  private static final String UNREACHED_LIBRARY_OBJECT =
      "tf2_test_container_elements_unreached_library_object.py";

  private static final String DELETES_UNMODELED = "tf2_test_container_elements_del_unmodeled.py";

  private static final String UNREACHED_SUM = "tf2_test_container_elements_unreached_sum.py";

  private static final String UNREACHED_PROPERTY =
      "tf2_test_container_elements_unreached_property.py";

  private static final String UNREACHED_DESCRIPTOR =
      "tf2_test_container_elements_unreached_descriptor.py";

  private static final String UNREACHED_VARS = "tf2_test_container_elements_unreached_vars.py";

  private static final String UNREACHED_OBJECT_SETATTR =
      "tf2_test_container_elements_unreached_object_setattr.py";

  private static final String UNREACHED_NAMESPACE =
      "tf2_test_container_elements_unreached_namespace.py";

  private static final String UNREACHED_NAMESPACE_SUBCLASS =
      "tf2_test_container_elements_unreached_namespace_subclass.py";

  private static final String UNREACHED_DATACLASS =
      "tf2_test_container_elements_unreached_dataclass.py";

  private static final String TENSOR_LIKE = "tf2_test_container_elements_tensor_like.py";

  private static final String UNREACHED_REPEAT = "tf2_test_container_elements_unreached_repeat.py";

  private static final String UNREACHED_COPY = "tf2_test_container_elements_unreached_copy.py";

  private static final String UNREACHED_FIELD = "tf2_test_container_elements_unreached_field.py";

  private static final String UNREACHED_UNNAMED =
      "tf2_test_container_elements_unreached_unnamed.py";

  private static final String UNREACHED_AUGMENTED =
      "tf2_test_container_elements_unreached_augmented.py";

  private static final String UNREACHED_REFLECTED =
      "tf2_test_container_elements_unreached_reflected.py";

  private static final String UNREACHED_GETITEM =
      "tf2_test_container_elements_unreached_getitem.py";

  private static final String UNREACHED_STORED = "tf2_test_container_elements_unreached_stored.py";

  private static final String UNREACHED_HOOKED = "tf2_test_container_elements_unreached_hooked.py";

  private static final String UNREACHED_SETATTR =
      "tf2_test_container_elements_unreached_setattr.py";

  private static final String UNREACHED_CLASS_ATTRIBUTE =
      "tf2_test_container_elements_unreached_class_attribute.py";

  private static final String UNREACHED_DECORATED =
      "tf2_test_container_elements_unreached_decorated.py";

  private static final String UNREACHED_GETATTR =
      "tf2_test_container_elements_unreached_getattr.py";

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
   * The tensor types the analysis of the main fixture gives a container's positions.
   *
   * @param elements The container's elements.
   * @return Per position, its types.
   */
  private static List<Set<TensorType>> positionTypes(ContainerElements elements) {
    return ElementTypes.positionTypes(analysis(), elements);
  }

  /**
   * The tensor types the analysis of a fixture gives a container's elements.
   *
   * @param file The fixture.
   * @param elements The container's elements.
   * @return The union of the elements' types.
   */
  private static Set<TensorType> elementTypes(String file, ContainerElements elements) {
    return ElementTypes.elementTypes(analysis(file), elements);
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
    assertEquals(file + " " + sink, expected, elements(file, sink).getArity());
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
   * The lists and tuples that reach a sink.
   *
   * @param file The fixture.
   * @param sink The sink's name.
   * @return The lists and tuples in its parameter's points-to set.
   */
  private static Set<InstanceKey> receivedLists(String file, String sink) {
    PythonSSAPropagationCallGraphBuilder builder = builder(file);
    IClassHierarchy cha = builder.getClassHierarchy();
    IClass list = cha.lookupClass(PythonTypes.list);
    IClass tuple = cha.lookupClass(PythonTypes.tuple);
    boolean reached = false;
    Set<InstanceKey> ret = new HashSet<>();
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink)) {
        reached = true;
        for (InstanceKey value :
            builder
                .getPointerAnalysis()
                .getPointsToSet(
                    builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2)))
          if (cha.isSubclassOf(value.concreteType(), list)
              || cha.isSubclassOf(value.concreteType(), tuple)) ret.add(value);
      }
    assertTrue("Expecting " + sink + " to be reached.", reached);
    return ret;
  }

  /**
   * Asserts that a container reports no length, and so no positions, but still has element keys.
   *
   * @param sink The sink's name.
   */
  private static void assertNoArity(String sink) {
    ContainerElements elements = elements(sink);
    assertEquals(sink, OptionalInt.empty(), elements.getArity());
    assertTrue(sink, elements.getPositions().isEmpty());
    assertTrue(sink, positionTypes(elements).isEmpty());
    assertFalse(sink, elements.getAllElements().isEmpty());
  }

  /** A tuple literal of two tensors: two positions, each with its tensor's type. */
  @Test
  public void testTuple() {
    ContainerElements elements = elements("sink_tuple");
    assertEquals(OptionalInt.of(2), elements.getArity());
    assertEquals(2, elements.getPositions().size());
    assertEquals(List.of(Set.of(F32_2_3), Set.of(I32_4)), positionTypes(elements));
    assertFalse(elements.mayHoldNone());
  }

  /** A list literal of three tensors. */
  @Test
  public void testList() {
    ContainerElements elements = elements("sink_list");
    assertEquals(OptionalInt.of(3), elements.getArity());
    assertEquals(List.of(Set.of(F32_2_3), Set.of(I32_4), Set.of(F32_2_3)), positionTypes(elements));
  }

  /** A list only read through a method that changes neither its length nor its order. */
  @Test
  public void testListReadThroughCopy() {
    assertEquals(OptionalInt.of(2), elements("sink_counted").getArity());
  }

  /** A dict has no positions, but its values are elements. */
  @Test
  public void testDict() {
    ContainerElements elements = elements("sink_dict");
    assertEquals(OptionalInt.empty(), elements.getArity());
    assertEquals(2, elements.getAllElements().size());
  }

  /** A position holding a tuple has no tensor type of its own. */
  @Test
  public void testNested() {
    ContainerElements elements = elements("sink_nested");
    assertEquals(OptionalInt.of(2), elements.getArity());
    assertEquals(List.of(Set.of(F32_2_3), Set.of()), positionTypes(elements));
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

  /**
   * A slice read, {@code l[0:1]}, which leaves the list as it is, so its length stays known
   * (wala/ML#1250).
   */
  @Test
  public void testSliceRead() {
    assertArity(MUTATORS, "sink_slice_read", OptionalInt.of(2));
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
   * A program's list stored as an attribute of a TensorFlow object, {@code m.items}, and resized in
   * a function the call graph does not reach, reports no length: only TensorFlow's module itself is
   * taken to resize no list, not an object of the library (wala/ML#1182).
   */
  @Test
  public void testUnreachedLibraryObjectAttributeLosesArity() {
    assertArity(UNREACHED_LIBRARY_OBJECT, "sink_list", OptionalInt.empty());
  }

  /**
   * A list passed to a TensorFlow function no summary models, {@code tf.strings.join}, keeps its
   * length: TensorFlow is registered as a library whose functions resize no list passed to them,
   * which a summary would override (wala/ML#1182).
   */
  @Test
  public void testUnmodeledLibraryArgumentKeepsArity() {
    assertArity(UNMODELED_LIBRARY, "sink_parts", OptionalInt.of(2));
  }

  /**
   * A list a TensorFlow function reads, {@code tf.image.random_crop(t, size)}, keeps its length:
   * the call resolves to the function's summary, which changes nothing it is passed (wala/ML#1182).
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

  /** {@code del sl[1:]}, a slice. */
  @Test
  public void testSliceDelete() {
    assertArity(DELETES, "sink_slice_del", OptionalInt.empty());
  }

  /**
   * A list no {@code del} touches keeps its length beside lists one does, since the statement names
   * what it deletes from (wala/ML#1239).
   */
  @Test
  public void testUntouchedListBesideDelete() {
    assertArity(DELETES, "sink_untouched", OptionalInt.of(2));
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
   * A {@code del} of an attribute, {@code del o.a}, or by a string key, {@code del d["k"]}, in a
   * function the call graph reaches or one it does not, resizes no list: a list keeps its length
   * (wala/ML#1239).
   */
  @Test
  public void testKeyDeleteKeepsArity() {
    assertArity(DELETES_KEY, "sink_list", OptionalInt.of(2));
  }

  /**
   * A {@code del l[0]} of a parameter in a function the call graph does not reach, but the program
   * names, may delete from any list, so a list reports no length.
   */
  @Test
  public void testUnreachedParameterDelete() {
    assertArity(DELETES_UNREACHED, "sink_list", OptionalInt.empty());
  }

  /**
   * A {@code del} of an element of a list an unmodeled call returns, {@code del
   * random.choice([xs])[0]}, may delete from any list, so a list reports no length.
   */
  @Test
  public void testUnmodeledContainerDelete() {
    assertArity(DELETES_UNMODELED, "sink_list", OptionalInt.empty());
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
    assertTrue(positionTypes(elements).isEmpty());
    assertEquals(Set.of(F32_2_3, I32_4), elementTypes(FILE, elements));
  }

  /**
   * A comprehension of tensors has no length, but its elements' types: {@code tf.ones([i + 1])} is
   * a rank-1 float32 tensor whose extent the analysis does not fold.
   */
  @Test
  public void testElementTypesOfComprehension() {
    ContainerElements elements = elements("sink_comprehension");
    assertTrue(positionTypes(elements).isEmpty());
    assertEquals(
        Set.of(new TensorType("float32", List.of(UnresolvedDim.INSTANCE))),
        elementTypes(FILE, elements));
  }

  /** A list of ints has a length, but no element with tensor evidence. */
  @Test
  public void testElementTypesOfInts() {
    ContainerElements elements = elements("sink_ints");
    assertEquals(OptionalInt.of(3), elements.getArity());
    assertEquals(Set.of(), elementTypes(FILE, elements));
  }

  /** A tuple's element types are the union of its position types. */
  @Test
  public void testElementTypesOfTuple() {
    assertEquals(Set.of(F32_2_3, I32_4), elementTypes(FILE, elements("sink_tuple")));
  }

  /** A dict keyed by a float: its tensor value is an element. */
  @Test
  public void testFloatKeyElementTypes() {
    ContainerElements elements = elements(DICT_KEYS, "sink_float_tensor");
    assertTrue(elements.isComplete());
    assertEquals(Set.of(F32_2_3), elementTypes(DICT_KEYS, elements));
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
    assertEquals(Set.of(F32_2_3), elementTypes(DICT_KEYS, elements));
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

  /**
   * In a function the call graph does not reach: A sum of a list the body builds, {@code list(a) +
   * [1]}, is a new list, so a list keeps its length (wala/ML#1255).
   */
  @Test
  public void testUnreachedSum() {
    assertArity(UNREACHED_SUM, "sink_list", OptionalInt.of(2));
  }

  /**
   * In a function the call graph does not reach: A repetition of a list the body builds, {@code [1]
   * * n}, is a new list, so a list keeps its length (wala/ML#1255).
   */
  @Test
  public void testUnreachedRepeat() {
    assertArity(UNREACHED_REPEAT, "sink_list", OptionalInt.of(2));
  }

  /**
   * In a function the call graph does not reach: A slice copy, {@code a[:]}, is a new list where no
   * class defines {@code __getitem__}, so a list keeps its length (wala/ML#1255).
   */
  @Test
  public void testUnreachedCopy() {
    assertArity(UNREACHED_COPY, "sink_list", OptionalInt.of(2));
  }

  /**
   * In a function the call graph does not reach: An attribute the body stores a list it builds
   * under, {@code self.items}, holds no list the analysis has, beside a class whose {@code
   * __setattr__} stores into its items, so a list keeps its length (wala/ML#1255).
   */
  @Test
  public void testUnreachedField() {
    assertArity(UNREACHED_FIELD, "sink_list", OptionalInt.of(2));
  }

  /**
   * In a function the call graph does not reach: A parameter of a function nothing names holds no
   * list the analysis has, so a list keeps its length (wala/ML#1255).
   */
  @Test
  public void testUnreachedUnnamedFunction() {
    assertArity(UNREACHED_UNNAMED, "sink_list", OptionalInt.of(2));
  }

  /**
   * In a function the call graph does not reach: An augmented sum, {@code a += b}, is its
   * parameter, which may be any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedAugmented() {
    assertArity(UNREACHED_AUGMENTED, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: A sum whose right operand defines {@code __radd__}
   * may be any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedReflected() {
    assertArity(UNREACHED_REFLECTED, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: A slice of an object whose class defines {@code
   * __getitem__} may be any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedGetItem() {
    assertArity(UNREACHED_GETITEM, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute a body the call graph does not reach
   * stores a global list under holds it, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedStored() {
    assertArity(UNREACHED_STORED, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute of an object whose class defines
   * {@code __getattr__} may be any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedHooked() {
    assertArity(UNREACHED_HOOKED, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute, where the program stores one by a
   * computed name, may hold any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedSetAttr() {
    assertArity(UNREACHED_SETATTR, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute a class body stores a list under
   * holds it, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedClassAttribute() {
    assertArity(UNREACHED_CLASS_ATTRIBUTE, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: A parameter of a function a decorator names may
   * hold any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedDecorated() {
    assertArity(UNREACHED_DECORATED, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: A parameter of a function a string names may hold
   * any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedGetAttr() {
    assertArity(UNREACHED_GETATTR, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute a property returns a global list for
   * holds it, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedProperty() {
    assertArity(UNREACHED_PROPERTY, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute a descriptor's {@code __get__}
   * returns a global list for holds it, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedDescriptor() {
    assertArity(UNREACHED_DESCRIPTOR, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute stored through {@code vars} may hold
   * any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedVars() {
    assertArity(UNREACHED_VARS, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute stored through {@code
   * object.__setattr__} may hold any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedObjectSetAttr() {
    assertArity(UNREACHED_OBJECT_SETATTR, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute of an object whose class has a base
   * no summary models, {@code types.SimpleNamespace}, whose initializer stores the list where the
   * analysis sees no store, may be any list, so no list reports a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedNamespace() {
    assertArity(UNREACHED_NAMESPACE, "sink_list", OptionalInt.empty());
  }

  /**
   * In a function the call graph does not reach: An attribute of an object whose class has no base
   * but a subclass of which adds {@code types.SimpleNamespace} may be any list, so no list reports
   * a length (wala/ML#1255).
   */
  @Test
  public void testUnreachedNamespaceSubclass() {
    assertArity(UNREACHED_NAMESPACE_SUBCLASS, "sink_list", OptionalInt.empty());
  }

  /**
   * A list times an array is the array's multiplication, which makes no list (wala/ML#960,
   * wala/ML#1259).
   *
   * <p>TODO: wala/ML#960; the list literal reaches the list operation before the array does, and
   * the repetition its rule allows against the operand's empty points-to set is not withdrawn, so
   * the sink receives the list the operation makes.
   */
  @Test(expected = AssertionError.class)
  public void testArrayTimesList() {
    assertEquals(Set.of(), receivedLists(TENSOR_LIKE, "sink_array_times_list"));
  }

  /**
   * A list times a tensor is the tensor's multiplication, which makes no list (wala/ML#960,
   * wala/ML#1259).
   *
   * <p>TODO: wala/ML#960; the list literal reaches the list operation before the tensor does, and
   * the repetition its rule allows against the operand's empty points-to set is not withdrawn, so
   * the sink receives the list the operation makes.
   */
  @Test(expected = AssertionError.class)
  public void testTensorTimesList() {
    assertEquals(Set.of(), receivedLists(TENSOR_LIKE, "sink_tensor_times_list"));
  }

  /**
   * A list times an integer is a repetition, which makes a list (wala/ML#960): the control of
   * {@link #testArrayTimesList()} and {@link #testTensorTimesList()}.
   */
  @Test
  public void testListTimesInt() {
    assertFalse(receivedLists(TENSOR_LIKE, "sink_list_times_int").isEmpty());
  }

  /**
   * In a function the call graph does not reach: An attribute a dataclass's generated initializer
   * stores is a list the analysis does not see stored, since the front end drops a class's
   * decorators, so the list reads as its literal's length though the run grows it.
   *
   * <p>TODO: wala/ML#1260; once the front end translates a class's decorators, no list reports a
   * length, {@code OptionalInt.empty()}.
   */
  @Test
  public void testUnreachedDataclass() {
    assertArity(UNREACHED_DATACLASS, "sink_list", OptionalInt.of(2));
  }
}
