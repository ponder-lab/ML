package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder.DescriptorKey;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.types.TypeReference;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Tests what a points-to client reads for the {@code element_spec} of a {@code map} over a file's
 * lines that several functions reach, on {@code tf2_test_element_spec_places.py} (<a
 * href="https://github.com/wala/ML/issues/1300">wala/ML#1300</a>): each sink receives a spec, a
 * structure of specs or a shape read off one, and the test reads the allocations in the points-to
 * set of the sink's parameter. A read describes the dataset's element once per place and class,
 * whatever number of functions return a value there, where it described every allocation of every
 * function's value. {@link com.ibm.wala.cast.python.ml.test.tensorflow.v2.TestElementSpecPlaces}
 * pins the tensors made from the specs.
 */
public class TestSpecPlaceProducers extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_element_spec_places.py";

  private static final Set<String> SPEC = names(TensorFlowTypes.TENSOR_SPEC);

  private static final Set<String> STATIC_SHAPE = names(TensorFlowTypes.STATIC_TENSOR_SHAPE);

  private static final Set<String> DICT = names(PythonTypes.dict);

  private static final Set<String> NAMED_TUPLE = Set.of("Lcollections/namedtuple/instance");

  private static PythonSSAPropagationCallGraphBuilder builder;

  /**
   * Analyzes the fixture, once.
   *
   * @return Its builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestSpecPlaceProducers().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        engine.performAnalysis(b);
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * The names of the given types.
   *
   * @param types The types.
   * @return Their names, sorted.
   */
  private static Set<String> names(TypeReference... types) {
    Set<String> ret = new TreeSet<>();
    for (TypeReference type : types) ret.add(type.getName().toString());
    return ret;
  }

  /**
   * The parameter of a sink, which has one node.
   *
   * @param sink The sink's name.
   * @return The pointer key of its parameter.
   */
  private static PointerKey parameter(String sink) {
    List<PointerKey> parameters = new ArrayList<>();
    for (CGNode node : builder().getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
        parameters.add(
            builder().getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for " + sink + ".", 1, parameters.size());
    return parameters.get(0);
  }

  /**
   * The allocations a sink receives.
   *
   * @param sink The sink's name.
   * @return The points-to set of its parameter.
   */
  private static List<InstanceKey> keys(String sink) {
    List<InstanceKey> ret = new ArrayList<>();
    for (InstanceKey ik : builder().getPointerAnalysis().getPointsToSet(parameter(sink)))
      ret.add(ik);
    return ret;
  }

  /**
   * The types of the allocations a sink receives.
   *
   * @param sink The sink's name.
   * @return The names of the types in the points-to set of its parameter, sorted.
   */
  private static Set<String> types(String sink) {
    Set<String> ret = new TreeSet<>();
    for (InstanceKey ik : keys(sink)) ret.add(ik.concreteType().getName().toString());
    return ret;
  }

  /**
   * The one description a sink receives.
   *
   * @param sink The sink's name.
   * @return The description in the points-to set of its parameter.
   */
  private static DescriptorKey description(String sink) {
    List<InstanceKey> keys = keys(sink);
    assertEquals("Expecting one description at " + sink + ".", 1, keys.size());
    assertTrue(keys.get(0) instanceof DescriptorKey);
    return (DescriptorKey) keys.get(0);
  }

  /**
   * Three functions that return dicts with the same keys reach one {@code map} call: the read has
   * one description of the dict and one of each value.
   */
  @Test
  public void testFunctionsShareADescription() {
    assertEquals(DICT, types("consume_several_spec"));
    assertEquals(1, keys("consume_several_spec").size());
    assertEquals(SPEC, types("consume_several_leaf_spec"));
    assertEquals(1, keys("consume_several_leaf_spec").size());
  }

  /**
   * The shapes a program maps out of those specs with {@code tf.nest.map_structure} are one dict,
   * with a tensor shape in a spec's place.
   */
  @Test
  public void testMappedShapes() {
    assertEquals(DICT, types("consume_several_shapes"));
    assertEquals(1, keys("consume_several_shapes").size());
    assertEquals(STATIC_SHAPE, types("consume_several_leaf_shape"));
  }

  /**
   * A description from a dataset records no key it copies. It records the dataset and where it is
   * in the dataset's element: the element itself, a value of it, and a value three levels down.
   */
  @Test
  public void testADescriptionRecordsItsPlace() {
    DescriptorKey whole = description("consume_several_spec");
    assertNull(whole.getDescribed());
    assertNotNull(whole.getReceiver());
    assertEquals(0, whole.getPath().size());
    DescriptorKey value = description("consume_several_leaf_spec");
    assertNull(value.getDescribed());
    assertEquals(whole.getReceiver(), value.getReceiver());
    assertEquals(1, value.getPath().size());
    assertEquals(3, description("consume_nested_leaf_spec").getPath().size());
  }

  /**
   * Two functions whose dicts hold a tensor and a tuple under one key: the dicts share a
   * description, and the key's place has a description of each class, a spec and a tuple.
   */
  @Test
  public void testClassesAtOnePlaceStayApart() {
    assertEquals(1, keys("consume_varied_spec").size());
    assertEquals(
        names(TensorFlowTypes.TENSOR_SPEC, PythonTypes.tuple), types("consume_varied_member_spec"));
    assertEquals(2, keys("consume_varied_member_spec").size());
  }

  /**
   * Two functions that return a namedtuple instance: one description of the instance, whose field
   * is read by its name, also in the shapes mapped out of the specs.
   */
  @Test
  public void testNamedTuple() {
    assertEquals(NAMED_TUPLE, types("consume_paired_spec"));
    assertEquals(1, keys("consume_paired_spec").size());
    assertEquals(SPEC, types("consume_paired_left_spec"));
    assertEquals(1, keys("consume_paired_left_spec").size());
    assertEquals(STATIC_SHAPE, types("consume_paired_left_shape"));
  }

  /**
   * Nine levels of dicts, where a place is recorded down to eight: the innermost dict has one
   * description, at its place, and its value's descriptions have no place and record the keys they
   * copy, one for each.
   */
  @Test
  public void testPastTheRecordedPlaces() {
    DescriptorKey inner = description("consume_deep_inner_spec");
    assertNull(inner.getDescribed());
    assertEquals(8, inner.getPath().size());
    List<InstanceKey> leaves = keys("consume_deep_leaf_spec");
    assertTrue(leaves.size() > 1);
    Set<InstanceKey> copied = new HashSet<>();
    for (InstanceKey leaf : leaves) {
      DescriptorKey description = (DescriptorKey) leaf;
      assertNull(description.getPath());
      assertNotNull(description.getDescribed());
      copied.add(description.getDescribed());
    }
    assertEquals(leaves.size(), copied.size());
  }

  /** One function: one description of the dict and of its value. */
  @Test
  public void testOneFunction() {
    assertEquals(DICT, types("consume_alone_spec"));
    assertEquals(1, keys("consume_alone_spec").size());
    assertEquals(1, keys("consume_alone_leaf_spec").size());
  }

  /** Two datasets read at one place in the program: each has a description of its own. */
  @Test
  public void testDatasetsStayApart() {
    List<InstanceKey> keys = keys("consume_either_spec");
    assertEquals(2, keys.size());
    Set<InstanceKey> receivers = new HashSet<>();
    for (InstanceKey key : keys) {
      DescriptorKey description = (DescriptorKey) key;
      assertNull(description.getDescribed());
      receivers.add(description.getReceiver());
    }
    assertEquals(2, receivers.size());
  }
}
