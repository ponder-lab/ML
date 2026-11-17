package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
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
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Tests what a points-to client reads for the {@code element_spec} of a {@code tf.data.Dataset.zip}
 * that has an argument whose element is not mirrored, a {@code map} over a file's lines, on {@code
 * tf2_test_zip_element_spec.py} (<a
 * href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>): each sink receives a spec, a
 * structure of specs or a shape read off one, and the test reads the types of the allocations in
 * the points-to set of the sink's parameter. The place of such an argument in the specs holds the
 * description of that argument's own element, where it held nothing, so the specs have the nesting
 * of the datasets the zip was given, and the shapes a program maps out of them are tensor shapes.
 * {@link com.ibm.wala.cast.python.ml.test.tensorflow.v2.TestZipElementSpec} pins the tensors made
 * from them.
 */
public class TestZipSpecProducers extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_zip_element_spec.py";

  private static final Set<String> SPEC = names(TensorFlowTypes.TENSOR_SPEC);

  private static final Set<String> STATIC_SHAPE = names(TensorFlowTypes.STATIC_TENSOR_SHAPE);

  private static final Set<String> TUPLE = names(PythonTypes.tuple);

  private static final Set<String> DICT = names(PythonTypes.dict);

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
            new TestZipSpecProducers().makeEngine(Collections.<File>emptyList(), FILE);
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
   * The number of allocations a sink receives.
   *
   * @param sink The sink's name.
   * @return The size of the points-to set of its parameter.
   */
  private static int count(String sink) {
    return builder().getPointerAnalysis().getPointsToSet(parameter(sink)).size();
  }

  /**
   * The types of the allocations a sink receives.
   *
   * @param sink The sink's name.
   * @return The names of the types in the points-to set of its parameter, sorted.
   */
  private static Set<String> types(String sink) {
    Set<String> ret = new TreeSet<>();
    for (InstanceKey ik : builder().getPointerAnalysis().getPointsToSet(parameter(sink)))
      ret.add(ik.concreteType().getName().toString());
    return ret;
  }

  /**
   * The specs of a zip of two functions' results, each a dict: a tuple, whose member is a dict of
   * specs, whose value is a spec with a tensor shape.
   */
  @Test
  public void testSpecs() {
    assertEquals(TUPLE, types("consume_spec"));
    assertEquals(DICT, types("consume_member_spec"));
    assertEquals(SPEC, types("consume_leaf_spec"));
    assertEquals(STATIC_SHAPE, types("consume_leaf_shape"));
  }

  /**
   * The shapes a program maps out of those specs with {@code tf.nest.map_structure} have the specs'
   * nesting, with a tensor shape in each spec's place.
   */
  @Test
  public void testMappedShapes() {
    assertEquals(TUPLE, types("consume_shapes"));
    assertEquals(DICT, types("consume_shapes_member"));
    assertEquals(STATIC_SHAPE, types("consume_shape_leaf"));
  }

  /** The same in a function a lambda handed to {@code apply} passes its dataset to. */
  @Test
  public void testMappedShapesOfAppliedDataset() {
    assertEquals(DICT, types("consume_applied_shapes_member"));
    assertEquals(STATIC_SHAPE, types("consume_applied_shape_leaf"));
  }

  /** A zip of two functions' results that are each one tensor: a member is a spec. */
  @Test
  public void testTensorResults() {
    assertEquals(SPEC, types("consume_tensor_spec"));
    assertEquals(STATIC_SHAPE, types("consume_tensor_shape"));
  }

  /**
   * A zip of a dataset whose element is mirrored, a function's results and a file's lines: each
   * member's place holds that dataset's own specs.
   */
  @Test
  public void testMixed() {
    assertEquals(DICT, types("consume_mixed_mirrored_spec"));
    assertEquals(SPEC, types("consume_mixed_mirrored_leaf_spec"));
    assertEquals(DICT, types("consume_mixed_mapped_spec"));
    assertEquals(SPEC, types("consume_mixed_root_spec"));
    assertEquals(STATIC_SHAPE, types("consume_mixed_shape_leaf"));
  }

  /** A zip in a zip, and a zip of a dict of datasets, whose specs are a dict. */
  @Test
  public void testNesting() {
    assertEquals(SPEC, types("consume_nested_leaf_spec"));
    assertEquals(DICT, types("consume_keyed_spec"));
    assertEquals(SPEC, types("consume_keyed_leaf_spec"));
  }

  /** A {@code shuffle} of a zip hands the specs on, and a batch of one has a shape to read. */
  @Test
  public void testDatasetsMadeFromAZip() {
    assertEquals(SPEC, types("consume_shuffled_leaf_spec"));
    assertEquals(STATIC_SHAPE, types("consume_batched_leaf_shape"));
  }

  /**
   * Controls: a zip that has no argument with a function's results is described once, as before,
   * whether its arguments' elements are mirrored or a file's lines.
   */
  @Test
  public void testZipWithoutMappedArgumentIsDescribedOnce() {
    assertEquals(TUPLE, types("consume_roots_spec"));
    assertEquals(1, count("consume_roots_spec"));
    assertEquals(SPEC, types("consume_roots_member_spec"));
    assertEquals(1, count("consume_roots_member_spec"));
    assertEquals(TUPLE, types("consume_mirrored_spec"));
    assertEquals(1, count("consume_mirrored_spec"));
    assertEquals(SPEC, types("consume_mirrored_leaf_spec"));
    assertEquals(1, count("consume_mirrored_leaf_spec"));
  }

  /**
   * A zip that has an argument with a function's results is described from each of the attribute's
   * two sources, so its specs are two tuples, and so are two the specs of a member that either
   * source describes alone in another zip: one whose element is mirrored, and a file's lines. Each
   * pair reads one type, which {@link
   * com.ibm.wala.cast.python.ml.test.tensorflow.v2.TestZipElementSpec#testMixed} pins.
   */
  @Test
  public void testZipWithMappedArgumentIsDescribedTwice() {
    assertEquals(2, count("consume_spec"));
    assertEquals(2, count("consume_shapes"));
    assertEquals(2, count("consume_keyed_spec"));
    assertEquals(2, count("consume_mixed_mirrored_spec"));
    assertEquals(2, count("consume_mixed_mirrored_leaf_spec"));
    assertEquals(2, count("consume_mixed_root_spec"));
  }

  /** A structure projection with no path is refused where it is named. */
  @Test(expected = IllegalArgumentException.class)
  public void testStructureProjectionWithoutPath() {
    builder().setStructureProjection(TensorFlowTypes.DATASET_ZIP_STRUCTURE, List.of());
  }
}
