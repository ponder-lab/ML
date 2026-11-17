package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.ValueElements;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.Util;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Tests that a tensor shape is one to a points-to client however a program reaches it, on {@code
 * tf2_test_shape_producers.py} (<a href="https://github.com/wala/ML/issues/1280">wala/ML#1280</a>):
 * each sink receives a shape, and the test reads the types of the allocations in the points-to set
 * of the sink's parameter. A read of a tensor's {@code shape} attribute holds an allocation of
 * {@link TensorFlowTypes#STATIC_TENSOR_SHAPE}, and so does a call of its {@code get_shape} method
 * and the {@code shape} of a {@code tf.TensorSpec} a dataset's {@code element_spec} holds, read
 * directly or through a {@code tf.nest.map_structure} over the specs. The shapes read through
 * {@code tf.compat.v1.data.get_output_shapes} are not modeled yet, and their sinks pin an empty
 * points-to set.
 */
public class TestShapeProducers extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_shape_producers.py";

  private static final Set<String> STATIC_SHAPE = names(TensorFlowTypes.STATIC_TENSOR_SHAPE);

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
            new TestShapeProducers().makeEngine(Collections.<File>emptyList(), FILE);
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
    for (InstanceKey ik : builder().getPointerAnalysis().getPointsToSet(parameter(sink))) {
      TypeName name = ik.concreteType().getName();
      ret.add(name.toString());
    }
    return ret;
  }

  /** A control: the {@code shape} attribute of a tensor is a static shape. */
  @Test
  public void testAttribute() {
    assertEquals(STATIC_SHAPE, types("consume_attribute"));
  }

  /** A control: the {@code shape} attribute of an element a program iterates is a static shape. */
  @Test
  public void testElementAttribute() {
    assertEquals(STATIC_SHAPE, types("consume_element_shape"));
  }

  /** A call of a tensor's {@code get_shape} returns a static shape, as the attribute reads. */
  @Test
  public void testMethod() {
    assertEquals(STATIC_SHAPE, types("consume_method"));
  }

  /** A call of a variable's {@code get_shape} returns a static shape. */
  @Test
  public void testVariableMethod() {
    assertEquals(STATIC_SHAPE, types("consume_variable_method"));
  }

  /** The shape {@code get_shape} returns has an {@code as_list}, whose list holds Python ints. */
  @Test
  public void testMethodList() {
    assertEquals(names(PythonTypes.list), types("consume_method_list"));
    boolean holdsInt = false;
    for (PointerKey element :
        ValueElements.of(builder(), parameter("consume_method_list")).getAllElements())
      for (InstanceKey ik : builder().getPointerAnalysis().getPointsToSet(element))
        holdsInt |= Util.isPythonScalarKey(ik);
    assertTrue("The list holds no Python int.", holdsInt);
  }

  /** A subscript of the shape {@code get_shape} returns is a Python int, as the attribute's is. */
  @Test
  public void testMethodDimension() {
    assertEquals(names(PythonTypes.PYTHON_INT), types("use_dimension_attribute"));
    assertEquals(names(PythonTypes.PYTHON_INT), types("use_dimension_method"));
  }

  /**
   * A dataset's {@code element_spec} is a {@code tf.TensorSpec} for an element that is a tensor,
   * and the element's structure for one that is a tuple, whose member is a spec.
   */
  @Test
  public void testSpec() {
    assertEquals(names(TensorFlowTypes.TENSOR_SPEC), types("consume_spec"));
    assertEquals(names(PythonTypes.tuple), types("consume_tuple_spec"));
    assertEquals(names(TensorFlowTypes.TENSOR_SPEC), types("consume_tuple_spec_member"));
  }

  /**
   * The {@code shape} of a dataset's {@code element_spec}, for an element that is a tensor, a
   * tuple's member, a dict's value, a dict's value in a tuple and a namedtuple's field, for a
   * dataset that is a function's parameter, and for the dataset a {@code map} returns.
   */
  @Test
  public void testSpecShape() {
    assertEquals(STATIC_SHAPE, types("consume_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_tuple_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_dict_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_nested_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_named_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_parameter_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_map_spec_shape"));
  }

  /**
   * The batch size read from the {@code shape} of a batched dataset's {@code element_spec} is the
   * Python int the batch has, where at run time it is {@code None}, since the last batch may be
   * smaller.
   */
  // TODO: Expect no int once the spec of a batch that may be partial has no size for the batch axis
  // (wala/ML#1280).
  @Test
  public void testSpecBatchSize() {
    assertEquals(names(PythonTypes.PYTHON_INT), types("consume_spec_batch_size"));
  }

  /**
   * The {@code element_spec} of a dataset whose element is not mirrored, a dataset of a file's
   * lines, is a spec, and its {@code shape} a static shape, also for the dataset a {@code map} over
   * the lines returns. Such a dataset carries its element's structure on its method objects, which
   * is what the read describes.
   */
  @Test
  public void testSpecOfUnmirroredElement() {
    assertEquals(names(TensorFlowTypes.TENSOR_SPEC), types("consume_text_spec"));
    assertEquals(STATIC_SHAPE, types("consume_text_spec_shape"));
    assertEquals(STATIC_SHAPE, types("consume_text_map_spec_shape"));
  }

  /**
   * The {@code element_spec} of a {@code map} over a file's lines whose function returns a dict is
   * a dict, and its member a spec.
   */
  @Test
  public void testSpecOfUnmirroredStructure() {
    assertEquals(names(PythonTypes.dict), types("consume_text_dict_spec"));
    assertEquals(names(TensorFlowTypes.TENSOR_SPEC), types("consume_text_dict_spec_member"));
  }

  /**
   * The shapes a {@code tf.nest.map_structure} maps out of the specs of a dataset whose element is
   * not mirrored: a static shape for an element that is one tensor, and for a dict of specs a dict
   * whose member is the shape.
   */
  @Test
  public void testMappedShapesOfUnmirroredElement() {
    assertEquals(STATIC_SHAPE, types("consume_text_mapped"));
    assertEquals(names(PythonTypes.dict), types("consume_text_dict_mapped"));
    assertEquals(STATIC_SHAPE, types("consume_text_dict_mapped_member"));
  }

  /**
   * A dataset whose element is mirrored is described once, from the mirror, also when a {@code map}
   * returned it. The structure a dataset's method objects carry is read only for a dataset that
   * holds the marker of an element that is not mirrored.
   */
  @Test
  public void testSpecOfMirroredElementIsDescribedOnce() {
    assertEquals(1, count("consume_spec"));
    assertEquals(1, count("consume_map_spec"));
  }

  /** The {@code element_spec} of an {@code enumerate} of a dataset of a file's lines. */
  // TODO: Expect a `tuple` once the spec of an `enumerate` of a dataset whose element is not
  // mirrored is described (wala/ML#1280).
  @Test
  public void testSpecOfEnumeratedUnmirroredElement() {
    assertEquals(Set.of(), types("consume_text_enumerate_spec"));
  }

  /**
   * The {@code shape} of a {@code tf.TensorSpec} a program constructs is a static shape, beside the
   * list the program passed, which the spec's summary stores as its {@code shape}.
   */
  // TODO: Expect `STATIC_SHAPE` alone once a constructed spec no longer stores the list it was
  // passed (wala/ML#1280).
  @Test
  public void testConstructedSpecShape() {
    assertEquals(
        names(PythonTypes.list, TensorFlowTypes.STATIC_TENSOR_SHAPE),
        types("consume_constructed_spec_shape"));
  }

  /** What {@code tf.compat.v1.data.get_output_shapes} returns, and a member of it for a tuple. */
  // TODO: Expect a tensor shape type for each once `get_output_shapes` is modeled (wala/ML#1280).
  @Test
  public void testOutputShapes() {
    assertEquals(Set.of(), types("consume_output_shapes"));
    assertEquals(Set.of(), types("consume_tuple_output_shape"));
  }

  /**
   * What {@code tf.nest.map_structure} returns for a function reading each spec's {@code shape}:
   * the shape for a dataset of tensors, the same for a dataset that is a parameter, and for a tuple
   * of specs a tuple whose member is the shape.
   */
  @Test
  public void testMappedShapes() {
    assertEquals(STATIC_SHAPE, types("consume_mapped"));
    assertEquals(STATIC_SHAPE, types("consume_parameter_mapped"));
    assertEquals(names(PythonTypes.tuple), types("consume_mapped_structure"));
    assertEquals(STATIC_SHAPE, types("consume_mapped_member"));
  }
}
