package com.ibm.wala.cast.python.ml.test;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.analysis.TensorTypeAnalysis;
import com.ibm.wala.cast.python.ml.client.ContainerElements;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;
import org.junit.Test;

/**
 * Tests {@link ContainerElements#ofValue} on {@code tf2_test_container_elements_of_value.py}, where
 * each sink receives one value: the {@code inputs} a Keras step hands to {@code call} when {@code
 * fit} is given a dataset, which no container instance stands for, and an ordinary local tuple.
 */
public class TestContainerElementsOfValue extends TestPythonMLCallGraphShape {

  private static final String FILE = "tf2_test_container_elements_of_value.py";

  private static PythonSSAPropagationCallGraphBuilder builder;

  private static TensorTypeAnalysis analysis;

  /**
   * Analyzes the fixture, once.
   *
   * @return Its builder.
   */
  private static synchronized PythonSSAPropagationCallGraphBuilder builder() {
    if (builder == null)
      try {
        PythonTensorAnalysisEngine engine =
            new TestContainerElementsOfValue().makeEngine(Collections.<File>emptyList(), FILE);
        PythonSSAPropagationCallGraphBuilder b = engine.defaultCallGraphBuilder();
        b.makeCallGraph(b.getOptions());
        analysis = engine.performAnalysis(b);
        builder = b;
      } catch (Exception e) {
        throw new IllegalStateException(e);
      }
    return builder;
  }

  /**
   * A type whose first axis is the batch axis a signature leaves unspecified.
   *
   * @param cellType The dtype's name.
   * @param size The second axis's size.
   * @return The type.
   */
  private static TensorType batched(String cellType, int size) {
    return new TensorType(cellType, List.of(DynamicDim.INSTANCE, new NumericDim(size)));
  }

  /**
   * The parameter key of the one node of a sink.
   *
   * @param sink The sink's name.
   * @return Its parameter's key.
   */
  private static PointerKey parameter(String sink) {
    List<PointerKey> ret = new ArrayList<>();
    for (CGNode node : builder().getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/" + sink))
        ret.add(builder().getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2));
    assertEquals("Expecting one node for " + sink + ".", 1, ret.size());
    return ret.get(0);
  }

  /**
   * The elements of the value a sink receives.
   *
   * @param sink The sink's name.
   * @return Its parameter's elements.
   */
  private static ContainerElements ofValue(String sink) {
    return ContainerElements.ofValue(builder(), parameter(sink));
  }

  /**
   * {@code fit(from_tensor_slices(((x, mask), y)).batch(2))} hands {@code call} the tuple {@code
   * (x, mask)}: two positions, the batched {@code x} and {@code mask}.
   */
  @Test
  public void testPair() {
    ContainerElements elements = ofValue("consume_pair");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertTrue(elements.isComplete());
    assertTrue(elements.positions().isEmpty());
    assertEquals(
        List.of(Set.of(TensorType.of(FLOAT32, 2, 4)), Set.of(TensorType.of(FLOAT32, 2, 3))),
        elements.positionTypes(analysis));
    assertEquals(
        Set.of(TensorType.of(FLOAT32, 2, 4), TensorType.of(FLOAT32, 2, 3)),
        elements.elementTypes(analysis));
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A dict component has no declared order, so it has no length, and it is no tensor, so its types
   * are not all of it.
   */
  @Test
  public void testKeyed() {
    ContainerElements elements = ofValue("consume_keyed");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /** A {@code from_generator} component its {@code output_signature} declares as three specs. */
  @Test
  public void testTriple() {
    ContainerElements elements = ofValue("consume_triple");
    assertEquals(OptionalInt.of(3), elements.arity());
    assertTrue(elements.isComplete());
    assertEquals(3, elements.positionTypes(analysis).size());
    assertEquals(
        List.of(
            Set.of(batched("float32", 4)),
            Set.of(batched("float32", 3)),
            Set.of(batched("int32", 5))),
        elements.positionTypes(analysis));
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A {@code from_generator} component its {@code output_types} declares as two dtypes: two
   * positions.
   */
  @Test
  public void testPairFromGenerator() {
    ContainerElements elements = ofValue("consume_pair_from_generator");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(Set.of(batched("float32", 4)), Set.of(batched("float32", 3))),
        elements.positionTypes(analysis));
  }

  /**
   * A zipped dataset's whole element, a pair {@code from_generator}'s {@code output_signature}
   * declares: two positions.
   */
  @Test
  public void testZippedPair() {
    ContainerElements elements = ofValue("consume_zipped_pair");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(Set.of(batched("float32", 4)), Set.of(batched("float32", 3))),
        elements.positionTypes(analysis));
  }

  /**
   * A zipped dataset's whole element, the pair {@code from_tensors} is given: two positions, which
   * {@code fit}'s data reads as wholly unknown, so they may be anything.
   */
  @Test
  public void testZippedTensors() {
    ContainerElements elements = ofValue("consume_zipped_tensors");
    assertEquals(OptionalInt.of(2), elements.arity());
    TensorType top = new TensorType("unknown", null);
    assertEquals(List.of(Set.of(top), Set.of(top)), elements.positionTypes(analysis));
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /**
   * A component {@code ((a, b), m)}: two positions, the first itself a tuple, so its types are not
   * all of it.
   */
  @Test
  public void testNested() {
    ContainerElements elements = ofValue("consume_nested");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        "Positions: " + elements.positionTypes(analysis) + ".",
        Set.of(TensorType.of(FLOAT32, 2, 5)),
        elements.positionTypes(analysis).get(1));
    assertFalse(elements.isComplete());
  }

  /** A component {@code (*xs, c)}, whose length the display does not show. */
  @Test
  public void testStarred() {
    ContainerElements elements = ofValue("consume_starred");
    assertEquals(
        "Positions: " + elements.positionTypes(analysis) + ".",
        OptionalInt.empty(),
        elements.arity());
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /** A component of a {@code flat_map}'s datasets, which no declared structure reaches. */
  @Test
  public void testFlatMapped() {
    ContainerElements elements = ofValue("consume_flat_mapped");
    assertEquals(
        "Types: " + elements.elementTypes(analysis) + ".", OptionalInt.empty(), elements.arity());
    assertFalse("Types: " + elements.elementTypes(analysis) + ".", elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /** A component {@code tuple(parts)} builds is a tuple the program does not write as a display. */
  @Test
  public void testBuilt() {
    ContainerElements elements = ofValue("consume_built");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
  }

  /**
   * In one frame, the dataset is either a structured one or a flat one, so {@code inputs} is a
   * tuple or a tensor: it has no length, and its types include the flat tensor's.
   */
  @Test
  public void testMixed() {
    ContainerElements elements = ofValue("consume_mixed");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertTrue(
        "Types: " + elements.elementTypes(analysis) + ".",
        elements.elementTypes(analysis).contains(TensorType.of(FLOAT32, 2, 6)));
    assertFalse("Types: " + elements.elementTypes(analysis) + ".", elements.isComplete());
  }

  /**
   * As {@link #testMixed}, with the batching applied to the chosen dataset: no length, and the flat
   * tensor's type.
   */
  @Test
  public void testMixedBatched() {
    ContainerElements elements = ofValue("consume_mixed_batched");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertTrue(
        "Types: " + elements.elementTypes(analysis) + ".",
        elements.elementTypes(analysis).contains(TensorType.of(FLOAT32, 2, 6)));
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /**
   * A shuffled and batched dataset that is either a pair dataset or a triple one: the lengths
   * disagree, so there is none.
   */
  @Test
  public void testLengthsShuffled() {
    ContainerElements elements = ofValue("consume_lengths_shuffled");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
  }

  /**
   * A shuffled and batched dataset that is either a flat dataset or a pair one: no length, and its
   * types are not all of it.
   */
  @Test
  public void testMixedShuffled() {
    ContainerElements elements = ofValue("consume_mixed_shuffled");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(
        "Types: " + elements.elementTypes(analysis) + ".",
        elements.elementTypes(analysis).contains(TensorType.of(FLOAT32, 2, 6)));
    assertFalse(elements.isComplete());
    assertTrue(elements.mayHoldNone());
  }

  /**
   * A batched dataset that is either of two pair datasets: they agree on the length, so it has that
   * length, and each position holds the types of both.
   */
  @Test
  public void testAgreeing() {
    ContainerElements elements = ofValue("consume_agreeing");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(
            Set.of(TensorType.of(FLOAT32, 2, 4)),
            Set.of(TensorType.of(FLOAT32, 2, 3), TensorType.of(FLOAT32, 2, 5))),
        elements.positionTypes(analysis));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A {@code map} that keeps the element's pair, {@code lambda inp, y: inp}: {@code fit} takes the
   * pair's first member for the inputs, a tensor.
   */
  @Test
  public void testMappedInputs() {
    ContainerElements elements = ofValue("consume_mapped_inputs");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertEquals(Set.of(TensorType.of(FLOAT32, 2, 4)), elements.elementTypes(analysis));
    assertTrue(elements.isComplete());
  }

  /**
   * A loop over a batched dataset that is either a pair dataset or a triple one: the batch's
   * receiver may be either, whose lengths disagree, so there is none.
   */
  @Test
  public void testLoopLengths() {
    ContainerElements elements = ofValue("consume_loop_lengths");
    assertEquals(OptionalInt.empty(), elements.arity());
  }

  /**
   * A loop over a batched dataset that is either of two pair datasets: the batch's receivers agree
   * on the length.
   */
  @Test
  public void testLoopAgreeing() {
    ContainerElements elements = ofValue("consume_loop_agreeing");
    assertEquals(
        "Types: " + elements.positionTypes(analysis) + ".", OptionalInt.of(2), elements.arity());
  }

  /**
   * A {@code map} that keeps the element's {@code ((x, mask), w)}, {@code lambda inp, y: inp}:
   * {@code fit} takes {@code (x, mask)} for the inputs, which the receiver's element declares
   * within the returned component.
   */
  @Test
  public void testMappedPair() {
    ContainerElements elements = ofValue("consume_mapped_pair");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(Set.of(TensorType.of(FLOAT32, 2, 4)), Set.of(TensorType.of(FLOAT32, 2, 3))),
        elements.positionTypes(analysis));
    assertTrue(elements.isComplete());
  }

  /** A flat dataset input is a tensor: no length, and its own type. */
  @Test
  public void testFlat() {
    ContainerElements elements = ofValue("consume_flat");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertEquals(Set.of(TensorType.of(FLOAT32, 2, 4)), elements.elementTypes(analysis));
    assertTrue(elements.isComplete());
  }

  /** An ordinary local tuple reads as its one instance does. */
  @Test
  public void testLocal() {
    PointerKey parameter = parameter("consume_local");
    List<InstanceKey> instances = new ArrayList<>();
    builder().getPointerAnalysis().getPointsToSet(parameter).forEach(instances::add);
    assertEquals(1, instances.size());
    ContainerElements of = ContainerElements.of(builder(), instances.get(0));
    ContainerElements value = ContainerElements.ofValue(builder(), parameter);
    assertEquals(OptionalInt.of(2), value.arity());
    assertEquals(of.arity(), value.arity());
    assertEquals(of.positions(), value.positions());
    assertEquals(of.isComplete(), value.isComplete());
    assertEquals(of.positionTypes(analysis), value.positionTypes(analysis));
    assertEquals(of.elementTypes(analysis), value.elementTypes(analysis));
    assertEquals(of.mayHoldNone(), value.mayHoldNone());
  }
}
