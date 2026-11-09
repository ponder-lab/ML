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
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ssa.IR;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.OptionalInt;
import java.util.Set;
import org.junit.Test;

/**
 * Tests the elements of a value, read as a client reads them ({@link ValueElements}), on {@code
 * tf2_test_container_elements_of_value.py}, where each sink receives one value: the {@code inputs}
 * a Keras step hands to {@code call} when {@code fit} is given a dataset, which no container
 * instance stands for, and an ordinary local tuple.
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
   * The tensor types the analysis gives a sink's parameter.
   *
   * @param sink The sink's name.
   * @return The types, empty when it gives none.
   */
  private static Set<TensorType> typesOf(String sink) {
    PointsToSetVariable variable =
        builder().getPropagationSystem().findOrCreatePointsToSet(parameter(sink));
    Set<TensorType> state = analysis.getOutState(variable);
    return state == null ? Set.of() : state;
  }

  /**
   * The elements of the value a sink receives.
   *
   * @param sink The sink's name.
   * @return Its parameter's elements.
   */
  private static ValueElements ofValue(String sink) {
    return ValueElements.of(builder(), parameter(sink));
  }

  /**
   * {@code fit(from_tensor_slices(((x, mask), y)).batch(2))} hands {@code call} the tuple {@code
   * (x, mask)}: two positions, the batched {@code x} and {@code mask}.
   */
  @Test
  public void testPair() {
    ValueElements elements = ofValue("consume_pair");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertTrue(elements.isComplete());
    // The component is the element's copy of the pair, so each position has its key.
    assertEquals(2, elements.positions().size());
    assertEquals(
        List.of(Set.of(TensorType.of(FLOAT32, 2, 4)), Set.of(TensorType.of(FLOAT32, 2, 3))),
        elements.positionTypes(analysis));
    assertEquals(
        Set.of(TensorType.of(FLOAT32, 2, 4), TensorType.of(FLOAT32, 2, 3)),
        elements.elementTypes(analysis));
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A dict component has no declared order, so it has no length. It is the element's copy of the
   * dict, so its entries are all its elements, each of its own type.
   */
  @Test
  public void testKeyed() {
    ValueElements elements = ofValue("consume_keyed");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertTrue(elements.isComplete());
    assertEquals(
        Set.of(TensorType.of(FLOAT32, 2, 5), TensorType.of(INT32, 2, 3)),
        elements.elementTypes(analysis));
    assertFalse(elements.mayHoldNone());
  }

  /** A {@code from_generator} component its {@code output_signature} declares as three specs. */
  @Test
  public void testTriple() {
    ValueElements elements = ofValue("consume_triple");
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
    ValueElements elements = ofValue("consume_pair_from_generator");
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
    ValueElements elements = ofValue("consume_zipped_pair");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(Set.of(batched("float32", 4)), Set.of(batched("float32", 3))),
        elements.positionTypes(analysis));
  }

  /**
   * A zipped dataset's whole element, the pair {@code from_tensors} is given: two positions, each
   * the tensor at its position of the pair, so the element is complete and holds no {@code None}
   * (wala/ML#1211).
   */
  @Test
  public void testZippedTensors() {
    ValueElements elements = ofValue("consume_zipped_tensors");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(Set.of(TensorType.of(FLOAT32, 2, 4)), Set.of(TensorType.of(FLOAT32, 2, 3))),
        elements.positionTypes(analysis));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A component {@code ((a, b), m)}: two positions, the first itself a tuple, which is no tensor
   * and whose members are its own elements.
   */
  @Test
  public void testNested() {
    ValueElements elements = ofValue("consume_nested");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        "Positions: " + elements.positionTypes(analysis) + ".",
        Set.of(TensorType.of(FLOAT32, 2, 5)),
        elements.positionTypes(analysis).get(1));
    assertEquals(Set.of(), elements.positionTypes(analysis).get(0));
    assertTrue(elements.isComplete());
  }

  /**
   * A component {@code (*xs, c)}, whose length the display does not show. Its members are the
   * element's copy's, at positions it does not know, each of its own type (wala/ML#1212).
   */
  @Test
  public void testStarred() {
    ValueElements elements = ofValue("consume_starred");
    assertEquals(
        "Positions: " + elements.positionTypes(analysis) + ".",
        OptionalInt.empty(),
        elements.arity());
    assertEquals(
        Set.of(
            TensorType.of(FLOAT32, 2, 4),
            TensorType.of(FLOAT32, 2, 3),
            TensorType.of(FLOAT32, 2, 5)),
        elements.elementTypes(analysis));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A {@code flat_map} whose function returns a dataset of {@code ((a, b), y)} hands {@code call}
   * the pair {@code (a, b)}: the element is the returned dataset's (wala/ML#1214). A batch of a
   * {@code flat_map} may be short, since its cardinality is unknown.
   */
  @Test
  public void testFlatMapped() {
    ValueElements elements = ofValue("consume_flat_mapped");
    assertEquals(
        "Types: " + elements.elementTypes(analysis) + ".", OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(
            Set.of(TensorType.of(FLOAT32, 2, 4), unknownBatchOf(4)),
            Set.of(TensorType.of(FLOAT32, 2, 3), unknownBatchOf(3))),
        elements.positionTypes(analysis));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A {@code float32} batch of rows of a size whose count is unknown.
   *
   * @param size The rows' size.
   * @return The batch's type.
   */
  private static TensorType unknownBatchOf(int size) {
    return new TensorType(FLOAT32, List.of(new SymbolicDim("?"), new NumericDim(size)));
  }

  /** A component {@code tuple(parts)} builds is a tuple the program does not write as a display. */
  @Test
  public void testBuilt() {
    ValueElements elements = ofValue("consume_built");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
  }

  /**
   * In one frame, the dataset is either a structured one or a flat one, so {@code inputs} is a
   * tuple or a tensor: it has no length, and its elements are the tuple's members. The flat tensor
   * is the value itself, not an element of it.
   */
  @Test
  public void testMixed() {
    ValueElements elements = ofValue("consume_mixed");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertEquals(
        Set.of(TensorType.of(FLOAT32, 2, 4), TensorType.of(FLOAT32, 2, 3)),
        elements.elementTypes(analysis));
    assertTrue(elements.isComplete());
  }

  /**
   * As {@link #testMixed}, with the batching applied to the chosen dataset: no length, and the
   * tuple's members, beside an unknown type from the batch over either dataset.
   */
  @Test
  public void testMixedBatched() {
    ValueElements elements = ofValue("consume_mixed_batched");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertTrue(
        "Types: " + elements.elementTypes(analysis) + ".",
        elements
            .elementTypes(analysis)
            .containsAll(Set.of(TensorType.of(FLOAT32, 2, 4), TensorType.of(FLOAT32, 2, 3))));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A shuffled and batched dataset that is either a pair dataset or a triple one: the lengths
   * disagree, so there is none.
   */
  @Test
  public void testLengthsShuffled() {
    ValueElements elements = ofValue("consume_lengths_shuffled");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
  }

  /**
   * A shuffled and batched dataset that is either a flat dataset or a pair one: no length, and the
   * pair's members among its elements.
   */
  @Test
  public void testMixedShuffled() {
    ValueElements elements = ofValue("consume_mixed_shuffled");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(
        "Types: " + elements.elementTypes(analysis) + ".",
        elements
            .elementTypes(analysis)
            .containsAll(Set.of(TensorType.of(FLOAT32, 2, 4), TensorType.of(FLOAT32, 2, 3))));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /**
   * A batched dataset that is either of two pair datasets: they agree on the length, so it has that
   * length, and each position holds the types of both.
   */
  @Test
  public void testAgreeing() {
    ValueElements elements = ofValue("consume_agreeing");
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
   * pair's first member for the inputs, a tensor, which a client reads from the analysis.
   */
  @Test
  public void testMappedInputs() {
    ValueElements elements = ofValue("consume_mapped_inputs");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertEquals(Set.of(TensorType.of(FLOAT32, 2, 4)), typesOf("consume_mapped_inputs"));
  }

  /**
   * A loop over a batched dataset that is either a pair dataset or a triple one: the batch's
   * receiver may be either, whose lengths disagree, so there is none.
   */
  @Test
  public void testLoopLengths() {
    ValueElements elements = ofValue("consume_loop_lengths");
    assertEquals(OptionalInt.empty(), elements.arity());
  }

  /**
   * A loop over a batched dataset that is either of two pair datasets: the batch's receivers agree
   * on the length.
   */
  @Test
  public void testLoopAgreeing() {
    ValueElements elements = ofValue("consume_loop_agreeing");
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
    ValueElements elements = ofValue("consume_mapped_pair");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(Set.of(TensorType.of(FLOAT32, 2, 4)), Set.of(TensorType.of(FLOAT32, 2, 3))),
        elements.positionTypes(analysis));
    assertTrue(elements.isComplete());
  }

  /**
   * A flat dataset input is a tensor: no length, and the analysis gives it its own type, which a
   * client reads from the analysis rather than as an element.
   */
  @Test
  public void testFlat() {
    ValueElements elements = ofValue("consume_flat");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertEquals(Set.of(TensorType.of(FLOAT32, 2, 4)), typesOf("consume_flat"));
  }

  /** An ordinary local tuple reads as its one instance does. */
  @Test
  public void testLocal() {
    PointerKey parameter = parameter("consume_local");
    List<InstanceKey> instances = new ArrayList<>();
    builder().getPointerAnalysis().getPointsToSet(parameter).forEach(instances::add);
    assertEquals(1, instances.size());
    ContainerElements of = ContainerElements.of(builder(), instances.get(0));
    ValueElements value = ValueElements.of(builder(), parameter);
    assertEquals(OptionalInt.of(2), value.arity());
    assertEquals(of.arity(), value.arity());
    assertEquals(of.positions(), value.positions());
    assertEquals(of.isComplete(), value.isComplete());
    assertEquals(ElementTypes.positionTypes(analysis, of), value.positionTypes(analysis));
    assertEquals(ElementTypes.elementTypes(analysis, of), value.elementTypes(analysis));
    assertEquals(of.mayHoldNone(), value.mayHoldNone());
  }

  /**
   * A value that is either of two local pairs merges them: a length of 2, each position the union
   * of both pairs' types, and complete since both are.
   */
  @Test
  public void testLocalJoin() {
    ValueElements elements = ofValue("consume_local_join");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertEquals(
        List.of(
            Set.of(TensorType.of(FLOAT32, 2, 3), TensorType.of(FLOAT32, 2, 5)),
            Set.of(TensorType.of(FLOAT32, 2, 4), TensorType.of(FLOAT32, 2, 6))),
        elements.positionTypes(analysis));
    assertTrue(elements.isComplete());
    assertFalse(elements.mayHoldNone());
  }

  /** A value that is a local pair or a local triple has no one length, but both are complete. */
  @Test
  public void testLocalLengths() {
    ValueElements elements = ofValue("consume_local_lengths");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertTrue(elements.positionTypes(analysis).isEmpty());
    assertEquals(
        Set.of(
            TensorType.of(FLOAT32, 2, 3),
            TensorType.of(FLOAT32, 2, 4),
            TensorType.of(FLOAT32, 2, 5)),
        elements.elementTypes(analysis));
    assertTrue(elements.isComplete());
  }

  /**
   * A {@code map} that builds {@code ((x, x * 2.0), y)}: {@code fit} takes the built pair for the
   * inputs, which the mapped element declares, so it has a length of 2. Both positions are typed,
   * so it is complete.
   */
  @Test
  public void testMappedBuilt() {
    ValueElements elements = ofValue("consume_mapped_built");
    assertEquals(OptionalInt.of(2), elements.arity());
    assertTrue(elements.isComplete());
  }

  /** As {@link #testMappedBuilt()}, the built pair's positions: each is {@code (2, 4)} float32. */
  @Test
  public void testMappedBuiltPositions() {
    ValueElements elements = ofValue("consume_mapped_built");
    assertEquals(
        List.of(Set.of(TensorType.of(FLOAT32, 2, 4)), Set.of(TensorType.of(FLOAT32, 2, 4))),
        elements.positionTypes(analysis));
  }

  /**
   * A dataset's element built from a tuple with a starred part, {@code (*parts, c)}, whose
   * positions the analysis does not know: it has no length, and its element types are every
   * member's, each shape with its own dtype, so a client reads it as a container of those tensors
   * rather than of none (wala/ML#1212).
   */
  @Test
  public void testStarredElement() {
    ValueElements elements = ofValue("consume_starred_element");
    assertEquals(OptionalInt.empty(), elements.arity());
    assertEquals(
        Set.of(TensorType.of(FLOAT32, 4), TensorType.of(INT32, 3), TensorType.of(FLOAT32)),
        elements.elementTypes(analysis));
  }

  /**
   * A member of that element, read by position: any member's type, since the position of each is
   * not known (wala/ML#1212).
   */
  @Test
  public void testStarredMember() {
    assertEquals(
        Set.of(TensorType.of(FLOAT32, 4), TensorType.of(INT32, 3), TensorType.of(FLOAT32)),
        typesOf("consume_starred_member"));
  }

  /**
   * A sliced dataset's element read at one position by path reads as it does by index, which reads
   * some tensor, for every sliced dataset the script holds.
   */
  @Test
  public void testSlicesPathAtOnePosition() {
    PythonSSAPropagationCallGraphBuilder b = builder();
    int read = 0;
    for (CGNode node : b.getCallGraph()) {
      IR ir = node.getIR();
      if (ir == null
          || !node.getMethod().getDeclaringClass().getName().toString().startsWith("Lscript "))
        continue;
      for (int vn = 1; vn <= ir.getSymbolTable().getMaxValueNumber(); vn++) {
        PointerKey value = b.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
        if (b.getPropagationSystem().isImplicit(value)) continue;
        TensorGenerator generator;
        try {
          generator =
              TensorGeneratorFactory.getGenerator(
                  b.getPropagationSystem().findOrCreatePointsToSet(value), b);
        } catch (IllegalArgumentException e) {
          // A value with no generator is not a sliced dataset.
          continue;
        }
        if (generator instanceof DatasetFromTensorSlicesGenerator slices) {
          assertFalse(slices.getTensorTypesForIndex(b, 0).isEmpty());
          assertEquals(
              slices.getTensorTypesForIndex(b, 0), slices.getTensorTypesForPath(b, List.of(0)));
          read++;
        }
      }
    }
    assertTrue("The script holds a sliced dataset.", read > 0);
  }
}
