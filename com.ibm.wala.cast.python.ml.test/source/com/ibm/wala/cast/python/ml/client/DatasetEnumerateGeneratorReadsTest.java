package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ssa.IR;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;
import org.junit.Test;

/**
 * The per-position and per-path reads of an enumerated dataset's generator, which types the element
 * of {@code enumerate} as {@code (index, element)} (wala/ML#164).
 */
public class DatasetEnumerateGeneratorReadsTest extends TestPythonMLCallGraphShape {

  /**
   * The generators of the enumerated datasets the fixture's script code holds.
   *
   * @param builder The builder.
   * @return The generators.
   */
  private static List<DatasetEnumerateGenerator> enumerated(
      PythonSSAPropagationCallGraphBuilder builder) {
    List<DatasetEnumerateGenerator> ret = new ArrayList<>();
    for (CGNode node : builder.getCallGraph()) {
      IR ir = node.getIR();
      if (ir == null
          || !node.getMethod().getDeclaringClass().getName().toString().startsWith("Lscript "))
        continue;
      for (int vn = 1; vn <= ir.getSymbolTable().getMaxValueNumber(); vn++) {
        PointerKey value =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, vn);
        if (builder.getPropagationSystem().isImplicit(value)) continue;
        try {
          if (TensorGeneratorFactory.getGenerator(
                  builder.getPropagationSystem().findOrCreatePointsToSet(value), builder)
              instanceof DatasetEnumerateGenerator generator) ret.add(generator);
        } catch (RuntimeException e) {
          // A value with no generator is not an enumerated dataset.
        }
      }
    }
    return ret;
  }

  /**
   * Every enumerated dataset's element is a pair whose index is an int64 scalar; positions past the
   * pair, and paths that do not select the element, read nothing.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testReads() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_dataset_enumerate_element.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    List<DatasetEnumerateGenerator> generators = enumerated(builder);
    assertFalse("The fixture enumerates datasets.", generators.isEmpty());
    boolean tuple = false;
    boolean dict = false;
    for (DatasetEnumerateGenerator generator : generators) {
      assertTrue(generator.yieldsTuple(builder));
      assertEquals(OptionalInt.of(2), generator.declaredArity(builder, List.of()));
      assertEquals(Set.of(List.<Dimension<?>>of()), generator.getShapesForIndex(builder, 0));
      assertEquals(EnumSet.of(DType.INT64), generator.getDTypesForIndex(builder, 0));
      assertEquals(Set.of(), generator.getShapesForIndex(builder, 2));
      assertEquals(Set.of(), generator.getDTypesForIndex(builder, 2));
      assertTrue(generator.resolvesPath(builder, List.of(0)));
      assertTrue(generator.resolvesPath(builder, List.of(1)));
      assertFalse(generator.resolvesPath(builder, List.of(2)));
      assertFalse(generator.resolvesPath(builder, List.of("index", 0)));
      assertFalse(generator.isStructureAt(builder, List.of()));
      assertFalse(generator.isStructureAt(builder, List.of(0)));
      assertEquals(OptionalInt.empty(), generator.declaredArity(builder, List.of(0)));
      assertEquals(Set.of(), generator.getShapesForPath(builder, List.of("index", 0)));
      assertEquals(Set.of(), generator.getDTypesForPath(builder, List.of("index", 0)));
      assertEquals(
          generator.getShapesForIndex(builder, 0), generator.getShapesForPath(builder, List.of(0)));
      assertEquals(
          generator.getDTypesForIndex(builder, 0), generator.getDTypesForPath(builder, List.of(0)));
      assertFalse(generator.resolvesPath(builder, List.of("a")));
      assertEquals(Set.of(), generator.getShapesForPath(builder, List.of("a")));
      assertEquals(Set.of(), generator.getDTypesForPath(builder, List.of("a")));
      assertNotNull(generator.getDTypesForIndex(builder, 1));
      if (generator.isStructureAt(builder, List.of(1))) {
        // A tuple element is selected by position, a dict element by key.
        List<Object> member =
            generator.declaredArity(builder, List.of(1)).isPresent()
                ? List.of(1, 0)
                : List.of(1, "a");
        if (member.get(1) instanceof Integer) tuple = true;
        else dict = true;
        assertTrue(generator.resolvesPath(builder, member));
        assertFalse(generator.getShapesForPath(builder, member).isEmpty());
        assertFalse(generator.getDTypesForPath(builder, member).isEmpty());
      }
    }
    assertTrue("The fixture enumerates a dataset whose element is a tuple.", tuple);
    assertTrue("The fixture enumerates a dataset whose element is a dict.", dict);
  }

  /**
   * A generator whose receiver resolves no dataset still reads its index, and reads its element as
   * a tensor of unknown type, not as nothing.
   *
   * @throws Exception On analysis error.
   */
  @Test
  public void testUnresolvedReceiver() throws Exception {
    PythonTensorAnalysisEngine engine =
        makeEngine(Collections.<File>emptyList(), "tf2_test_dataset_enumerate_element.py");
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    CGNode noReceiver = null;
    for (CGNode node : builder.getCallGraph())
      if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/pair_dataset"))
        noReceiver = node;
    assertNotNull("The fixture defines a function that takes no receiver.", noReceiver);
    DatasetEnumerateGenerator generator = new DatasetEnumerateGenerator(noReceiver);
    assertNull(generator.getUnderlyingGenerator(builder));
    assertNull(generator.getShapesForIndex(builder, 1));
    assertEquals(EnumSet.of(DType.UNKNOWN), generator.getDTypesForIndex(builder, 1));
    assertFalse(generator.resolvesPath(builder, List.of(1, 0)));
    assertFalse(generator.isStructureAt(builder, List.of(1)));
    assertEquals(OptionalInt.empty(), generator.declaredArity(builder, List.of(1)));
    assertNull(generator.getShapesForPath(builder, List.of(1, 0)));
    assertEquals(EnumSet.of(DType.UNKNOWN), generator.getDTypesForPath(builder, List.of(1, 0)));
    assertEquals(
        Set.of(new TensorType(DType.INT64.name().toLowerCase(Locale.ROOT), List.of())),
        generator.getTensorTypes(builder));
  }
}
