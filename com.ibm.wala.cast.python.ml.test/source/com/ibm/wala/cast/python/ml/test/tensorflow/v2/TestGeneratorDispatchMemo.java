package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.util.Util.addPytestEntrypoints;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGenerator;
import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import org.junit.Test;

/**
 * Tests that memoizing {@link TensorGeneratorFactory}'s dispatch per builder does not change its
 * answers. For every points-to set variable of a fixture, in the pointer analysis's order and its
 * reverse, with the memo filling as the walk goes, the memoized dispatch must give the same
 * generator, field by field and through every generator it holds, as the dispatch computed afresh,
 * or fail as it does.
 */
public class TestGeneratorDispatchMemo extends AbstractTensorTest {

  /** The package whose objects are compared field by field rather than by their own equality. */
  private static final String ANALYSIS_PACKAGE = "com.ibm.wala.cast.python.ml.";

  /** A recursive function whose result flows back into itself, so the dispatch cuts cycles. */
  @Test
  public void testRecursiveFunction()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_recursive_function.py");
  }

  /**
   * Two functions whose results flow into each other. A dispatch of either result reaches the other
   * and comes back to its own, where it is cut; the other's dispatch, made during it, holds only
   * for that path, since dispatched first it would resolve the first and cut itself.
   */
  @Test
  public void testMutualRecursion()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_dispatch_mutual_recursion.py");
  }

  /** As {@link #testRecursiveFunction()}, with only tensors flowing around the cycle. */
  @Test
  public void testRecursionTensorOnly()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_recursion_tensor_only.py");
  }

  /** Dict comprehension reads, whose generators carry a value binding. */
  @Test
  public void testDictComprehensionItems()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_dict_comprehension_items.py");
  }

  /** A {@code Sequential} fold, which binds arguments onto the layer generators it dispatches. */
  @Test
  public void testSequentialFold()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_sequential_fold.py");
  }

  /** Dataset elements, unwrapped through delegating and receiver generators. */
  @Test
  public void testDatasetZipDict()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_dataset_zip_dict.py");
  }

  /** Containers and iteration passed to Keras layer calls. */
  @Test
  public void testKerasCallContainer()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_keras_symbolic_call_container.py");
  }

  /** Shape arithmetic over binary operations and tuple reads. */
  @Test
  public void testReshapeRoundTrip()
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged("tf2_test_reshape_round_trip.py");
  }

  /**
   * Compares the memoized dispatch of every points-to set variable of a fixture with the unmemoized
   * one, walking the variables in the pointer analysis's order and again, on a fresh call graph, in
   * the reverse order, since what the memo holds when a variable is reached depends on the order.
   *
   * @param fixture The fixture.
   */
  private void assertDispatchUnchanged(String fixture)
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    this.assertDispatchUnchanged(fixture, false);
    this.assertDispatchUnchanged(fixture, true);
  }

  /**
   * Builds the fixture's call graph and compares the memoized dispatch of every points-to set
   * variable with the unmemoized one.
   *
   * @param fixture The fixture.
   * @param reversed Whether to walk the variables in the reverse of the pointer analysis's order.
   */
  private void assertDispatchUnchanged(String fixture, boolean reversed)
      throws ClassHierarchyException, CancelException, IOException, IllegalAccessException {
    PythonTensorAnalysisEngine engine =
        makeEngine(
            PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, this.getPathFiles(""), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    addPytestEntrypoints(builder);
    builder.makeCallGraph(builder.getOptions());
    try {
      int generators = 0;
      // A snapshot: dispatch creates points-to set variables for keys it reads.
      List<PointerKey> keys = new ArrayList<>();
      builder.getPointerAnalysis().getPointerKeys().forEach(keys::add);
      if (reversed) Collections.reverse(keys);
      for (PointerKey key : keys) {
        if (builder.getPropagationSystem().isImplicit(key)) continue;
        PointsToSetVariable variable = builder.getPropagationSystem().findOrCreatePointsToSet(key);
        Object memoized = dispatch(TensorGeneratorFactory::getGenerator, variable, builder);
        Object fresh = dispatch(TensorGeneratorFactory::getUnmemoizedGenerator, variable, builder);
        // A replay, for a variable whose dispatch the first call memoized.
        Object replayed = dispatch(TensorGeneratorFactory::getGenerator, variable, builder);
        assertSameDispatch(key + "", fresh, memoized, new IdentityHashMap<>());
        assertSameDispatch(key + " (replayed)", fresh, replayed, new IdentityHashMap<>());
        if (memoized instanceof TensorGenerator) {
          generators++;
          // Callers bind arguments onto the generator they are given, so a replay is a copy.
          assertTrue(key + "", memoized != replayed);
        }
      }
      assertTrue(fixture + " dispatched no generator", generators > 0);
    } finally {
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * One dispatch.
   *
   * @return The generator, {@code null}, or the {@link IllegalArgumentException} the dispatch
   *     threw.
   */
  private static Object dispatch(
      BiFunction<PointsToSetVariable, PythonSSAPropagationCallGraphBuilder, TensorGenerator>
          factory,
      PointsToSetVariable variable,
      PythonSSAPropagationCallGraphBuilder builder) {
    try {
      return factory.apply(variable, builder);
    } catch (IllegalArgumentException e) {
      return e;
    }
  }

  /**
   * Asserts two dispatch results are the same: the same failure message, or objects of the analysis
   * equal field by field, recursively, and other values equal.
   *
   * @param where What is compared, for the failure message.
   * @param expected The unmemoized result.
   * @param actual The memoized result.
   * @param compared The analysis objects already compared, against their counterparts.
   */
  private static void assertSameDispatch(
      String where, Object expected, Object actual, Map<Object, Object> compared)
      throws IllegalAccessException {
    if (expected == actual) return;
    assertNotNull(where, expected);
    assertNotNull(where, actual);
    assertEquals(where, expected.getClass(), actual.getClass());
    if (expected instanceof IllegalArgumentException e) {
      assertEquals(where, e.getMessage(), ((IllegalArgumentException) actual).getMessage());
      return;
    }
    if (expected instanceof List<?> e) {
      List<?> a = (List<?>) actual;
      assertEquals(where, e.size(), a.size());
      for (int i = 0; i < e.size(); i++)
        assertSameDispatch(where + "[" + i + "]", e.get(i), a.get(i), compared);
      return;
    }
    if (!expected.getClass().getName().startsWith(ANALYSIS_PACKAGE)) {
      assertEquals(where, expected, actual);
      return;
    }
    Object counterpart = compared.get(expected);
    if (counterpart != null) {
      assertSame(where, counterpart, actual);
      return;
    }
    compared.put(expected, actual);
    for (Class<?> c = expected.getClass(); c != null && c != Object.class; c = c.getSuperclass())
      for (Field field : c.getDeclaredFields()) {
        if (Modifier.isStatic(field.getModifiers())) continue;
        field.setAccessible(true);
        assertSameDispatch(
            where + "." + field.getName(), field.get(expected), field.get(actual), compared);
      }
  }
}
