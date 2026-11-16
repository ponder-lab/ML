package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.Test;

/**
 * Tests the generator dispatch of values that form a cycle: the walk computes each value's dispatch
 * a number of times that does not grow with the number of paths through the cycle, and what the
 * values read does not depend on which of them is dispatched first (<a
 * href="https://github.com/wala/ML/issues/1274">wala/ML#1274</a>). Computed again at every visit, a
 * cycle's dispatches numbered its simple paths.
 */
public class TestDispatchCycles extends AbstractTensorTest {

  /** Twelve mutually recursive functions, each returning a call to one of the next two. */
  private static final String RETURN_CYCLE = "tf2_test_dispatch_return_cycle.py";

  /** As {@link #RETURN_CYCLE}, with five functions. */
  private static final String SMALL_RETURN_CYCLE = "tf2_test_dispatch_return_cycle_small.py";

  /** A chain of eighteen binary operators in a loop that carries its first two values. */
  private static final String BINOP_LOOP = "tf2_test_dispatch_binop_loop.py";

  /** Two functions whose results flow into each other. */
  private static final String MUTUAL_RECURSION = "tf2_test_dispatch_mutual_recursion.py";

  /**
   * What either call into the cycle reads: the two calls enter the same functions, which the
   * analysis reads once for both, so each result is either call's tensor.
   */
  private static final Set<TensorType> EITHER_ARGUMENT = Set.of(TENSOR_2_3_FLOAT32, TENSOR_4_INT32);

  /** The result of a call to the cycle's first function. */
  @Test
  public void testReturnCycle() throws ClassHierarchyException, CancelException, IOException {
    test(RETURN_CYCLE, "consume", 1, 1, Map.of(2, EITHER_ARGUMENT));
  }

  /** As {@link #testReturnCycle()}, for a call entering the cycle at another function. */
  @Test
  public void testReturnCycleOtherEntry()
      throws ClassHierarchyException, CancelException, IOException {
    test(RETURN_CYCLE, "consume_other", 1, 1, Map.of(2, EITHER_ARGUMENT));
  }

  /**
   * The analysis computes a few hundred dispatches for the twelve functions. Computing each again
   * at every visit, it computed more than twenty million.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testReturnCycleDispatches() throws Exception {
    long dispatches = this.dispatchComputations(RETURN_CYCLE);
    assertTrue(dispatches + " dispatches", dispatches < 5000);
  }

  /** The chain's last value is a tensor of its argument's type. */
  @Test
  public void testBinopLoop() throws ClassHierarchyException, CancelException, IOException {
    test(BINOP_LOOP, "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_FLOAT32)));
  }

  /**
   * As {@link #testReturnCycleDispatches()}, for the loop's chain, whose analysis computed more
   * than seventy thousand dispatches and now computes fewer than a hundred.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testBinopLoopDispatches() throws Exception {
    long dispatches = this.dispatchComputations(BINOP_LOOP);
    assertTrue(dispatches + " dispatches", dispatches < 1000);
  }

  /**
   * The values of the functions' cycle read the same types whichever of them is dispatched first.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testReturnCycleDispatchOrder() throws Exception {
    this.assertSameReadingsInEitherOrder(SMALL_RETURN_CYCLE);
  }

  /**
   * As {@link #testReturnCycleDispatchOrder()}, for the loop's chain.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testBinopLoopDispatchOrder() throws Exception {
    this.assertSameReadingsInEitherOrder(BINOP_LOOP);
  }

  /**
   * As {@link #testReturnCycleDispatchOrder()}, for two functions whose results flow into each
   * other.
   *
   * @throws Exception On analysis failure.
   */
  @Test
  public void testMutualRecursionDispatchOrder() throws Exception {
    this.assertSameReadingsInEitherOrder(MUTUAL_RECURSION);
  }

  /**
   * How many dispatches a fixture's analysis computes.
   *
   * @param fixture The fixture.
   * @return The count.
   */
  private long dispatchComputations(String fixture) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    return TensorGeneratorFactory.dispatchComputations(builder);
  }

  /**
   * Asserts that every value of a fixture with a generator reads the same types when the values are
   * dispatched in the pointer analysis's order as when they are dispatched in its reverse, so a
   * value in a cycle is reached first from different values.
   *
   * @param fixture The fixture.
   */
  private void assertSameReadingsInEitherOrder(String fixture) throws Exception {
    Map<String, Set<TensorType>> forward = this.readings(fixture, false);
    Map<String, Set<TensorType>> reversed = this.readings(fixture, true);
    assertFalse(fixture + " dispatched no generator", forward.isEmpty());
    assertEquals(forward, reversed);
  }

  /**
   * Dispatches every value of a fixture, as an analysis does, with the resolver installed so the
   * dispatches are memoized, and reads each generator's types.
   *
   * @param fixture The fixture.
   * @param reversed Whether to dispatch the values in the reverse of the pointer analysis's order.
   * @return The types each value with a generator reads, by the value's pointer key.
   */
  private Map<String, Set<TensorType>> readings(String fixture, boolean reversed) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), fixture);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    WorklistTypeResolver.install(builder);
    try {
      // A snapshot: dispatch creates points-to set variables for keys it reads.
      List<PointerKey> keys = new ArrayList<>();
      builder.getPointerAnalysis().getPointerKeys().forEach(keys::add);
      if (reversed) Collections.reverse(keys);
      Map<String, Set<TensorType>> ret = new TreeMap<>();
      for (PointerKey key : keys) {
        if (builder.getPropagationSystem().isImplicit(key)) continue;
        PointsToSetVariable variable = builder.getPropagationSystem().findOrCreatePointsToSet(key);
        TensorGenerator generator;
        try {
          generator = TensorGeneratorFactory.getGenerator(variable, builder);
        } catch (IllegalArgumentException e) {
          continue;
        }
        if (generator == null) continue;
        Set<TensorType> other = ret.put(key + "", generator.getTensorTypes(builder));
        assertTrue(key + " names two values", other == null);
      }
      return ret;
    } finally {
      WorklistTypeResolver.uninstall(builder, false);
      TensorGenerator.clearCaches(builder);
    }
  }
}
