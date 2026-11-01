package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.test.TestPythonMLCallGraphShape;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import java.io.File;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.Test;

/**
 * A read made after the analysis evaluates its queries one inside another, so its depth is the
 * length of the chain of queries its value depends on. Past {@link
 * TensorGenerator#nullEngineReadDepthLimit} a query is answered with the unknown element rather
 * than overflowing the reading thread's stack.
 */
public class NullEngineReadDepthLimitTest extends TestPythonMLCallGraphShape {

  /**
   * A chain of 160 rank-guarded matrix products in one function, whose end a read reaches through
   * every link, 639 queries deep.
   */
  private static final String FIXTURE = "tf2_test_rank_guard_chain.py";

  /** The stack of a thread a client's parallel stream reads on, the platform default. */
  private static final long DEFAULT_STACK = 1L << 20;

  /**
   * Reads the tensor types of the chain's end, the argument of {@code consume}, after the analysis,
   * on a thread with the given stack.
   *
   * @param stack The reading thread's stack size in bytes.
   * @return The types read, rendered.
   */
  private String readChainEnd(long stack) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), FIXTURE);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    engine.performAnalysis(builder);
    try {
      PointerKey value = null;
      for (CGNode node : builder.getCallGraph())
        if (node.getMethod().getDeclaringClass().getName().toString().endsWith("/consume"))
          value = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, 2);
      PointerKey argument = value;
      AtomicReference<String> types = new AtomicReference<>();
      AtomicReference<Throwable> failure = new AtomicReference<>();
      Thread reader =
          new Thread(
              null,
              () -> {
                try {
                  types.set(
                      String.valueOf(
                          TensorGeneratorFactory.getGenerator(
                                  builder.getPropagationSystem().findOrCreatePointsToSet(argument),
                                  builder)
                              .getTensorTypes(builder)));
                } catch (Throwable t) {
                  failure.set(t);
                }
              },
              "chain-reader",
              stack);
      reader.start();
      reader.join();
      assertNull("Expecting the read to complete.", failure.get());
      return types.get();
    } finally {
      TensorGenerator.clearCaches(builder);
    }
  }

  /**
   * The read completes on a thread with the default stack, which evaluating every link of the chain
   * overflows, and keeps the queries it evaluated above the cut, so it reads each link once rather
   * than once per path to it. The end's type does not depend on the links past the cut, so it is
   * exact. The cuts are counted.
   */
  @Test(timeout = 120_000)
  public void testDeepReadCompletes() throws Exception {
    long cuts = TensorGenerator.getNullEngineReadDepthCuts();
    assertEquals("[{[D:Constant,2, D:Constant,4] of float32}]", readChainEnd(DEFAULT_STACK));
    assertTrue(
        "Expecting the read to cut a query.", TensorGenerator.getNullEngineReadDepthCuts() > cuts);
  }

  /**
   * Without the limit, on a stack deep enough to hold the whole chain, the read is exact too, and
   * no query is cut.
   */
  @Test(timeout = 120_000)
  public void testUnlimitedReadOnDeepStack() throws Exception {
    int limit = TensorGenerator.nullEngineReadDepthLimit;
    TensorGenerator.nullEngineReadDepthLimit = Integer.MAX_VALUE;
    try {
      long cuts = TensorGenerator.getNullEngineReadDepthCuts();
      assertEquals("[{[D:Constant,2, D:Constant,4] of float32}]", readChainEnd(64L << 20));
      assertEquals(
          "Expecting no query to be cut.", cuts, TensorGenerator.getNullEngineReadDepthCuts());
    } finally {
      TensorGenerator.nullEngineReadDepthLimit = limit;
    }
  }
}
