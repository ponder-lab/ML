package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
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

  /**
   * A value that is a constant on one path and the end of a three-step chain on the other, which a
   * read reaches through a few queries.
   */
  private static final String UNION_FIXTURE = "tf2_test_rank_guard_chain_union.py";

  /**
   * A chain of 600 matrix products in one function with no guards, so a read meets every link in
   * the order of the chain.
   */
  private static final String GUARDLESS_FIXTURE = "tf2_test_matmul_chain.py";

  /** A chain of 300 links in one function, each adding a matrix product to its input. */
  private static final String SHARED_FIXTURE = "tf2_test_matmul_chain_shared.py";

  /** How long a read may take before the test fails, within the test's own timeout. */
  private static final long READ_TIMEOUT_MILLIS = 100_000;

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
    return readConsumed(FIXTURE, stack);
  }

  /**
   * Reads the tensor types of the argument of a fixture's {@code consume} after the analysis, on a
   * thread with the given stack.
   *
   * @param fixture The fixture.
   * @param stack The reading thread's stack size in bytes.
   * @return The types read, rendered.
   */
  private String readConsumed(String fixture, long stack) throws Exception {
    PythonTensorAnalysisEngine engine = makeEngine(Collections.<File>emptyList(), fixture);
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
      // A read that does not finish cannot be stopped. The test stops waiting for it and fails
      // within its own timeout, and the thread is a daemon, so it does not keep the fork from
      // exiting.
      reader.setDaemon(true);
      reader.start();
      reader.join(READ_TIMEOUT_MILLIS);
      assertFalse("Expecting the read to finish.", reader.isAlive());
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

  /**
   * A read whose chain is longer than the limit allows still gives the end's exact type: the
   * queries the limit cut are read again from an empty stack, and the read is repeated with their
   * results. Nothing warms the chain before the read meets it, so without the passes the cut
   * decides the answer. The cuts are counted, and none is left standing.
   */
  @Test(timeout = 120_000)
  public void testGuardlessChainIsExact() throws Exception {
    long cuts = TensorGenerator.getNullEngineReadDepthCuts();
    long unrepaired = TensorGenerator.getNullEngineReadUnrepairedCuts();
    assertEquals(
        "[{[D:Constant,2, D:Constant,4] of float32}]",
        readConsumed(GUARDLESS_FIXTURE, DEFAULT_STACK));
    assertTrue(
        "Expecting the read to cut a query.", TensorGenerator.getNullEngineReadDepthCuts() > cuts);
    assertEquals(
        "Expecting every cut to be repaired.",
        unrepaired,
        TensorGenerator.getNullEngineReadUnrepairedCuts());
  }

  /**
   * A chain whose links each read their input twice: the second read finds the first's kept result,
   * so a result that met a cut reaches the read's answer through it as well as through the cut.
   */
  @Test(timeout = 120_000)
  public void testSharedChainIsExact() throws Exception {
    long unrepaired = TensorGenerator.getNullEngineReadUnrepairedCuts();
    assertEquals(
        "[{[D:Constant,2, D:Constant,4] of float32}]", readConsumed(SHARED_FIXTURE, DEFAULT_STACK));
    assertEquals(
        "Expecting every cut to be repaired.",
        unrepaired,
        TensorGenerator.getNullEngineReadUnrepairedCuts());
  }

  /**
   * Without passes, the cut decides the answer of the same read, which is counted as one a cut
   * still decides.
   */
  @Test(timeout = 120_000)
  public void testGuardlessChainWithoutPasses() throws Exception {
    int passes = TensorGenerator.nullEngineReadPassLimit;
    TensorGenerator.nullEngineReadPassLimit = 0;
    try {
      long unrepaired = TensorGenerator.getNullEngineReadUnrepairedCuts();
      assertEquals("[{? of unknown}]", readConsumed(GUARDLESS_FIXTURE, DEFAULT_STACK));
      assertTrue(
          "Expecting the read to be counted as decided by a cut.",
          TensorGenerator.getNullEngineReadUnrepairedCuts() > unrepaired);
    } finally {
      TensorGenerator.nullEngineReadPassLimit = passes;
    }
  }

  /**
   * A cut query whose evaluation throws when it is read again keeps its cut answer: the failure
   * does not escape the read, which gives what it gives with no passes and is counted as one a cut
   * still decides.
   */
  @Test(timeout = 120_000)
  public void testFailingCutReadKeepsCutAnswer() throws Exception {
    int limit = TensorGenerator.nullEngineReadDepthLimit;
    try {
      TensorGenerator.nullEngineReadDepthLimit = 2;
      TensorGenerator.cutReadHook =
          () -> {
            throw new IllegalStateException("A cut query's evaluation failing when read again.");
          };
      long unrepaired = TensorGenerator.getNullEngineReadUnrepairedCuts();
      assertEquals(
          "[{[D:Constant,3, D:Constant,3] of float32}, {? of unknown}]",
          readConsumed(UNION_FIXTURE, DEFAULT_STACK));
      assertTrue(
          "Expecting the read to be counted as decided by a cut.",
          TensorGenerator.getNullEngineReadUnrepairedCuts() > unrepaired);
    } finally {
      TensorGenerator.nullEngineReadDepthLimit = limit;
      TensorGenerator.cutReadHook = null;
    }
  }

  /**
   * A cut contributes the unknown element and keeps the members the read resolved without it: a
   * value that is a constant on one path and a chain's end on the other reads, under a limit the
   * chain exceeds and with no passes to read the cut queries again, as the constant beside an
   * unknown member, and under no limit as both exactly.
   */
  @Test(timeout = 120_000)
  public void testCutKeepsResolvedMembers() throws Exception {
    int limit = TensorGenerator.nullEngineReadDepthLimit;
    int passes = TensorGenerator.nullEngineReadPassLimit;
    try {
      TensorGenerator.nullEngineReadDepthLimit = 2;
      TensorGenerator.nullEngineReadPassLimit = 0;
      long cuts = TensorGenerator.getNullEngineReadDepthCuts();
      assertEquals(
          "[{[D:Constant,3, D:Constant,3] of float32}, {? of unknown}]",
          readConsumed(UNION_FIXTURE, DEFAULT_STACK));
      assertTrue(
          "Expecting the read to cut a query.",
          TensorGenerator.getNullEngineReadDepthCuts() > cuts);
      TensorGenerator.nullEngineReadDepthLimit = Integer.MAX_VALUE;
      assertEquals(
          "[{[D:Constant,3, D:Constant,3] of float32}, {[D:Constant,2, D:Constant,4] of float32}]",
          readConsumed(UNION_FIXTURE, DEFAULT_STACK));
    } finally {
      TensorGenerator.nullEngineReadDepthLimit = limit;
      TensorGenerator.nullEngineReadPassLimit = passes;
    }
  }
}
