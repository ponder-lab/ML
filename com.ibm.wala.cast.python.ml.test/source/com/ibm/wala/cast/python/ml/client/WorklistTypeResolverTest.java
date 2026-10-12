package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.After;
import org.junit.Test;

/**
 * Unit tests for the wala/ML#365 Phase 2 engine's value semantics, driven directly with synthetic
 * queries so the corners the integration tests cannot force are pinned: the legacy null-dtype
 * translation, the {@code IllegalArgumentException} fallback, and the pure-cycle promotion.
 *
 * <p>Lives in the {@code client} package (split-package, like {@link LoggablesTest}) for the
 * engine's package-private API.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class WorklistTypeResolverTest {

  @After
  public void uninstall() {
    WorklistTypeResolver.uninstall(null, true);
  }

  /** An acyclic chain evaluates inline and the demanded root sees its dependency's final value. */
  @Test
  public void testAcyclicChain() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    Object result = engine.demand("A", () -> engine.read("B", () -> scalar, true), true);

    assertEquals(scalar, result);
  }

  /**
   * The legacy dtype convention returns {@code null} for ⊤ and callers null-check it to run
   * fallback arms; the engine must hand the reader back {@code null}, not a normalized set.
   */
  @Test
  public void testNullDtypeSurvives() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    assertNull(engine.demand("D", () -> null, false));
  }

  /**
   * A transfer that throws {@code IllegalArgumentException} reads as the unknown value of the
   * query's kind.
   */
  @Test
  public void testIllegalArgumentReadsAsUnknown() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result =
        engine.demand(
            "E",
            () -> {
              throw new IllegalArgumentException("synthetic");
            },
            true);

    assertEquals(ShapeResult.unknown(), result);
  }

  /**
   * A transfer that throws is counted (wala/ML#925). The census's caught-exception count is the
   * instrument the integration witnesses read as 0, and a 0 means nothing unless a throw reads as 1
   * somewhere: this is that control.
   */
  @Test
  public void testCaughtExceptionIsCounted() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    engine.demand(
        "E",
        () -> {
          throw new IllegalArgumentException("synthetic");
        },
        true);
    engine.demand("F", () -> EnumSet.of(DType.FLOAT32), false);

    PythonTensorAnalysisEngine.ResolverCensus census = WorklistTypeResolver.uninstall(null, true);
    assertEquals(1, census.caughtQueryExceptions());
    assertNull("uninstalled once", WorklistTypeResolver.uninstall(null, true));
  }

  /**
   * The catch's fine line names the throwing frame and the first frame outside the throwing class
   * (wala/ML#925): a shared helper's own line names the mechanism, the caller is the site that has
   * to change. Here the transfer throws from this class and is called from the resolver.
   */
  @Test
  public void testCaughtExceptionNamesItsCaller() {
    Logger logger = Logger.getLogger(WorklistTypeResolver.class.getName());
    List<String> messages = new ArrayList<>();
    Handler handler =
        new Handler() {
          @Override
          public void publish(LogRecord record) {
            messages.add(record.getMessage());
          }

          @Override
          public void flush() {}

          @Override
          public void close() {}
        };
    Level old = logger.getLevel();
    logger.setLevel(Level.FINE);
    logger.addHandler(handler);
    try {
      WorklistTypeResolver engine = WorklistTypeResolver.install(null);
      engine.demand(
          "E",
          () -> {
            throw new IllegalArgumentException("synthetic");
          },
          true);
    } finally {
      logger.removeHandler(handler);
      logger.setLevel(old);
    }

    String line =
        messages.stream()
            .filter(m -> m != null && m.startsWith("Worklist transfer IAE"))
            .findFirst()
            .orElse("");
    assertTrue(line, line.contains(WorklistTypeResolverTest.class.getName() + ".lambda$"));
    assertTrue(line, line.contains(" called from " + WorklistTypeResolver.class.getName() + "."));
    assertTrue(line, line.endsWith("treating as ⊤."));
  }

  /**
   * A dependency chain longer than the Java stack can hold as nested evaluations still resolves:
   * each query reads the next, and only the last has a value of its own. Evaluating every
   * first-sighted read inline made the chain's length the stack's depth, which overflowed on a
   * whole-program analysis.
   */
  @Test
  public void testDeepAcyclicChain() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result = engine.demand(0, chainLink(engine, 0, false), false);

    assertEquals(EnumSet.of(DType.FLOAT32), result);
  }

  /**
   * A deep chain whose links read an empty dependency as unknown, as a generator's fallback arm
   * does, settles at the base's value: a link that ran before its dependency was evaluated must not
   * keep the unknown it read then.
   */
  @Test
  public void testDeepChainRetractsInterimUnknown() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result = engine.demand(0, chainLink(engine, 0, true), false);

    assertEquals(EnumSet.of(DType.FLOAT32), result);
  }

  /** The length of the chains in the deep-chain tests, far beyond a default thread stack. */
  private static final int DEEP_CHAIN_LENGTH = 100_000;

  /**
   * The transfer of one link of a deep dtype chain: link {@code i} reads link {@code i + 1}, and
   * the last link is float32.
   *
   * @param engine The engine.
   * @param i The link's index, which is also its key.
   * @param emptyIsUnknown Whether an empty dependency reads as unknown ({@code null}).
   * @return The link's transfer.
   */
  private static Supplier<Object> chainLink(
      WorklistTypeResolver engine, int i, boolean emptyIsUnknown) {
    return () -> {
      if (i == DEEP_CHAIN_LENGTH) return EnumSet.of(DType.FLOAT32);
      Object next = engine.read(i + 1, chainLink(engine, i + 1, emptyIsUnknown), false);
      if (emptyIsUnknown && next instanceof Set<?> set && set.isEmpty()) return null;
      return next;
    };
  }

  /** A dtype query's value passes through unchanged. */
  @Test
  public void testDtypeValue() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result = engine.demand("F", () -> EnumSet.of(DType.FLOAT32), false);

    assertEquals(EnumSet.of(DType.FLOAT32), result);
  }

  /**
   * A dependency cycle with no external base stabilizes at the iteration bottom, which would read
   * as "not a tensor"; the promotion pass must lift it to the unknown-marked element instead.
   */
  @Test
  public void testPureCyclePromotesToUnknown() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    // Mutually recursive transfers: each reads the other and contributes nothing of its own.
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          engine.read("CYCLE_B", transfers.get(1), true);
          return ShapeResult.bottom();
        });
    transfers.add(
        () -> {
          engine.read("CYCLE_A", transfers.get(0), true);
          return ShapeResult.bottom();
        });

    Object result = engine.demand("CYCLE_A", transfers.get(0), true);

    assertEquals(ShapeResult.unknown(), result);
  }

  /**
   * The wala/ML#753 join-history class in vitro: a transfer that collapsed to the unknown-marked
   * element because it consumed an interim (still-bottom) cycle read joins the mark permanently,
   * even though its recomputation against the settled value is precise. The post-fixpoint
   * canonicalization must replace the marked value with the precise recomputation.
   */
  @Test
  public void testCanonicalizationRetractsStaleCollapse() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    List<Supplier<Object>> transfers = new ArrayList<>();
    // STALE_C collapses to ⊤ while its partner is interim and is precise once it settles.
    transfers.add(
        () -> {
          ShapeResult d = (ShapeResult) engine.read("STALE_D", transfers.get(1), true);
          return d.members().isEmpty() ? ShapeResult.unknown() : d;
        });
    // STALE_D reads its partner (recording the cycle) but has its own external base.
    transfers.add(
        () -> {
          engine.read("STALE_C", transfers.get(0), true);
          return scalar;
        });

    assertEquals(scalar, engine.demand("STALE_D", transfers.get(1), true));
    assertEquals(scalar, engine.demand("STALE_C", transfers.get(0), true));
  }

  /**
   * The canonicalization recomputes with the settled state, and a non-monotone transfer may then
   * yield ⊥ where the iteration had settled a real value; the replacement must keep the settled
   * value, since the evidence that the value is a tensor stands and the pure-cycle promotion's ⊤
   * would otherwise be reclassified as "not a tensor."
   */
  @Test
  public void testCanonicalizationKeepsSettledOverBottom() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] evaluations = {0};
    List<Supplier<Object>> transfers = new ArrayList<>();
    // GUARD_G contributes its base only on the first evaluation and ⊥ afterwards.
    transfers.add(
        () -> {
          engine.read("GUARD_H", transfers.get(1), true);
          return evaluations[0]++ == 0 ? scalar : ShapeResult.bottom();
        });
    transfers.add(
        () -> {
          engine.read("GUARD_G", transfers.get(0), true);
          return ShapeResult.bottom();
        });

    assertEquals(scalar, engine.demand("GUARD_G", transfers.get(0), true));
  }

  /**
   * The wala/ML#756 perturbation knob end to end: the seed parses defensively (an unparsable value
   * disables the knob instead of aborting the analysis), and a perturbed engine converges cycles to
   * the same values as the unperturbed one, exercising the shuffle's enqueue and re-enqueue arms.
   * One method rather than one per property scenario: the suite runs test methods in parallel and
   * the knob is a shared system property, so split methods race on it.
   */
  @Test
  public void testShuffleSeedKnob() {
    assertNull(WorklistTypeResolver.parseCycleShuffleSeed(null));
    assertNull(WorklistTypeResolver.parseCycleShuffleSeed("bogus"));
    assertEquals(Long.valueOf(42), WorklistTypeResolver.parseCycleShuffleSeed("42"));
    System.setProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY, "11");
    try {
      testCanonicalizationRetractsStaleCollapse();
    } finally {
      System.clearProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY);
    }
  }
}
