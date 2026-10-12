package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
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

  /**
   * A dependency cycle longer than the inline depth bound, entered from a member with a base value,
   * settles at that value at every member: the cycle closes at the member it was entered from, as
   * it does on an unbounded stack, rather than being re-entered from a later member on each retry.
   */
  @Test(timeout = 60_000)
  public void testCycleLongerThanInlineBound() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    Object result = engine.demand(0, ringLink(engine, 0), false);

    assertEquals(EnumSet.of(DType.FLOAT32), result);
    assertEquals(
        EnumSet.of(DType.FLOAT32),
        engine.demand(RING_LENGTH / 2, ringLink(engine, RING_LENGTH / 2), false));
  }

  /** The length of the ring in the long-cycle test, beyond the inline depth bound. */
  private static final int RING_LENGTH = 1_000;

  /**
   * The transfer of one link of a dtype ring: link {@code i} reads link {@code (i + 1) mod n}, and
   * link 0 also contributes float32.
   *
   * @param engine The engine.
   * @param i The link's index, which is also its key.
   * @return The link's transfer.
   */
  private static Supplier<Object> ringLink(WorklistTypeResolver engine, int i) {
    return () -> {
      Object next =
          engine.read((i + 1) % RING_LENGTH, ringLink(engine, (i + 1) % RING_LENGTH), false);
      if (i != 0) return next;
      Set<DType> joined = EnumSet.of(DType.FLOAT32);
      if (next instanceof Set<?> set) for (Object o : set) joined.add((DType) o);
      return joined;
    };
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
   * A settled cyclic query whose reads have not changed since its last settled recomputation is not
   * recomputed by a later settlement pass: the recomputation would return the value it settled at.
   * The second demand's own cycle runs that pass.
   */
  @Test
  public void testSettledCycleNotRecomputedWhenUnchanged() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    // The check mode recomputes what the pass skips; this test counts recomputations.
    engine.disableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = {0};
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          runs[0]++;
          engine.read("SKIP_Q", transfers.get(1), true);
          return scalar;
        });
    transfers.add(
        () -> {
          engine.read("SKIP_P", transfers.get(0), true);
          return scalar;
        });
    assertEquals(scalar, engine.demand("SKIP_P", transfers.get(0), true));
    int afterFirstDemand = runs[0];

    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("SKIP_S", others.get(1), true);
          return scalar;
        });
    others.add(
        () -> {
          engine.read("SKIP_R", others.get(0), true);
          return scalar;
        });
    assertEquals(scalar, engine.demand("SKIP_R", others.get(0), true));

    assertEquals(afterFirstDemand, runs[0]);
    assertEquals(scalar, engine.demand("SKIP_P", transfers.get(0), true));
  }

  /**
   * A settled cyclic query whose partner the settlement pass rewrote is recomputed by the next
   * pass, as before: the partner's settled-arm value is a read that changed after the query's own
   * settled recomputation began.
   */
  @Test
  public void testSettledCycleRecomputedWhenPartnerRewritten() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    List<Supplier<Object>> transfers = new ArrayList<>();
    // REWRITE_A passes its partner through and adds a scalar.
    transfers.add(
        () -> ((ShapeResult) engine.read("REWRITE_B", transfers.get(1), true)).union(scalar));
    // REWRITE_B passes REWRITE_A through, and adds a vector only once the state is settled, as a
    // finality-dependent arm does.
    transfers.add(
        () -> {
          ShapeResult a = (ShapeResult) engine.read("REWRITE_A", transfers.get(0), true);
          return engine.isSettling() ? a.union(vector) : a;
        });
    assertEquals(scalar, engine.demand("REWRITE_A", transfers.get(0), true));

    // An unrelated cycle runs another settlement pass, in which REWRITE_A reads the rewritten
    // REWRITE_B.
    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("REWRITE_D", others.get(1), true);
          return scalar;
        });
    others.add(
        () -> {
          engine.read("REWRITE_C", others.get(0), true);
          return scalar;
        });
    engine.demand("REWRITE_C", others.get(0), true);

    assertEquals(scalar.union(vector), engine.demand("REWRITE_A", transfers.get(0), true));
  }

  /**
   * The reads a skip decision compares are those of the query's last settled recomputation, which a
   * later fixpoint evaluation of the query, taking its non-final arm, does not replace.
   */
  @Test
  public void testSettledReadsSurviveFixpointEvaluation() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    List<Supplier<Object>> transfers = new ArrayList<>();
    // SNAP_K reads SNAP_S on its settled arm and SNAP_T otherwise, and is in a cycle with SNAP_J.
    transfers.add(
        () -> {
          engine.read("SNAP_J", transfers.get(1), true);
          engine.read(engine.isSettling() ? "SNAP_S" : "SNAP_T", () -> scalar, true);
          return scalar;
        });
    transfers.add(
        () -> {
          engine.read("SNAP_K", transfers.get(0), true);
          return scalar;
        });
    engine.demand("SNAP_K", transfers.get(0), true);
    assertTrue(engine.settledReadsOf("SNAP_K").contains("SNAP_S"));

    engine.evaluateForTest("SNAP_K");

    assertTrue(engine.settledReadsOf("SNAP_K").contains("SNAP_S"));
  }

  /**
   * The check mode recomputes each query a settlement pass skips, without writing the result back,
   * and passes when the recomputation matches the settled value.
   */
  @Test
  public void testSkipCheckModeVerifiesSkips() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          engine.read("CHECK_Q", transfers.get(1), true);
          return scalar;
        });
    transfers.add(
        () -> {
          engine.read("CHECK_P", transfers.get(0), true);
          return scalar;
        });
    engine.demand("CHECK_P", transfers.get(0), true);
    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("CHECK_S", others.get(1), true);
          return scalar;
        });
    others.add(
        () -> {
          engine.read("CHECK_R", others.get(0), true);
          return scalar;
        });
    engine.demand("CHECK_R", others.get(0), true);

    assertTrue(engine.checkedSkips() > 0);
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
