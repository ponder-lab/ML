package com.ibm.wala.cast.python.ml.client;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeFalse;

import com.ibm.wala.cast.python.ml.client.WorklistTypeResolver.RecomputationCause;
import com.ibm.wala.cast.python.ml.client.WorklistTypeResolver.RecomputationOutcome;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
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
    assertEquals(
        engine.settlementCensus(),
        2,
        engine.recomputations(RecomputationCause.STALE, RecomputationOutcome.KEPT));
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
   * settled recomputation began. That is what a pass that does not iterate leaves, as a pass inside
   * a generator guard does not; {@link #testIterationSettlesRewrittenPartnerInOnePass} is the same
   * cycle in a pass that iterates.
   */
  @Test
  public void testSettledCycleRecomputedWhenPartnerRewritten() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableIteration();

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
   * A settled cyclic query that reads itself is recomputed by the next pass after the pass wrote it
   * back, as before: its own written value is an input of its recomputation, though the engine
   * records no read edge from a query to itself. That is what a pass that does not iterate leaves;
   * {@link #testIterationReachesSelfReaderFixpoint} is the same cycle in a pass that iterates.
   */
  @Test
  public void testSelfReadingCycleRecomputedAfterOwnWriteBack() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkipCheck();
    engine.disableIteration();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult matrix = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(2))));
    List<Supplier<Object>> transfers = new ArrayList<>();
    // SELF_A reads itself and SELF_B. Once settled, it adds a vector, and a matrix once it holds
    // the
    // vector, so one settled recomputation does not reach its own fixpoint.
    transfers.add(
        () -> {
          ShapeResult self = (ShapeResult) engine.read("SELF_A", transfers.get(0), true);
          ShapeResult b = (ShapeResult) engine.read("SELF_B", transfers.get(1), true);
          if (!engine.isSettling()) return b.union(scalar);
          return self.members().containsAll(vector.members())
              ? self.union(matrix)
              : self.union(vector);
        });
    transfers.add(
        () -> {
          engine.read("SELF_A", transfers.get(0), true);
          return scalar;
        });
    engine.demand("SELF_A", transfers.get(0), true);

    // An unrelated cycle runs another settlement pass.
    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("SELF_D", others.get(1), true);
          return scalar;
        });
    others.add(
        () -> {
          engine.read("SELF_C", others.get(0), true);
          return scalar;
        });
    engine.demand("SELF_C", others.get(0), true);

    assertEquals(
        scalar.union(vector).union(matrix), engine.demand("SELF_A", transfers.get(0), true));
    assertEquals(
        engine.settlementCensus(),
        1,
        engine.recomputations(RecomputationCause.OWN_WRITE, RecomputationOutcome.REPLACED));
  }

  /**
   * Runs one more settlement pass, by demanding a cycle of two new queries that each hold a scalar.
   *
   * @param engine The engine.
   * @param name The name the two queries start with.
   */
  private static void settleAnotherCycle(WorklistTypeResolver engine, String name) {
    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          engine.read(name + "_2", transfers.get(1), true);
          return scalar;
        });
    transfers.add(
        () -> {
          engine.read(name + "_1", transfers.get(0), true);
          return scalar;
        });
    engine.demand(name + "_1", transfers.get(0), true);
  }

  /**
   * The settlement census counts a value that moves back and forth between passes (wala/ML#1293).
   * Two queries on a cycle each take the other's settled value, and a pass recomputes both against
   * the state it found, so each pass swaps the two values. In the first swap, one query has no
   * settled recomputation standing, and the other is stale: it consumed its partner's value before
   * the partner had one. Each later swap is caused by the partner's write-back alone, and gives
   * each query back the value its previous replacement replaced. That is what passes that do not
   * iterate leave; {@link #testIterationSettlesSwapToJoin} is the same cycle in a pass that
   * iterates.
   */
  @Test
  public void testCensusCountsValuesMovingBackAndForth() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkipCheck();
    engine.countRestoredValues(true);
    engine.disableIteration();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          Object partner = engine.read("SWAP_B", transfers.get(1), true);
          return engine.isSettling() ? partner : scalar;
        });
    transfers.add(
        () -> {
          Object partner = engine.read("SWAP_A", transfers.get(0), true);
          return engine.isSettling() ? partner : vector;
        });

    assertEquals(vector, engine.demand("SWAP_A", transfers.get(0), true));
    String first = engine.settlementCensus();
    assertTrue(
        first,
        first.endsWith(
            "recomputations by cause: stale: 0 the same, 1 replaced (0 restored), 0 kept; on a"
                + " cycle with no settled recomputation standing: 0 the same, 1 replaced (0"
                + " restored), 0 kept"));

    settleAnotherCycle(engine, "SWAP_C");
    assertEquals(scalar, engine.demand("SWAP_A", transfers.get(0), true));
    assertEquals(vector, engine.demand("SWAP_B", transfers.get(1), true));
    settleAnotherCycle(engine, "SWAP_D");
    assertEquals(vector, engine.demand("SWAP_A", transfers.get(0), true));

    String census = engine.settlementCensus();
    // The two later passes replaced both queries, and each replacement restored a value.
    assertEquals(
        census,
        4,
        engine.recomputations(RecomputationCause.READ_SETTLED, RecomputationOutcome.REPLACED));
    assertEquals(census, 4, engine.restored(RecomputationCause.READ_SETTLED));
    // Each of the other two cycles has one query with no settled recomputation standing and one
    // stale query, and both kept their values.
    assertEquals(
        census, 2, engine.recomputations(RecomputationCause.NO_RECORD, RecomputationOutcome.SAME));
    assertEquals(
        census, 2, engine.recomputations(RecomputationCause.STALE, RecomputationOutcome.SAME));
    assertEquals(census, 0, engine.restored(RecomputationCause.NO_RECORD));
    assertEquals(census, 0, engine.restored(RecomputationCause.STALE));
    assertTrue(census, census.contains("; 0 passes replaced nothing; "));
    assertTrue(
        census,
        census.endsWith(
            "recomputations by cause: stale: 2 the same, 1 replaced (0 restored), 0 kept; on a"
                + " cycle with no settled recomputation standing: 2 the same, 1 replaced (0"
                + " restored), 0 kept; on a cycle with reads only settlement passes wrote since: 0"
                + " the same, 4 replaced (4 restored), 0 kept"));
  }

  /**
   * Makes a ring of queries. Before the state is settled each holds its initial value, and once it
   * is settled each takes its predecessor's value, changed by the ring's function.
   *
   * @param engine The engine.
   * @param name The name the queries start with; the index of each follows.
   * @param initial The initial value of each query.
   * @param settled The function of a query's index and its predecessor's value.
   * @param runs A counter of the transfers' runs.
   * @return The transfers.
   */
  private static List<Supplier<Object>> ring(
      WorklistTypeResolver engine,
      String name,
      List<ShapeResult> initial,
      BiFunction<Integer, ShapeResult, ShapeResult> settled,
      int[] runs) {
    List<Supplier<Object>> transfers = new ArrayList<>();
    int size = initial.size();
    for (int i = 0; i < size; i++) {
      int index = i;
      int previous = (i + size - 1) % size;
      transfers.add(
          () -> {
            runs[0]++;
            ShapeResult read =
                (ShapeResult) engine.read(name + previous, transfers.get(previous), true);
            return engine.isSettling() ? settled.apply(index, read) : initial.get(index);
          });
    }
    return transfers;
  }

  /**
   * Makes a ring of four queries with a fifth beside it. Once the state is settled each query of
   * the ring takes its predecessor's value, and the fifth takes a function of one ring query's
   * value. The ring's first query reads the fifth and ignores its value, so the five are one
   * component.
   *
   * @param engine The engine.
   * @param name The name the queries start with; the index of each follows, and the fifth's is 4.
   * @param initial The initial value of each of the five queries.
   * @param source The index of the ring query the fifth reads.
   * @param side The function of that query's value.
   * @return The transfers.
   */
  private static List<Supplier<Object>> ringWithSide(
      WorklistTypeResolver engine,
      String name,
      List<ShapeResult> initial,
      int source,
      Function<ShapeResult, ShapeResult> side) {
    List<Supplier<Object>> transfers = new ArrayList<>();
    for (int i = 0; i < 4; i++) {
      int index = i;
      int previous = (i + 3) % 4;
      transfers.add(
          () -> {
            if (index == 0) engine.read(name + 4, transfers.get(4), true);
            Object read = engine.read(name + previous, transfers.get(previous), true);
            return engine.isSettling() ? read : initial.get(index);
          });
    }
    transfers.add(
        () -> {
          ShapeResult read = (ShapeResult) engine.read(name + source, transfers.get(source), true);
          return engine.isSettling() ? side.apply(read) : initial.get(4);
        });
    return transfers;
  }

  /**
   * A settlement pass iterates a cycle to a fixpoint (wala/ML#1293): a query whose partner the pass
   * rewrote is recomputed by that pass, so the first read already returns the value the cycle
   * settles at, and the next pass skips both queries.
   */
  @Test
  public void testIterationSettlesRewrittenPartnerInOnePass() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    int[] runs = {0};
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          runs[0]++;
          return ((ShapeResult) engine.read("ITER_B", transfers.get(1), true)).union(scalar);
        });
    transfers.add(
        () -> {
          runs[0]++;
          ShapeResult a = (ShapeResult) engine.read("ITER_A", transfers.get(0), true);
          return engine.isSettling() ? a.union(vector) : a;
        });

    assertEquals(scalar.union(vector), engine.demand("ITER_A", transfers.get(0), true));
    assertEquals(scalar.union(vector), engine.demand("ITER_B", transfers.get(1), true));
    String census = engine.settlementCensus();
    // One further step recomputes the first query, the reader of the one the pass rewrote, and a
    // second finds the rewritten query unchanged by that.
    assertEquals(census, 2, engine.iterationSteps());
    assertEquals(census, 0, engine.orbits());
    assertEquals(
        census,
        1,
        engine.recomputations(RecomputationCause.ITERATED, RecomputationOutcome.REPLACED));
    // The check mode recomputed both queries at the fixpoint.
    assertTrue(census, census.contains(", 2 queries checked at a fixpoint)"));

    // The next pass skips both: every read of each was written before its last recomputation. The
    // check mode recomputes a skipped query, so the runs are counted with it off.
    engine.disableSkipCheck();
    int settledRuns = runs[0];
    settleAnotherCycle(engine, "ITER_C");
    assertEquals(settledRuns, runs[0]);
    assertEquals(scalar.union(vector), engine.demand("ITER_A", transfers.get(0), true));
    for (RecomputationOutcome outcome : RecomputationOutcome.values())
      assertEquals(0, engine.recomputations(RecomputationCause.READ_SETTLED, outcome));
  }

  /**
   * A pass that iterates a cycle recomputes a query that reads itself after its own write-back
   * (wala/ML#1293), though the engine records no read edge from a query to itself.
   */
  @Test
  public void testIterationReachesSelfReaderFixpoint() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult matrix = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(2))));
    List<Supplier<Object>> transfers = new ArrayList<>();
    // Once settled, the query adds a vector, and a matrix once it holds the vector, so one settled
    // recomputation does not reach its own fixpoint.
    transfers.add(
        () -> {
          ShapeResult self = (ShapeResult) engine.read("OWN_A", transfers.get(0), true);
          ShapeResult b = (ShapeResult) engine.read("OWN_B", transfers.get(1), true);
          if (!engine.isSettling()) return b.union(scalar);
          return self.members().containsAll(vector.members())
              ? self.union(matrix)
              : self.union(vector);
        });
    transfers.add(
        () -> {
          engine.read("OWN_A", transfers.get(0), true);
          return scalar;
        });

    assertEquals(
        scalar.union(vector).union(matrix), engine.demand("OWN_A", transfers.get(0), true));
    String census = engine.settlementCensus();
    assertEquals(
        census,
        1,
        engine.recomputations(RecomputationCause.ITERATED, RecomputationOutcome.REPLACED));
    assertEquals(
        census,
        0,
        engine.recomputations(RecomputationCause.OWN_WRITE, RecomputationOutcome.REPLACED));
  }

  /**
   * Two queries that only swap their values are an orbit with no value in common, so each takes the
   * join of the two (wala/ML#1293): each value is evidence of a tensor, and neither query's value
   * depends any longer on how many passes ran.
   */
  @Test
  public void testIterationSettlesSwapToJoin() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    engine.countRestoredValues(true);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    int[] runs = {0};
    List<Supplier<Object>> transfers =
        ring(engine, "BOTH_", List.of(scalar, vector), (index, read) -> read, runs);

    for (int pass = 0; pass < 3; pass++) {
      assertEquals(scalar.union(vector), engine.demand("BOTH_0", transfers.get(0), true));
      assertEquals(scalar.union(vector), engine.demand("BOTH_1", transfers.get(1), true));
      settleAnotherCycle(engine, "BOTH_PASS_" + pass + "_");
    }
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
    assertEquals(census, 2, engine.orbitJoins());
    assertEquals(census, 0, engine.orbitMeets());
    assertEquals(census, 0, engine.joinedComponents());
    // The swap back that closes the orbit gives each query its value back once, and no later pass
    // moves either again.
    for (RecomputationCause cause : RecomputationCause.values())
      assertEquals(census, cause == RecomputationCause.ITERATED ? 2 : 0, engine.restored(cause));
    assertEquals(census, 2, engine.iterationSteps());
  }

  /**
   * A value that only moves around a cycle, sustained by no query outside it, is dropped by the
   * pass that iterates the cycle (wala/ML#1293): the cycle returns to a state it was in, and each
   * query takes the meet of the values that orbit gave it. Without the iteration the value moves
   * one query a pass, so what a query holds depends on how many passes ran.
   */
  @Test
  public void testIterationDropsValueOnlyTheCycleSustains() {
    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    List<ShapeResult> initial = List.of(scalar.union(vector), scalar, scalar, scalar);

    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    int[] runs = {0};
    List<Supplier<Object>> transfers = ring(engine, "DROP_", initial, (index, read) -> read, runs);
    for (int i = 0; i < 4; i++)
      assertEquals(scalar, engine.demand("DROP_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
    assertEquals(census, 1, engine.orbitMeets());
    assertEquals(census, 0, engine.orbitJoins());
    assertEquals(census, 0, engine.joinedComponents());
    // The ring stands: the next pass recomputes none of it.
    engine.disableSkipCheck();
    int settledRuns = runs[0];
    settleAnotherCycle(engine, "DROP_PASS_");
    assertEquals(settledRuns, runs[0]);

    // The same ring in passes that do not iterate: the vector is at the next query after each.
    uninstall();
    WorklistTypeResolver stepping = WorklistTypeResolver.install(null);
    stepping.disableIteration();
    List<Supplier<Object>> moving =
        ring(stepping, "MOVE_", initial, (index, read) -> read, new int[1]);
    for (int pass = 1; pass <= 5; pass++) {
      if (pass > 1) settleAnotherCycle(stepping, "MOVE_PASS_" + pass + "_");
      for (int i = 0; i < 4; i++)
        assertEquals(
            "After pass " + pass + ", query " + i,
            i == pass % 4 ? scalar.union(vector) : scalar,
            stepping.demand("MOVE_" + i, moving.get(i), true));
    }
  }

  /**
   * A value a query outside the cycle sustains is kept by the pass that iterates the cycle, and
   * reaches every query of the cycle in that pass, while a value only the cycle sustains is dropped
   * (wala/ML#1293). The meet is over the states of an orbit, and the sustained value is in each of
   * them.
   */
  @Test
  public void testIterationKeepsValueSustainedFromOutside() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult matrix = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(2))));
    Supplier<Object> outside = () -> vector;
    // The first query adds what a query outside the ring holds, and the third starts with a matrix
    // that nothing sustains.
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "KEEP_",
            List.of(scalar, scalar, scalar.union(matrix), scalar),
            (index, read) ->
                index == 0
                    ? read.union((ShapeResult) engine.read("KEEP_OUTSIDE", outside, true))
                    : read,
            new int[1]);

    for (int i = 0; i < 4; i++)
      assertEquals(scalar.union(vector), engine.demand("KEEP_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
    assertEquals(census, 1, engine.orbitMeets());
    assertEquals(census, 0, engine.orbitJoins());
  }

  /**
   * The readers of the queries the step that closed an orbit wrote are recomputed after the orbit's
   * meet (wala/ML#1293), also where the meet leaves those queries as they are. Here the query
   * beside the ring holds a value computed from the ring's last query before that query lost the
   * value moving around, and its meet is the value it holds, so only the last query's write makes
   * it a target.
   */
  @Test
  public void testIterationRecomputesReadersOfOrbitClosingStep() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult matrix = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(2))));
    List<Supplier<Object>> transfers =
        ringWithSide(
            engine,
            "LAST_",
            List.of(scalar.union(vector), scalar, scalar, scalar, scalar),
            3,
            read -> read.members().containsAll(vector.members()) ? scalar : scalar.union(matrix));

    assertEquals(scalar.union(matrix), engine.demand("LAST_4", transfers.get(4), true));
    for (int i = 0; i < 4; i++)
      assertEquals(scalar, engine.demand("LAST_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
    assertEquals(census, 1, engine.orbitMeets());
    assertEquals(census, 0, engine.orbitJoins());
    assertEquals(census, 0, engine.joinedComponents());
  }

  /**
   * A query an orbit gave a value is recomputed after it (wala/ML#1293), also where no query it
   * reads was written since: the value is the orbit's and not its recomputation's. Here the query
   * beside the ring held a matrix and then a scalar, which have no value in common, so it takes
   * both, and its recomputation from the settled ring returns the scalar alone.
   */
  @Test
  public void testIterationRecomputesQueryGivenOrbitValue() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult matrix = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(2))));
    List<Supplier<Object>> transfers =
        ringWithSide(
            engine,
            "GIVEN_",
            List.of(scalar.union(vector), scalar, scalar, scalar, scalar),
            1,
            read -> read.members().containsAll(vector.members()) ? matrix : scalar);

    assertEquals(scalar, engine.demand("GIVEN_4", transfers.get(4), true));
    for (int i = 0; i < 4; i++)
      assertEquals(scalar, engine.demand("GIVEN_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
    assertEquals(census, 1, engine.orbitMeets());
    assertEquals(census, 1, engine.orbitJoins());
    assertEquals(census, 0, engine.joinedComponents());
  }

  /**
   * A pass inside a generator guard makes its one step and does not iterate (wala/ML#1293): a guard
   * can cut a recomputation at a frame that is gone by the next pass, so a value that moves there
   * is not an orbit. The next pass outside a guard iterates the cycle.
   */
  @Test
  public void testGuardedPassDoesNotIterate() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "INSIDE_",
            List.of(scalar.union(vector), scalar, scalar, scalar),
            (index, read) -> read,
            new int[1]);

    // The guarded pass moves the vector to the second query and leaves it there.
    assertEquals(
        scalar,
        TensorGenerator.underGuardForTest(() -> engine.demand("INSIDE_0", transfers.get(0), true)));
    assertEquals(scalar.union(vector), engine.demand("INSIDE_1", transfers.get(1), true));
    String census = engine.settlementCensus();
    assertEquals(census, 0, engine.iterationSteps());

    settleAnotherCycle(engine, "INSIDE_PASS_");
    for (int i = 0; i < 4; i++)
      assertEquals(scalar, engine.demand("INSIDE_" + i, transfers.get(i), true));
    census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
  }

  /**
   * The system property turns the iteration off (wala/ML#1293): each pass then makes its one step,
   * and a value that only moves around a cycle is at the next query after it.
   */
  @Test
  public void testIterationCanBeTurnedOff() {
    String oldIterationProperty =
        System.getProperty(WorklistTypeResolver.SETTLEMENT_ITERATION_PROPERTY);
    System.setProperty(WorklistTypeResolver.SETTLEMENT_ITERATION_PROPERTY, "false");
    try {
      WorklistTypeResolver engine = WorklistTypeResolver.install(null);

      ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
      ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
      List<Supplier<Object>> transfers =
          ring(
              engine,
              "OFF_",
              List.of(scalar.union(vector), scalar, scalar, scalar),
              (index, read) -> read,
              new int[1]);

      assertEquals(scalar, engine.demand("OFF_0", transfers.get(0), true));
      assertEquals(scalar.union(vector), engine.demand("OFF_1", transfers.get(1), true));
      String census = engine.settlementCensus();
      assertEquals(census, 0, engine.iterationSteps());
    } finally {
      restore(WorklistTypeResolver.SETTLEMENT_ITERATION_PROPERTY, oldIterationProperty);
    }
  }

  /**
   * Past its budget of recomputations the iteration joins (wala/ML#1293): each query of the cycle
   * then only grows, in this pass and in every later one, so the pass ends, at a value above the
   * one the orbit's meet gives.
   */
  @Test
  public void testIterationJoinsPastBudget() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    engine.setIterationBudget(0, 0);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "PAST_",
            List.of(scalar.union(vector), scalar, scalar, scalar),
            (index, read) -> read,
            new int[1]);

    for (int i = 0; i < 4; i++)
      assertEquals(scalar.union(vector), engine.demand("PAST_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.joinedComponents());
    assertEquals(census, 0, engine.orbits());
  }

  /**
   * Two states of a cycle with one fingerprint are not an orbit unless they are one state
   * (wala/ML#1293): the fingerprint only names a step to compare with, and the writes since that
   * step decide. Here the cycle passes through a state whose values differ from the first state's
   * and have the same hash codes, and goes on from it to its fixpoint.
   */
  @Test
  public void testIterationTellsStatesWithOneFingerprintApart() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult first = ShapeResult.of(Set.of(List.of(new NumericDim(0), new NumericDim(31))));
    ShapeResult second = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult third = ShapeResult.of(Set.of(List.of(new NumericDim(1), new NumericDim(0))));
    ShapeResult last = ShapeResult.of(Set.of(List.of()));
    // The test is of these two: values that differ and hash alike.
    assertNotEquals(first, third);
    assertEquals(first.hashCode(), third.hashCode());
    // The second query takes the value after the first query's, up to the last.
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "ALIKE_",
            List.of(first, first),
            (index, read) ->
                index == 0
                    ? read
                    : read.equals(first) ? second : read.equals(second) ? third : last,
            new int[1]);

    for (int i = 0; i < 2; i++)
      assertEquals(last, engine.demand("ALIKE_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 0, engine.orbits());
    assertEquals(census, 0, engine.joinedComponents());
    assertEquals(census, 6, engine.iterationSteps());
  }

  /**
   * An orbit whose meet moves no query leaves the cycle to join (wala/ML#1293). Here each query
   * holds a scalar and a vector when the other holds no vector, and a scalar when it does, so both
   * grow and shrink together, and each holds the meet of its orbit's values when the orbit closes.
   * The iteration from the meet would be the same orbit again.
   */
  @Test(timeout = 60000)
  public void testIterationJoinsAtOrbitItsMeetLeaves() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "BOTH_",
            List.of(scalar, scalar),
            (index, read) ->
                read.members().containsAll(vector.members()) ? scalar : scalar.union(vector),
            new int[1]);

    for (int i = 0; i < 2; i++)
      assertEquals(scalar.union(vector), engine.demand("BOTH_" + i, transfers.get(i), true));
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.orbits());
    assertEquals(census, 0, engine.orbitMeets());
    assertEquals(census, 0, engine.orbitJoins());
    assertEquals(census, 1, engine.joinedComponents());
    // The step that closes the orbit, the step that joins, and the step that finds nothing to add.
    assertEquals(census, 3, engine.iterationSteps());
  }

  /**
   * A cycle whose transfers are not monotone can leave an orbit again after each meet, and the
   * budget ends that (wala/ML#1293). A query the iteration left to join can then hold more than its
   * recomputation returns, which the check mode accepts for such a query alone. What the queries
   * hold then depends on where the budget ended the iteration: a transfer that is not monotone has
   * no least fixpoint to reach.
   */
  @Test(timeout = 60000)
  public void testIterationEndsOnTransferThatIsNotMonotone() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    engine.setIterationBudget(4, 8);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    // The second query holds a scalar when the first holds a vector, and a vector when it does not.
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "FLIP_",
            List.of(scalar, scalar),
            (index, read) ->
                index == 0 ? read : read.members().containsAll(vector.members()) ? scalar : vector,
            new int[1]);

    // The budget ends the third orbit at the first query, which joins the scalar it reads to the
    // vector it holds; the second then recomputes to the scalar it holds.
    Object joined = engine.demand("FLIP_0", transfers.get(0), true);
    String census = engine.settlementCensus();
    assertEquals(census, scalar.union(vector), joined);
    assertEquals(census, scalar, engine.demand("FLIP_1", transfers.get(1), true));
    assertEquals(census, 1, engine.joinedComponents());
    assertEquals(census, 2, engine.orbits());
    assertEquals(census, 4, engine.orbitJoins());

    // The next pass skips both queries, and the check mode recomputes the first to the scalar it
    // reads.
    settleAnotherCycle(engine, "FLIP_PASS_");
    assertEquals(scalar.union(vector), engine.demand("FLIP_0", transfers.get(0), true));
    census = engine.settlementCensus();
    assertTrue(census, census.contains(" skipped (0 inside a generator guard, 2 checked, "));
  }

  /**
   * A cycle whose value grows at every step reaches no fixpoint and no orbit, and the budget ends
   * it (wala/ML#1293): the joined values pass the widening cap, and a query that did is not written
   * again, so a join cannot restart from the widened value.
   */
  @Test(timeout = 60000)
  public void testIterationEndsOnValueThatAlwaysGrows() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    engine.setIterationBudget(1, 6);

    ShapeResult first = ShapeResult.of(Set.of(List.of(new NumericDim(0))));
    // The second query holds a vector one longer than the longest the first holds, and starts
    // again from a vector of one element when the first holds none, as it does once widened.
    List<Supplier<Object>> transfers =
        ring(
            engine,
            "GROW_",
            List.of(first, first),
            (index, read) -> {
              if (index == 0) return read;
              int longest = 0;
              for (List<Dimension<?>> member : read.members())
                longest = Math.max(longest, ((NumericDim) member.get(0)).value());
              return ShapeResult.of(Set.of(List.of(new NumericDim(longest + 1))));
            },
            new int[1]);

    assertTrue(((ShapeResult) engine.demand("GROW_0", transfers.get(0), true)).hasUnknown());
    assertTrue(((ShapeResult) engine.demand("GROW_1", transfers.get(1), true)).hasUnknown());
    String census = engine.settlementCensus();
    assertEquals(census, 1, engine.joinedComponents());
    assertEquals(census, 0, engine.orbits());
    settleAnotherCycle(engine, "GROW_PASS_");
    assertTrue(((ShapeResult) engine.demand("GROW_0", transfers.get(0), true)).hasUnknown());
  }

  /**
   * The check mode fails when a query recomputes, at the fixpoint the iteration reached, to a value
   * other than the one it holds (wala/ML#1293): its transfer is then not a function of its reads,
   * and the next pass's skip of it would keep a value its recomputation does not return.
   */
  @Test
  public void testFixpointCheckFailsOnTransferThatIsNotAFunctionOfItsReads() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    ShapeResult matrix = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(2))));
    int[] settledRuns = {0};
    List<Supplier<Object>> transfers = new ArrayList<>();
    // The first settled recomputation returns a vector, and each later one a matrix.
    transfers.add(
        () -> {
          engine.read("ONCE_B", transfers.get(1), true);
          if (!engine.isSettling()) return scalar;
          return settledRuns[0]++ == 0 ? vector : matrix;
        });
    transfers.add(
        () -> {
          engine.read("ONCE_A", transfers.get(0), true);
          return scalar;
        });

    try {
      engine.demand("ONCE_A", transfers.get(0), true);
    } catch (AssertionError e) {
      assertTrue(
          e.getMessage(),
          e.getMessage().startsWith("A settled recomputation at a fixpoint differs: "));
      return;
    }
    fail("The check mode missed a fixpoint that does not stand.");
  }

  /**
   * The settlement census tells a query the pass recomputes at its transfer's request from a reader
   * the pass recomputes because the first one's value changed (wala/ML#1293). Neither is on a
   * cycle.
   */
  @Test
  public void testCensusCountsRequestedAndChangedReads() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.countRestoredValues(false);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(2))));
    Supplier<Object> leaf =
        () -> {
          engine.requestSettledRecomputation();
          return engine.isSettling() ? vector : scalar;
        };
    Supplier<Object> reader = () -> engine.read("CAUSE_LEAF", leaf, true);

    assertEquals(vector, engine.demand("CAUSE_READER", reader, true));
    String census = engine.settlementCensus();
    // An engine that does not count restored values leaves them out of the line.
    assertTrue(
        census,
        census.endsWith(
            "recomputations by cause: requested: 0 the same, 1 replaced, 0 kept; reading a query"
                + " the pass changed: 0 the same, 1 replaced, 0 kept"));
    for (RecomputationCause cause : RecomputationCause.values())
      for (RecomputationOutcome outcome : RecomputationOutcome.values())
        assertEquals(
            census,
            outcome == RecomputationOutcome.REPLACED
                    && (cause == RecomputationCause.REQUESTED
                        || cause == RecomputationCause.CHANGED_READ)
                ? 1
                : 0,
            engine.recomputations(cause, outcome));
  }

  /**
   * The settlement census tells a read an evaluation wrote from a read only settlement passes wrote
   * (wala/ML#1293). A query on a cycle reads a third query only when it is recomputed at
   * settlement, so the third query is first evaluated inside that recomputation, after the clock
   * the recomputation's record holds. The next pass finds the record older than an evaluation's
   * write and recomputes the query, to the same value. The pass after that skips it.
   *
   * <p>That one recomputation is a skip the engine misses, and not a value it needs: the record
   * holds the clock at the recomputation's entry, and the read is evaluated after it but before it
   * is consumed. A change that skips it moves this test's second-pass counts, and is an
   * improvement.
   */
  @Test
  public void testCensusCountsReadFirstEvaluatedAtSettlement() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkipCheck();
    engine.countRestoredValues(true);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = {0};
    Supplier<Object> late = () -> scalar;
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          runs[0]++;
          engine.read("LATE_B", transfers.get(1), true);
          if (engine.isSettling()) engine.read("LATE_READ", late, true);
          return scalar;
        });
    transfers.add(
        () -> {
          engine.read("LATE_A", transfers.get(0), true);
          return scalar;
        });

    assertEquals(scalar, engine.demand("LATE_A", transfers.get(0), true));
    assertEquals(
        engine.settlementCensus(),
        0,
        engine.recomputations(RecomputationCause.READ_EVALUATED, RecomputationOutcome.SAME));

    settleAnotherCycle(engine, "LATE_C");
    int afterSecondPass = runs[0];
    String census = engine.settlementCensus();
    assertEquals(
        census,
        1,
        engine.recomputations(RecomputationCause.READ_EVALUATED, RecomputationOutcome.SAME));
    assertTrue(
        census,
        census.contains(
            "on a cycle with a read an evaluation or a promotion wrote since: 1 the same, 0"
                + " replaced (0 restored), 0 kept"));

    settleAnotherCycle(engine, "LATE_D");
    assertEquals(afterSecondPass, runs[0]);
    census = engine.settlementCensus();
    assertEquals(
        census,
        1,
        engine.recomputations(RecomputationCause.READ_EVALUATED, RecomputationOutcome.SAME));
    assertEquals(
        census,
        0,
        engine.recomputations(RecomputationCause.READ_SETTLED, RecomputationOutcome.SAME));
    assertTrue(census, census.contains("; 3 passes replaced nothing; "));
  }

  /** With the skip off, a later settlement pass recomputes a settled cycle, as before the skip. */
  @Test
  public void testSettlementSkipCanBeTurnedOff() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkip();
    engine.disableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = {0};
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          runs[0]++;
          engine.read("OFF_Q", transfers.get(1), true);
          return scalar;
        });
    transfers.add(
        () -> {
          engine.read("OFF_P", transfers.get(0), true);
          return scalar;
        });
    engine.demand("OFF_P", transfers.get(0), true);
    int afterFirstDemand = runs[0];
    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("OFF_S", others.get(1), true);
          return scalar;
        });
    others.add(
        () -> {
          engine.read("OFF_R", others.get(0), true);
          return scalar;
        });
    engine.demand("OFF_R", others.get(0), true);

    assertTrue(runs[0] > afterFirstDemand);
    // The first pass recomputed the cycle for other causes, and the second for the skip being off.
    assertEquals(
        engine.settlementCensus(),
        4,
        engine.recomputations(RecomputationCause.SKIP_OFF, RecomputationOutcome.SAME));
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
    String oldShuffleCyclesProperty =
        System.getProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY);
    System.setProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY, "11");
    try {
      testCanonicalizationRetractsStaleCollapse();
    } finally {
      restore(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY, oldShuffleCyclesProperty);
    }
  }

  /**
   * A binding check that fails at convergence declines a bound root that already settled typed
   * (wala/ML#1071): the root re-evaluates to the unknown-marked element, which the join keeps as an
   * unknown remainder beside the typed members, so the root's legacy view collapses to ⊤, and a
   * query reading the root takes the change. The decline must not keep the prior answer.
   */
  @Test
  public void testFailedBindingCheckDeclinesSettledShapes() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    ShapeResult typed = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(3))));
    Supplier<Object> root =
        () -> {
          engine.registerBindingCheck("binding", resolver -> false, "root");
          return engine.isBindingDeclined("binding") ? ShapeResult.unknown() : typed;
        };

    ShapeResult reader =
        (ShapeResult) engine.demand("reader", () -> engine.read("root", root, true), true);

    assertTrue("The binding was not declined.", engine.isBindingDeclined("binding"));
    ShapeResult settled = (ShapeResult) engine.demand("root", root, true);
    assertEquals(typed.members(), settled.members());
    assertTrue("The decline left no unknown remainder.", settled.hasUnknown());
    assertNull("The declined root's legacy view is not ⊤.", settled.toLegacy());
    assertEquals("The root's reader did not take the decline.", settled, reader);
  }

  /**
   * The dtype counterpart: a root that settled {@code float32} and then declines joins in {@link
   * DType#UNKNOWN}, which the engine's dtype join keeps as a member beside {@code float32}, so the
   * root and its reader read the partial union rather than the prior set (wala/ML#1071), as the
   * shape twin reads its typed members with an unknown remainder (wala/ML#862).
   */
  @Test
  public void testFailedBindingCheckDeclinesSettledDTypes() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    Supplier<Object> root =
        () -> {
          engine.registerBindingCheck("binding", resolver -> false, "root");
          return engine.isBindingDeclined("binding")
              ? EnumSet.of(DType.UNKNOWN)
              : EnumSet.of(DType.FLOAT32);
        };

    Object reader = engine.demand("reader", () -> engine.read("root", root, false), false);

    assertTrue("The binding was not declined.", engine.isBindingDeclined("binding"));
    assertEquals(EnumSet.of(DType.FLOAT32, DType.UNKNOWN), engine.demand("root", root, false));
    assertEquals(EnumSet.of(DType.FLOAT32, DType.UNKNOWN), reader);
  }

  /** A binding check that passes declines nothing, and the root keeps its typed answer. */
  @Test
  public void testPassingBindingCheckKeepsShapes() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    ShapeResult typed = ShapeResult.of(Set.of(List.of(new NumericDim(2), new NumericDim(3))));
    Supplier<Object> root =
        () -> {
          engine.registerBindingCheck("binding", resolver -> true, "root");
          return engine.isBindingDeclined("binding") ? ShapeResult.unknown() : typed;
        };

    assertEquals(typed, engine.demand("reader", () -> engine.read("root", root, true), true));
    assertTrue("The binding was declined.", !engine.isBindingDeclined("binding"));
  }

  /**
   * A binding declined after a cyclic bound root settled is a change of the root's input, though
   * not of a query it reads: the next settlement pass recomputes the root to the declined answer,
   * as it does with the skip off, instead of skipping it as unchanged. Here the root already holds
   * an unknown remainder, so the declined evaluation's join leaves its state, and so its stamps, as
   * they were. The second demand's unrelated cycle runs the validation that declines the binding
   * and then the settlement pass.
   */
  @Test
  public void testBindingDeclinedAfterSettlementIsNotSkipped() {
    assertEquals(ShapeResult.unknown(), declineAfterSettlement(true));
  }

  /** The control: with the skip off, the declined root recomputes to the declined answer. */
  @Test
  public void testBindingDeclinedAfterSettlementWithoutSkip() {
    assertEquals(ShapeResult.unknown(), declineAfterSettlement(false));
  }

  /**
   * Settles a cyclic root whose binding check passes, then fails the check and runs a second
   * settlement pass.
   *
   * @param skip Whether the settlement skip is on.
   * @return The root's value after the second pass.
   */
  private static Object declineAfterSettlement(boolean skip) {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    if (!skip) engine.disableSkip();
    // The check mode recomputes what the pass skips; this test reads the value the skip leaves.
    engine.disableSkipCheck();
    ShapeResult typed =
        ShapeResult.of(Set.of(List.of(new NumericDim(2)))).union(ShapeResult.unknown());
    boolean[] applies = {true};
    List<Supplier<Object>> transfers = new ArrayList<>();
    // DECLINE_ROOT is bound, and in a cycle with DECLINE_PARTNER.
    transfers.add(
        () -> {
          engine.registerBindingCheck("binding", resolver -> applies[0], "DECLINE_ROOT");
          engine.read("DECLINE_PARTNER", transfers.get(1), true);
          return engine.isBindingDeclined("binding") ? ShapeResult.unknown() : typed;
        });
    transfers.add(
        () -> {
          engine.read("DECLINE_ROOT", transfers.get(0), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    assertEquals(typed, engine.demand("DECLINE_ROOT", transfers.get(0), true));

    // An unrelated settlement pass in which the root's skip is observed to hold, so the root is no
    // longer marked for the restricted pass by its own settled recomputation.
    List<Supplier<Object>> before = new ArrayList<>();
    before.add(
        () -> {
          engine.read("DECLINE_Q", before.get(1), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    before.add(
        () -> {
          engine.read("DECLINE_P", before.get(0), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    engine.demand("DECLINE_P", before.get(0), true);

    applies[0] = false;
    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("DECLINE_S", others.get(1), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    others.add(
        () -> {
          engine.read("DECLINE_R", others.get(0), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    engine.demand("DECLINE_R", others.get(0), true);

    assertTrue("The binding was not declined.", engine.isBindingDeclined("binding"));
    return engine.demand("DECLINE_ROOT", transfers.get(0), true);
  }

  /**
   * The check mode verifies the settlement restriction against the whole graph: each restricted
   * pass's components are the whole condensation's that meet the region, in its order, and no query
   * outside the region is targeted, promoted or evaluated. The scenarios are the settlement tests
   * that leave the check mode on. One method, as for {@link #testShuffleSeedKnob}: the check mode
   * is a shared system property.
   */
  @Test
  public void testRestrictedSettlementMatchesWholeGraph() {
    String oldCheckSkipsProperty = System.getProperty(WorklistTypeResolver.CHECK_SKIPS_PROPERTY);
    System.setProperty(WorklistTypeResolver.CHECK_SKIPS_PROPERTY, "true");
    try {
      testPureCyclePromotesToUnknown();
      uninstall();
      testCanonicalizationRetractsStaleCollapse();
      uninstall();
      testCanonicalizationKeepsSettledOverBottom();
      uninstall();
      testSettledCycleRecomputedWhenPartnerRewritten();
      uninstall();
      testSettledReadsSurviveFixpointEvaluation();
    } finally {
      restore(WorklistTypeResolver.CHECK_SKIPS_PROPERTY, oldCheckSkipsProperty);
    }
  }

  /**
   * With the restriction off, a settled cycle that a rewritten partner makes stale is recomputed
   * all the same: the restriction changes which queries a pass walks, not what it computes.
   */
  @Test
  public void testSettlementRestrictionCanBeTurnedOff() {
    String oldRestrictSettlementProperty =
        System.getProperty(WorklistTypeResolver.RESTRICT_SETTLEMENT_PROPERTY);
    System.setProperty(WorklistTypeResolver.RESTRICT_SETTLEMENT_PROPERTY, "false");
    try {
      testSettledCycleRecomputedWhenPartnerRewritten();
    } finally {
      restore(WorklistTypeResolver.RESTRICT_SETTLEMENT_PROPERTY, oldRestrictSettlementProperty);
    }
  }

  /**
   * A cycle that forms without any query's value changing is settled all the same. A declined
   * binding re-evaluates its root, which now reads a partner that reads the root back, and the
   * declined evaluation's join leaves the root's state as it was. So no query in the new cycle is
   * changed, stale, requested or a shape bottom, and only the read edges that formed it put it in
   * the restricted pass's region.
   */
  @Test
  public void testCycleFormedWithoutChangeIsSettled() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    // The check mode recomputes what the pass skips; this test reads the value the pass leaves.
    engine.disableSkipCheck();
    ShapeResult typed =
        ShapeResult.of(Set.of(List.of(new NumericDim(2)))).union(ShapeResult.unknown());
    boolean[] applies = {true};
    List<Supplier<Object>> transfers = new ArrayList<>();
    // EDGE_ROOT is bound, and reads EDGE_PARTNER only once its binding is declined.
    transfers.add(
        () -> {
          engine.registerBindingCheck("edge", resolver -> applies[0], "EDGE_ROOT");
          if (!engine.isBindingDeclined("edge")) return typed;
          engine.read("EDGE_PARTNER", transfers.get(1), true);
          return ShapeResult.unknown();
        });
    transfers.add(
        () -> {
          engine.read("EDGE_ROOT", transfers.get(0), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    assertEquals(typed, engine.demand("EDGE_ROOT", transfers.get(0), true));

    applies[0] = false;
    List<Supplier<Object>> others = new ArrayList<>();
    others.add(
        () -> {
          engine.read("EDGE_S", others.get(1), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    others.add(
        () -> {
          engine.read("EDGE_R", others.get(0), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    engine.demand("EDGE_R", others.get(0), true);

    assertTrue("The binding was not declined.", engine.isBindingDeclined("edge"));
    assertEquals(ShapeResult.unknown(), engine.demand("EDGE_ROOT", transfers.get(0), true));
  }

  /**
   * The staleness scan's check mode fails when a stale reader is not a candidate (wala/ML#1176):
   * the scan checks only candidates, and the check compares it against the whole scan.
   */
  @Test
  public void testStaleNonCandidateFailsCheck() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    engine.demand("LAG_A", () -> engine.read("LAG_B", () -> scalar, true), true);
    // A first scan finds the reader current and drops it from the candidates.
    demandPureCycle(engine, "LAG_E", "LAG_F");
    engine.lagRecordForTest("LAG_A", "LAG_B");

    try {
      demandPureCycle(engine, "LAG_C", "LAG_D");
    } catch (AssertionError e) {
      assertEquals("A reader outside the staleness candidates is stale.", e.getMessage());
      return;
    }
    fail("The check mode missed a stale reader outside the candidates.");
  }

  /**
   * A reader whose read value changes while it cannot be re-enqueued is a staleness candidate, so
   * the scan finds and re-enqueues it, and the check mode agrees (wala/ML#1176).
   */
  @Test
  public void testStaleCandidateFoundByScan() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();
    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    engine.demand("BUMP_A", () -> engine.read("BUMP_B", () -> scalar, true), true);
    // A first scan finds the reader current and drops it from the candidates.
    demandPureCycle(engine, "BUMP_E", "BUMP_F");
    engine.bumpVersionForTest("BUMP_B");

    demandPureCycle(engine, "BUMP_C", "BUMP_D");

    assertEquals(1, engine.staleScanHits());
    assertEquals(scalar, engine.demand("BUMP_A", () -> scalar, true));
  }

  /**
   * The walk census counts a pure cycle's promotion walks and settlement sweep, and the queries the
   * graph held at each walk, their closures held and their condensations emitted (wala/ML#1176).
   */
  @Test
  public void testWalkCensusCountsCycleWalks() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    demandPureCycle(engine, "WALK_A", "WALK_B");

    // Two promotion walks, the second over an empty closure once the cycle is promoted, then one
    // restricted sweep, all in one outermost solve.
    assertArrayEquals(new long[] {2, 0, 4, 2, 2, 1, 0, 0, 2, 2, 2, 1}, engine.walkCounts());
    assertTrue(
        engine.walkCensus().startsWith("2 queries and 2 dependency edges; 1 outermost solves in "));
  }

  /**
   * With the settlement skip off, the census counts the full sweep as a whole-graph walk, which has
   * no closure (wala/ML#1176).
   */
  @Test
  public void testWalkCensusCountsWholeGraphSweep() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkip();

    demandPureCycle(engine, "WHOLE_A", "WHOLE_B");

    assertArrayEquals(new long[] {2, 0, 4, 2, 2, 1, 1, 0, 2, 0, 2, 1}, engine.walkCounts());
  }

  /**
   * A promotion pass walks only the reader closure of what changed since the previous pass, not of
   * every bottom-valued query (wala/ML#1176): a bottom-valued pair that is on no cycle, and an
   * already promoted cycle, are outside the closure of a later pass that promotes a new cycle.
   */
  @Test
  public void testPromotionWalksOnlyWhatChanged() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    // An acyclic bottom-valued pair stays in the bottoms, and its solve runs no promotion pass.
    engine.demand("SEED_X", () -> engine.read("SEED_Y", () -> ShapeResult.bottom(), true), true);
    demandPureCycle(engine, "SEED_A", "SEED_B");
    long[] first = engine.walkCounts();
    demandPureCycle(engine, "SEED_C", "SEED_D");
    long[] second = engine.walkCounts();

    // The first cycle's passes walk the pair and the cycle, then nothing once the cycle is
    // promoted; the second cycle's passes walk the second cycle alone, then nothing.
    assertEquals(2, first[0]);
    assertEquals(4, first[3]);
    assertEquals(4, second[0]);
    assertEquals(2, second[3] - first[3]);
  }

  /**
   * A promotion pass sees a cycle that a new edge closes between two queries that were already
   * bottom-valued at the previous pass (wala/ML#1176). Neither query enters the bottoms again, so
   * only the edge's reader seeds the pass, and the pass promotes both.
   */
  @Test
  public void testPromotionSeesCycleClosedByNewEdge() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    boolean[] applies = {true};
    List<Supplier<Object>> transfers = new ArrayList<>();
    // CLOSE_ROOT reads CLOSE_PARTNER, which is bound and reads CLOSE_ROOT back only once its
    // binding is declined.
    transfers.add(
        () -> {
          engine.read("CLOSE_PARTNER", transfers.get(1), true);
          return ShapeResult.bottom();
        });
    transfers.add(
        () -> {
          engine.registerBindingCheck("close", resolver -> applies[0], "CLOSE_PARTNER");
          if (engine.isBindingDeclined("close")) engine.read("CLOSE_ROOT", transfers.get(0), true);
          return ShapeResult.bottom();
        });
    assertEquals(ShapeResult.bottom(), engine.demand("CLOSE_ROOT", transfers.get(0), true));
    // A promotion pass runs while the pair is acyclic and bottom-valued.
    demandPureCycle(engine, "CLOSE_A", "CLOSE_B");

    applies[0] = false;
    demandPureCycle(engine, "CLOSE_C", "CLOSE_D");

    assertTrue("The binding was not declined.", engine.isBindingDeclined("close"));
    assertEquals(ShapeResult.unknown(), engine.demand("CLOSE_ROOT", transfers.get(0), true));
  }

  /**
   * The sweep's set intersection iterates the smaller set, whichever argument it is, so a query's
   * few reads tested against a large changed set cost the reads, not the changed set
   * (wala/ML#1176): {@link java.util.Collections#disjoint} iterates its second argument whenever
   * its first is a set.
   */
  @Test
  public void testIntersectionIteratesTheSmallerSet() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    Set<Object> changed = new HashSet<>();
    for (int i = 0; i < 4096; i++) changed.add("CHANGED" + i);

    assertFalse(engine.intersects(Set.of("READ"), changed));
    assertEquals(1, engine.readTestCounts()[3]);
    assertFalse(engine.intersects(changed, Set.of("READ")));
    assertEquals(2, engine.readTestCounts()[3]);

    // A shared element is found, from either side.
    assertTrue(engine.intersects(Set.of("CHANGED7"), changed));
    assertTrue(engine.intersects(changed, Set.of("READ", "CHANGED7")));
    assertTrue(engine.readTestCounts()[3] <= 5);

    // An empty set shares nothing and costs no probe.
    assertFalse(engine.intersects(Set.of(), changed));
    assertFalse(engine.intersects(changed, Set.of()));
    assertTrue(engine.readTestCounts()[3] <= 5);
  }

  /**
   * A star in one settlement sweep: leaves whose settled recomputation differs fill the changed
   * set, and readers that read none of them are tested against it, so with the changed set iterated
   * the probes would grow with the readers times the leaves (wala/ML#1176). The readers each read
   * only a gate over the leaves, which orders every reader after every leaf.
   */
  @Test
  public void testSweepReadTestsGrowWithTheReads() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    int leaves = 64;
    int readers = 64;
    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(3))));

    // Each leaf asks for a settled recomputation and recomputes, settling, to another value.
    List<Supplier<Object>> leafTransfers = new ArrayList<>();
    for (int i = 0; i < leaves; i++)
      leafTransfers.add(
          () -> {
            engine.requestSettledRecomputation();
            return engine.isSettling() ? vector : scalar;
          });
    Supplier<Object> gate =
        () -> {
          for (int i = 0; i < leaves; i++) engine.read("LEAF" + i, leafTransfers.get(i), true);
          return scalar;
        };
    Supplier<Object> reader =
        () -> {
          engine.read("GATE", gate, true);
          return scalar;
        };

    Object result =
        engine.demand(
            "STAR",
            () -> {
              engine.read("GATE", gate, true);
              for (int j = 0; j < readers; j++) engine.read("READER" + j, reader, true);
              return scalar;
            },
            true);

    assertEquals(scalar, result);
    for (int i = 0; i < leaves; i++)
      assertEquals(vector, engine.demand("LEAF" + i, leafTransfers.get(i), true));
    long[] counts = engine.readTestCounts();
    // The readers' tests alone would make leaves × readers probes with the changed set iterated.
    assertTrue(counts[2] >= (long) leaves * readers);
    assertTrue(counts[3] < (long) leaves * readers / 2);
  }

  /**
   * The settlement sweep selects its targets through the read test, so the read-set census counts
   * the tests, their reads and their probes, and the probes stay within the reads (wala/ML#1176).
   */
  @Test
  public void testSweepCountsItsReadTests() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    for (int i = 0; i < 4; i++) demandPureCycle(engine, "READS_A" + i, "READS_B" + i);

    long[] counts = engine.readTestCounts();
    assertTrue(counts[0] > 0);
    assertTrue(counts[1] >= counts[0]);
    assertTrue(counts[3] <= counts[1]);
    assertTrue(engine.walkCensus().contains(" read tests over "));
    assertTrue(engine.walkCensus().contains(" queries in buckets of 8 or more"));
  }

  /**
   * A tracked computation's value is kept and returned without recomputing while the query it read
   * keeps its value, and is recomputed once the query is written again (wala/ML#1272).
   */
  @Test
  public void testTrackedValueKeptUntilAReadIsWritten() {
    // The check mode computes a kept value again at every read of it, and this counts the runs.
    assumeFalse(Boolean.getBoolean(WorklistTypeResolver.CHECK_TRACKED_PROPERTY));
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult both = ShapeResult.of(Set.of(List.of(), List.of(new NumericDim(2))));
    Object[] value = {scalar};
    int[] runs = new int[1];
    Supplier<Object> computation =
        () -> {
          runs[0]++;
          return engine.demand("TRACK_K", () -> value[0], true);
        };

    assertEquals(scalar, engine.tracked("TRACK_T", computation, () -> true));
    assertEquals(scalar, engine.tracked("TRACK_T", computation, () -> true));
    assertEquals(1, runs[0]);
    assertEquals(1, engine.trackedHits());

    value[0] = both;
    engine.evaluateForTest("TRACK_K");

    assertEquals(both, engine.tracked("TRACK_T", computation, () -> true));
    assertEquals(2, runs[0]);
    assertEquals(2, engine.trackedComputations());
  }

  /**
   * A kept value returned inside an evaluation replays the reads it stands for, so the evaluating
   * query is evaluated again when a query the value was computed from changes. Without the replay
   * the query would keep the value it read, since nothing would tie it to the changed query.
   */
  @Test
  public void testTrackedHitReplaysItsReads() {
    // The check mode computes a kept value again at every read of it, and this counts the runs.
    assumeFalse(Boolean.getBoolean(WorklistTypeResolver.CHECK_TRACKED_PROPERTY));
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult both = ShapeResult.of(Set.of(List.of(), List.of(new NumericDim(2))));
    Object[] value = {scalar};
    int[] runs = new int[1];
    Supplier<Object> computation =
        () -> {
          runs[0]++;
          return engine.isEvaluating()
              ? engine.read("REPLAY_K", () -> value[0], true)
              : engine.demand("REPLAY_K", () -> value[0], true);
        };
    Supplier<Object> reader = () -> engine.tracked("REPLAY_T", computation, () -> true);
    engine.tracked("REPLAY_T", computation, () -> true);

    assertEquals(scalar, engine.demand("REPLAY_R", reader, true));
    assertEquals("The reader is answered from the kept value.", 1, runs[0]);

    value[0] = both;
    engine.evaluateForTest("REPLAY_K");

    assertEquals(both, engine.demand("REPLAY_R", reader, true));
  }

  /**
   * A tracked computation inside another hands its reads on, also when it is answered from a kept
   * value, so the outer value is recomputed once a query the inner one read is written again.
   */
  @Test
  public void testNestedTrackedValueHandsItsReadsOn() {
    // The check mode computes a kept value again at every read of it, and this counts the runs.
    assumeFalse(Boolean.getBoolean(WorklistTypeResolver.CHECK_TRACKED_PROPERTY));
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult both = ShapeResult.of(Set.of(List.of(), List.of(new NumericDim(2))));
    Object[] value = {scalar};
    int[] runs = new int[1];
    Supplier<Object> inner = () -> engine.demand("NEST_K", () -> value[0], true);
    Supplier<Object> outer =
        () -> {
          runs[0]++;
          return engine.tracked("NEST_INNER", inner, () -> true);
        };
    engine.tracked("NEST_INNER", inner, () -> true);

    assertEquals(scalar, engine.tracked("NEST_OUTER", outer, () -> true));
    assertEquals(scalar, engine.tracked("NEST_OUTER", outer, () -> true));
    assertEquals(1, runs[0]);

    value[0] = both;
    engine.evaluateForTest("NEST_K");

    assertEquals(both, engine.tracked("NEST_OUTER", outer, () -> true));
    assertEquals(2, runs[0]);
  }

  /**
   * A tracked computation's value is not kept when the computation deferred an arm for a settled
   * recomputation, or when its caller's own conditions do not hold after it.
   */
  @Test
  public void testTrackedValueNotKeptWhenDeferredOrRefused() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = new int[2];
    Supplier<Object> deferring =
        () -> {
          runs[0]++;
          engine.requestSettledRecomputation();
          return scalar;
        };
    Supplier<Object> refused =
        () -> {
          runs[1]++;
          return scalar;
        };

    engine.tracked("DEFER_T", deferring, () -> true);
    engine.tracked("DEFER_T", deferring, () -> true);
    engine.tracked("REFUSE_T", refused, () -> false);
    engine.tracked("REFUSE_T", refused, () -> false);

    assertEquals(2, runs[0]);
    assertEquals(2, runs[1]);
    assertEquals(0, engine.trackedHits());
  }

  /**
   * The check mode recomputes each kept value it returns: it passes when the recomputation gives
   * the kept value, and fails on a computation that is not a function of the queries it reads.
   */
  @Test
  public void testTrackedCheckMode() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableTrackedCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    Supplier<Object> stable = () -> engine.demand("CHECKED_K", () -> scalar, true);
    engine.tracked("CHECKED_T", stable, () -> true);
    assertEquals(scalar, engine.tracked("CHECKED_T", stable, () -> true));
    assertEquals(1, engine.checkedTracked());

    int[] runs = new int[1];
    Supplier<Object> counting = () -> ++runs[0];
    engine.tracked("COUNTING_T", counting, () -> true);
    AssertionError difference = null;
    try {
      engine.tracked("COUNTING_T", counting, () -> true);
    } catch (AssertionError e) {
      difference = e;
    }
    assertTrue(difference != null && difference.getMessage().contains("A kept value differs"));
  }

  /**
   * A settlement pass inside a generator guard skips a settled cycle whose inputs have not changed,
   * as a pass outside one does (wala/ML#1293).
   */
  @Test
  public void testGuardedPassSkipsSettledCycle() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    // The check mode recomputes what the pass skips; this test counts recomputations.
    engine.disableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = {0};
    Supplier<Object> settled =
        readingCycle(
            engine,
            "GSKIP_P",
            "GSKIP_Q",
            () -> {
              runs[0]++;
              return scalar;
            });
    assertEquals(scalar, engine.demand("GSKIP_P", settled, true));
    int afterFirstDemand = runs[0];
    assertEquals(0, engine.guardedSettledSkips());

    Supplier<Object> other = readingCycle(engine, "GSKIP_R", "GSKIP_S", () -> scalar);
    assertEquals(
        scalar, TensorGenerator.underGuardForTest(() -> engine.demand("GSKIP_R", other, true)));

    assertEquals(afterFirstDemand, runs[0]);
    assertEquals(2, engine.guardedSettledSkips());
  }

  /**
   * A settled recomputation made inside a generator guard that no guard decided keeps its record,
   * so a later pass skips its query (wala/ML#1293).
   */
  @Test
  public void testGuardedRecomputationKeepsItsRecord() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = {0};
    Supplier<Object> guarded =
        readingCycle(
            engine,
            "GKEEP_P",
            "GKEEP_Q",
            () -> {
              runs[0]++;
              return scalar;
            });
    assertEquals(
        scalar, TensorGenerator.underGuardForTest(() -> engine.demand("GKEEP_P", guarded, true)));
    int afterFirstDemand = runs[0];
    assertEquals(Set.of("GKEEP_Q"), engine.settledReadsOf("GKEEP_P"));

    Supplier<Object> other = readingCycle(engine, "GKEEP_R", "GKEEP_S", () -> scalar);
    assertEquals(scalar, engine.demand("GKEEP_R", other, true));

    assertEquals(afterFirstDemand, runs[0]);
  }

  /**
   * A settled recomputation that a guard decided, made inside a generator guard, leaves no record,
   * so the next pass recomputes its query. Outside every guard the same recomputation keeps its
   * record, as before: the frame it met was its own (wala/ML#1293).
   */
  @Test
  public void testGuardDecidedRecomputationLeavesNoRecord() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.disableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    int[] runs = {0};
    Supplier<Object> decided =
        readingCycle(
            engine,
            "GDEC_P",
            "GDEC_Q",
            () -> {
              runs[0]++;
              TensorGenerator.noteGuardDecision();
              return scalar;
            });
    assertEquals(
        scalar, TensorGenerator.underGuardForTest(() -> engine.demand("GDEC_P", decided, true)));
    int afterFirstDemand = runs[0];
    long decidedInGuard = engine.guardDecidedRecomputations();
    assertTrue(decidedInGuard > 0);
    assertEquals(Set.of(), engine.settledReadsOf("GDEC_P"));
    // Its partner, which no guard decided, keeps its record in the same pass.
    assertEquals(Set.of("GDEC_P"), engine.settledReadsOf("GDEC_Q"));

    Supplier<Object> unguarded =
        readingCycle(
            engine,
            "GDEC_R",
            "GDEC_S",
            () -> {
              TensorGenerator.noteGuardDecision();
              return scalar;
            });
    assertEquals(scalar, engine.demand("GDEC_R", unguarded, true));

    assertEquals(afterFirstDemand + 1, runs[0]);
    // Outside every guard a recomputation is not counted as one a guard decided.
    assertEquals(decidedInGuard, engine.guardDecidedRecomputations());
    assertEquals(Set.of("GDEC_Q"), engine.settledReadsOf("GDEC_P"));
    assertEquals(Set.of("GDEC_S"), engine.settledReadsOf("GDEC_R"));
  }

  /**
   * The check mode inside a generator guard: a check recomputation that a guard decided is not the
   * one the skip stands for, so its value is not compared, and one that no guard decided is
   * compared as outside a guard (wala/ML#1293).
   */
  @Test
  public void testSkipCheckModeInsideGuard() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    engine.enableSkipCheck();

    ShapeResult scalar = ShapeResult.of(Set.of(List.of()));
    ShapeResult vector = ShapeResult.of(Set.of(List.of(new NumericDim(3))));
    // What the transfer does once the cycle has settled: 1 has a guard decide another value, and 2
    // returns another value with no guard deciding it.
    int[] mode = {0};
    Supplier<Object> settled =
        readingCycle(
            engine,
            "GCHECK_P",
            "GCHECK_Q",
            () -> {
              if (mode[0] == 1) TensorGenerator.noteGuardDecision();
              return mode[0] == 0 ? scalar : vector;
            });
    assertEquals(scalar, engine.demand("GCHECK_P", settled, true));

    mode[0] = 1;
    Supplier<Object> other = readingCycle(engine, "GCHECK_R", "GCHECK_S", () -> scalar);
    TensorGenerator.underGuardForTest(() -> engine.demand("GCHECK_R", other, true));
    assertEquals(1, engine.guardDecidedCheckedSkips());
    assertEquals(scalar, engine.demand("GCHECK_P", settled, true));

    mode[0] = 2;
    Supplier<Object> last = readingCycle(engine, "GCHECK_T", "GCHECK_U", () -> scalar);
    AssertionError difference = null;
    try {
      TensorGenerator.underGuardForTest(() -> engine.demand("GCHECK_T", last, true));
    } catch (AssertionError e) {
      difference = e;
    }
    assertTrue(
        difference != null
            && difference.getMessage().startsWith("A skipped settled recomputation differs"));
    assertEquals(1, engine.guardDecidedCheckedSkips());
  }

  /**
   * The walk census counts a settlement sweep inside a generator guard as a restricted walk, and as
   * a guarded whole-graph walk only with the settlement skip off (wala/ML#1293).
   */
  @Test
  public void testWalkCensusCountsGuardedSweep() {
    WorklistTypeResolver engine = WorklistTypeResolver.install(null);
    TensorGenerator.underGuardForTest(
        () -> {
          demandPureCycle(engine, "GWALK_A", "GWALK_B");
          return null;
        });
    // The counts of the same cycle demanded outside a guard.
    assertArrayEquals(new long[] {2, 0, 4, 2, 2, 1, 0, 0, 2, 2, 2, 1}, engine.walkCounts());
    WorklistTypeResolver.uninstall(null, true);

    WorklistTypeResolver whole = WorklistTypeResolver.install(null);
    whole.disableSkip();
    TensorGenerator.underGuardForTest(
        () -> {
          demandPureCycle(whole, "GWHOLE_A", "GWHOLE_B");
          return null;
        });
    assertArrayEquals(new long[] {2, 0, 4, 2, 2, 1, 1, 1, 2, 0, 2, 1}, whole.walkCounts());
  }

  /**
   * Makes a cycle of two queries that read each other, of which the second returns a scalar.
   *
   * @param engine The engine.
   * @param first The first query's key.
   * @param second The second query's key.
   * @param value What the first query's transfer returns after its read.
   * @return The first query's transfer.
   */
  private static Supplier<Object> readingCycle(
      WorklistTypeResolver engine, Object first, Object second, Supplier<Object> value) {
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          engine.read(second, transfers.get(1), true);
          return value.get();
        });
    transfers.add(
        () -> {
          engine.read(first, transfers.get(0), true);
          return ShapeResult.of(Set.of(List.of()));
        });
    return transfers.get(0);
  }

  /**
   * Demands a fresh two-query cycle, so the fixpoint runs past its early return and scans for stale
   * readers.
   *
   * @param engine The engine.
   * @param first The first query's key.
   * @param second The second query's key.
   */
  private static void demandPureCycle(WorklistTypeResolver engine, Object first, Object second) {
    List<Supplier<Object>> transfers = new ArrayList<>();
    transfers.add(
        () -> {
          engine.read(second, transfers.get(1), true);
          return ShapeResult.bottom();
        });
    transfers.add(
        () -> {
          engine.read(first, transfers.get(0), true);
          return ShapeResult.bottom();
        });
    engine.demand(first, transfers.get(0), true);
  }

  private static void restore(String property, String value) {
    if (value == null) System.clearProperty(property);
    else System.setProperty(property, value);
  }
}
