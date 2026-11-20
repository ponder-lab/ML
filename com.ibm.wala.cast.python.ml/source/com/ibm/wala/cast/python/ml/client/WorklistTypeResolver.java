package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ipa.callgraph.TrampolineReceiverContextSelector.AnchoredCallerSiteContext;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.Context;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.ReturnValueKey;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContext;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallerContext;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallerSiteContext;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.BinaryOperator;
import java.util.function.BooleanSupplier;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * The Phase 2 worklist engine of the wala/ML#365 design: resolves shape and dtype queries by
 * chaotic iteration over the explicit query-dependency graph instead of guarded demand-driven
 * recursion, making the results independent of evaluation order.
 *
 * <p>A query is any keyed computation the memo layers already know: a {@code (node, vn, exact)}
 * value read, a whole-generator evaluation, or a producer delegation. When the engine evaluates a
 * query, its existing compute body runs unchanged as the transfer function; the memo layers divert
 * every nested query read to {@link #read}, which records the dependency edge and evaluates or
 * consults the state table rather than recursing unboundedly. Values ascend a lattice — {@link
 * ShapeResult} ordered by member inclusion with the unknown mark as a second component, dtype sets
 * by inclusion with an {@code UNKNOWN} member marking a partial union (wala/ML#862) and the legacy
 * null-⊤ preserved through a sentinel so callers' null-check fallback arms behave as under
 * recursion — via joins at every update, so iteration reaches a fixpoint; scheduling follows the
 * read-version staleness criterion (wala/ML#758): each reader records the version of every value it
 * consumes, a grown value re-enqueues exactly the dependents whose latest evaluation consumed it, a
 * solve-level sweep re-enqueues any reader whose recorded versions still lag before the fixpoint is
 * declared, and the per-key evaluation-count valve pins any oscillator at its current value.
 *
 * <p>Cycles need no guards: reads never recurse. A dependency SCC with no external base stabilizes
 * at the iteration bottom, which would read as "not a tensor," so after each stabilization the
 * engine promotes bottom-valued members of nontrivial SCCs to the unknown-marked element and
 * re-solves, preserving the sound answer for genuinely baseless loop-carried values (the design's
 * promotion step). Widenings bound divergence: a member-set cardinality cap and a rank cap, both
 * logged when they fire.
 *
 * <p>The default resolution engine since Phase 3, installed per analysis run by {@link
 * PythonTensorAnalysisEngine#performAnalysis}; the historical round-based resolution is retired.
 */
final class WorklistTypeResolver {

  private static final Logger LOGGER = Logger.getLogger(WorklistTypeResolver.class.getName());

  /** Member-set cardinality beyond which a value widens to unknown-marked. */
  private static final int MEMBER_WIDENING_CAP = 16;

  /** Evaluations of one query beyond which it pins at its current value. */
  private static final int EVALUATION_CAP = 100;

  /**
   * The depth of nested inline evaluations at which a first-sighted read unwinds them (see {@link
   * DepthUnwind}). A whole-program analysis can have a dependency chain longer than a default
   * thread stack holds as nested evaluations; below the bound, scheduling is unchanged.
   */
  private static final int INLINE_DEPTH_BOUND = 256;

  /**
   * Sentinel for the legacy null dtype result (⊤). The state table cannot hold {@code null}, but
   * legacy callers null-check dtype results to run fallback arms, so {@link #read} and {@link
   * #demand} translate this back to {@code null} at every boundary crossing.
   */
  private static final Object NULL_DTYPE =
      new Object() {
        @Override
        public String toString() {
          return "null dtype (⊤)";
        }
      };

  /** Rank beyond which a member widens away. */
  private static final int RANK_WIDENING_CAP = 12;

  /**
   * The recomputations of one query, each holding a shape that no query of its component holds
   * then, at which {@link #iterate} widens the query. A query that gathers the shapes its component
   * holds never gets there, and one whose transfer makes a new shape from each shape it is given
   * gets there in as many steps.
   */
  private static final int GROWTH_WIDENING_LIMIT = 3;

  /** The per-builder active engine; non-null only while an analysis run's resolution is active. */
  private static final Map<PropagationCallGraphBuilder, WorklistTypeResolver> ACTIVE =
      Collections.synchronizedMap(new WeakHashMap<>());

  /** Query values; absent means the iteration bottom for the query's kind. */
  private final Map<Object, Object> state = HashMapFactory.make();

  /** Transfer functions, registered on first encounter of each query. */
  private final Map<Object, Supplier<Object>> transfers = HashMapFactory.make();

  /** Whether each query is {@link ShapeResult}-valued ({@code true}) or dtype-valued. */
  private final Map<Object, Boolean> shapeKinds = HashMapFactory.make();

  /** Reverse dependency edges: query → the queries whose evaluations read it. */
  private final Map<Object, Set<Object>> dependents = HashMapFactory.make();

  /** Forward edges, for the SCC promotion pass. */
  private final Map<Object, Set<Object>> reads = HashMapFactory.make();

  /**
   * Each {@link #reads} key's insertion position, so a region-restricted {@link #tarjan(Set)} roots
   * its search in the whole graph's order without scanning every key (wala/ML#1176).
   */
  private final Map<Object, Integer> readsOrder = HashMapFactory.make();

  /**
   * A check a bound evaluation must pass at convergence (wala/ML#1071), with the queries whose
   * evaluations rely on it.
   *
   * @param check Whether the converged dependency graph shows the binding applied as required.
   * @param roots The queries to re-evaluate when the check fails.
   */
  private record BindingCheck(Predicate<WorklistTypeResolver> check, Set<Object> roots) {}

  /**
   * The registered binding checks, keyed by binding, in registration order so validation is
   * deterministic.
   */
  private final Map<Object, BindingCheck> bindingChecks = new LinkedHashMap<>();

  /** The bindings whose check failed: their evaluations decline (wala/ML#1071). */
  private final Set<Object> declinedBindings = HashSetFactory.make();

  /** Whether a check was registered since the last validation. */
  private boolean bindingChecksPending;

  /** Per-query value versions: the number of state changes; absent means 0 (never written). */
  private final Map<Object, Integer> versions = HashMapFactory.make();

  /**
   * Per-reader consumed versions: for each query, the {@link #versions} entry of every value its
   * latest evaluation read (wala/ML#758's read-version staleness criterion). Cleared at the start
   * of each evaluation, so an entry always reflects the run that produced the reader's current
   * contribution; a reader is stale iff any recorded version lags the value's current one.
   */
  private final Map<Object, Map<Object, Integer>> readVersions = HashMapFactory.make();

  /**
   * Queries that ever consumed a value that subsequently grew. Their settled values can carry join
   * history from the stale consumption even when re-evaluation caught them up (joins never remove
   * members), so {@link #canonicalize} targets them for replacement regardless of its cyclic-SCC
   * and changed-read criteria (the residual artifact class of wala/ML#758).
   */
  private final Set<Object> staleReaders = HashSetFactory.make();

  /**
   * The readers {@link #enqueueStaleReaders} must check (wala/ML#1176): every reader of a value
   * whose version moved since the last scan, every reader whose records {@link #checkSkip} restored
   * since then, and every reader that scan skipped or found stale. A record is written at its
   * value's current version, so it lags only once that value moves after it, which marks the
   * reader; a reader outside the set was therefore not stale at the last scan and is not stale now,
   * and a scan need not visit every record of every reader to find that nothing is stale.
   */
  private final Set<Object> staleCandidates = HashSetFactory.make();

  /** How many staleness scans found a stale reader. */
  private int staleScanHits;

  /**
   * Queries whose transfers deferred a finality-dependent arm (wala/ML#717's contract seed) during
   * a possibly-interim evaluation via {@link #requestSettledRecomputation}. {@link #canonicalize}
   * recomputes them against the settled state, where {@link #isSettling} holds and the arm can fire
   * knowing every read is final.
   */
  private final Set<Object> settlementRequested = HashSetFactory.make();

  /** Whether the post-fixpoint canonicalization pass is running; see {@link #isSettling}. */
  private boolean settling;

  /**
   * Queries whose recomputation in {@link #canonicalize} deferred under a re-entered {@code len}
   * fold (wala/ML#1021). Such a recomputation answers ⊥, which never replaces a non-⊥ settled
   * value; a query among these whose settled value is itself ⊥ is a member of a cycle the engine
   * cannot see (the fold runs on the Java stack, not as a query) that no evaluation ever gave a
   * base, and the pass gives it the unknown-marked element, the pure-cycle policy of {@link
   * #promotePureCycles}, and its readers are recomputed against the settled state by the pass's own
   * sweep: the value has tensor evidence (the gates defer only on a non-empty points-to union), so
   * ⊥ there is a drop and ⊤ the sound reading.
   */
  private final Set<Object> reentryDeferredAtSettlement = HashSetFactory.make();

  /**
   * Records that the query being recomputed in {@link #canonicalize} deferred under a re-entered
   * {@code len} fold; a no-op outside the pass.
   */
  void noteReentryDeferralAtSettlement() {
    Object query = this.evaluating.peek();
    if (this.settling && query != null) this.reentryDeferredAtSettlement.add(query);
    this.deferTracked();
  }

  private final Deque<Object> worklist = new ArrayDeque<>();
  private final Set<Object> enqueued = HashSetFactory.make();
  private final Map<Object, Integer> evaluationCounts = HashMapFactory.make();

  /**
   * How many query evaluations threw and were floored to unknown (wala/ML#925): a generator that
   * throws is not a crash and not a gap, and this is the only count that separates the two.
   */
  private int caughtQueryExceptions = 0;

  /** How many reads unwound the evaluations in progress at {@link #INLINE_DEPTH_BOUND}. */
  private int depthUnwinds = 0;

  /**
   * The cycle-order perturbation source (wala/ML#756), or {@code null} when the knob is off. The
   * hybrid inline scheduling fixes the acyclic region's evaluation order, so the orders the engine
   * does not determine are the ones the worklist and the dependent re-enqueues impose on
   * cycle-affected keys; both are identity-hash-seeded and vary with JVM run history, which is
   * exactly what an order-invariance test cannot rerun on demand. Seeding this source (the {@code
   * ariadne.typeResolution.shuffleCycles} system property or the {@code ARIADNE_SHUFFLE_CYCLES}
   * environment variable, holding a {@code long} seed) perturbs both orders deterministically; the
   * same setting also shuffles the demand-root order in {@link
   * PythonTensorAnalysisEngine#performAnalysis}, the third undetermined order and the one whose
   * reversal alone the reverse-seeds property exercises.
   */
  private final Random cycleShuffle = createCycleShuffle();

  /** The stack of currently-evaluating queries; reads record edges against the top. */
  private final Deque<Object> evaluating = new ArrayDeque<>();

  /**
   * The logical clock of state writes, for the settlement pass's skip of a query whose inputs have
   * not changed since its last settled recomputation. See {@link #cyclicCause}.
   */
  private long clock;

  /** Per query, the clock of the last write of its state, by any writer. */
  private final Map<Object, Long> stamps = HashMapFactory.make();

  /**
   * Per query, the clock of the last write of its state other than the settlement pass's write-back
   * of its own recomputation: an evaluation or a promotion.
   */
  private final Map<Object, Long> selfStamps = HashMapFactory.make();

  /**
   * Per query, the clock at the entry of its last settled recomputation that completed normally,
   * did not defer under a re-entered fold and, where it ran inside a generator guard, had no guard
   * decide what it did; absent if there is none, which makes the next pass recompute the query.
   */
  private final Map<Object, Long> settledAt = HashMapFactory.make();

  /**
   * Per query, the queries its last settled recomputation read. A fixpoint evaluation replaces
   * {@link #readVersions} with the reads of its non-final arms, so the settled reads are kept
   * apart.
   */
  private final Map<Object, Set<Object>> settledReads = HashMapFactory.make();

  /**
   * The queries whose last settled recomputation read their own value. {@link #read} records no
   * edge from a query to itself, so such a read is noted here; the query's own write-back then
   * counts as a change of an input.
   */
  private final Set<Object> settledSelfReaders = HashSetFactory.make();

  /** Whether a query read its own value during the settled recomputation in progress. */
  private boolean selfReadInRecomputation;

  /** Whether the settlement pass in progress was entered with a generator guard in progress. */
  private boolean guardedSettlement;

  /** Whether the settled recomputation that just ran caught an exception and kept the value. */
  private boolean recomputationThrew;

  /**
   * Whether a generator guard decided what the settled recomputation that just ran did; see {@link
   * TensorGenerator#guardDecisions}.
   */
  private boolean recomputationGuardDecided;

  /** How many check recomputations inside a generator guard a guard decided; not compared. */
  private long guardDecidedCheckedSkips;

  /** Whether each skipped query is also recomputed and checked; see {@link #enableSkipCheck}. */
  private boolean checkSkips = Boolean.getBoolean(CHECK_SKIPS_PROPERTY);

  /** The system property that turns on {@link #checkSkips}. */
  static final String CHECK_SKIPS_PROPERTY = "ariadne.typeResolution.checkSettlementSkips";

  /**
   * The system property that turns the settlement pass's skip off when {@code false}, so every
   * cyclic query is recomputed by every pass, as before the skip.
   */
  static final String SETTLEMENT_SKIP_PROPERTY = "ariadne.typeResolution.settlementSkip";

  /** Whether the settlement pass may skip a query; see {@link #SETTLEMENT_SKIP_PROPERTY}. */
  private boolean skipEnabled = !"false".equals(System.getProperty(SETTLEMENT_SKIP_PROPERTY));

  /**
   * The system property that turns the settlement pass's iteration off when {@code false}, so a
   * pass recomputes each query of a cyclic component at most once, as before the iteration.
   */
  static final String SETTLEMENT_ITERATION_PROPERTY = "ariadne.typeResolution.settlementIteration";

  /** Whether a settlement pass iterates a cyclic component; see {@link #iterate}. */
  private boolean iterationEnabled =
      !"false".equals(System.getProperty(SETTLEMENT_ITERATION_PROPERTY));

  /**
   * The recomputations {@link #iterate} may make on one component in one pass before it joins: this
   * many for each of the component's queries, and {@link #iterationBudgetBase} more.
   */
  private long iterationBudgetFactor = 64;

  /** See {@link #iterationBudgetFactor}. */
  private long iterationBudgetBase = 1024;

  /**
   * The queries whose settled recomputations are joined into their settled values and no longer
   * replace them: the members of each component {@link #iterate} could not bring to a fixpoint by
   * replacement. Monotone, so the queries cannot move back and forth in any later pass either. A
   * pass inside a generator guard writes none of them ({@link #step}), since a query that only
   * grows cannot give back a value a guard's cut put in it. A joined shape holds no member another
   * of its members covers ({@link #joinCovering}).
   */
  private final Set<Object> joinSettled = HashSetFactory.make();

  /**
   * The queries of {@link #joinSettled} whose join passed a widening cap. Such a query holds the
   * unknown-marked element and no settlement pass writes it again: a join that restarts from the
   * widened value would pass the cap again, with no end.
   */
  private final Set<Object> pinnedAtSettlement = HashSetFactory.make();

  /**
   * The queries of {@link #joinSettled} that {@link #iterate} widened: each kept making shapes no
   * query of its component held. Such a query is written the widening of the recomputations that
   * made those shapes, and from then on its settled value is widened by each recomputation ({@link
   * ShapeResult#widening}) and not joined with it. It is the point where its cycle's growth stops,
   * and the component's other queries go on being replaced.
   */
  private final Set<Object> widenedAtSettlement = HashSetFactory.make();

  /**
   * The shapes a widening made that neither the widened value nor the recomputation held: each
   * stands for the shapes its query kept making. A value that holds one of them, a settled
   * recomputation or a first evaluation, is kept without the members it covers ({@link #absorb}).
   * Those members are what the growth left behind, and a cycle that hands a value on as it is would
   * keep them for good, since each of its queries reads them from the one before.
   */
  private final Set<List<Dimension<?>>> generalized = HashSetFactory.make();

  /**
   * The iteration census: the components {@link #iterate} iterated, the steps it made beyond each
   * pass's own, the largest component it iterated, the orbits it found, the values it set to an
   * orbit's meet and to an orbit's join, the components it left to join, the queries it widened,
   * the queries first sighted during an iterated component's steps, and the queries the check mode
   * recomputed at a fixpoint. A component is counted as left to join when it puts a query in {@link
   * #joinSettled}, so one that goes past its budget again in a later pass is not counted again. A
   * first-sighted query is not among the readers {@link #iterate} recomputes, and one that reads
   * the component is on its cycle, so the count of those is of the queries a fixpoint left out
   * ({@link #countSighted}). The last two count the recomputations a pass inside a generator guard
   * made of a query of {@link #joinSettled}, none of which it wrote ({@link #step}), and those of
   * them that differ from the settled value: the query holds that value, and not what its transfer
   * returns, until a later pass recomputes it.
   */
  private long iteratedComponents,
      iterationSteps,
      largestIteratedComponent,
      orbits,
      orbitMeets,
      orbitJoins,
      joinedComponents,
      widenedQueries,
      sightedWhileIterating,
      sightedOnCycle,
      checkedFixpointQueries,
      guardHeld,
      guardHeldDiffering;

  /**
   * The queries first sighted during the steps of the component a sweep is on, in the order of
   * their sighting, or {@code null} where the sweep is on no component it may iterate.
   */
  private List<Object> sightedInSteps;

  /**
   * The system property that turns the restriction off when {@code false}, so every settlement walk
   * covers the whole query graph, as before.
   */
  static final String RESTRICT_SETTLEMENT_PROPERTY = "ariadne.typeResolution.restrictSettlement";

  /** Whether the settlement walks are restricted; see {@link #RESTRICT_SETTLEMENT_PROPERTY}. */
  private final boolean restrictSettlement =
      !"false".equals(System.getProperty(RESTRICT_SETTLEMENT_PROPERTY));

  /**
   * The queries for which {@link #cyclicCause} may give a cause: a superset, kept by the writes
   * that can give one (registration, a stamp of the query or of a settled read, and every settled
   * recomputation), and shrunk only when it is observed to give none.
   */
  private final Set<Object> settleDirty = HashSetFactory.make();

  /** The queries some Tarjan pass found in a cyclic component; monotone, as edges never go. */
  private final Set<Object> everCyclic = HashSetFactory.make();

  /** {@link #settleDirty} restricted to {@link #everCyclic}. */
  private final Set<Object> dirtyCyclic = HashSetFactory.make();

  /** The readers that gained an edge since the last full settlement sweep began. */
  private final Set<Object> newEdgeReaders = HashSetFactory.make();

  /** The inverse of {@link #settledReads}, over-approximate (entries are never retracted). */
  private final Map<Object, Set<Object>> settledReaders = HashMapFactory.make();

  /**
   * The region of the restricted sweep in progress under the check mode, or {@code null}; an
   * evaluation of a graph query outside it is a write the restriction does not see.
   */
  private Set<Object> checkedSweepRegion;

  /** The shape queries whose state is absent or bottom: a superset of the promotion candidates. */
  private final Set<Object> shapeBottoms = HashSetFactory.make();

  /**
   * The shape queries that entered {@link #shapeBottoms} since the last promotion pass began: a
   * query whose state became absent or bottom can be a promotion candidate on a cycle it was
   * already on.
   */
  private final Set<Object> newShapeBottoms = HashSetFactory.make();

  /**
   * The readers that gained an edge since the last promotion pass began. Unlike {@link
   * #newEdgeReaders}, which a full settlement sweep clears, only {@link #promotePureCycles} clears
   * these, so a promotion pass sees every edge added since the previous one.
   */
  private final Set<Object> promotionEdgeReaders = HashSetFactory.make();

  /**
   * The settlement census: passes, guarded passes, recomputations, the recomputations made by
   * guarded passes and those of them a guard decided, skips, the skips made by guarded passes and
   * checked skips.
   */
  private long settlementPasses,
      guardedSettlementPasses,
      settledRecomputations,
      guardedSettledRecomputations,
      guardDecidedRecomputations,
      settledSkips,
      guardedSettledSkips,
      checkedSkips;

  /** Why a settlement sweep recomputes a query, for the settlement census (wala/ML#1293). */
  enum RecomputationCause {
    /** The query consumed a value that grew after it was consumed. */
    STALE("stale"),

    /** The query's transfer withheld an arm that depends on the settled state. */
    REQUESTED("requested"),

    /** The query read a query the pass in progress replaced or promoted. */
    CHANGED_READ("reading a query the pass changed"),

    /** The query is on a cycle and the skip is off. */
    SKIP_OFF("on a cycle with the skip off"),

    /** The query is on a cycle and no settled recomputation of it stands. */
    NO_RECORD("on a cycle with no settled recomputation standing"),

    /** The query is on a cycle and was written after its last settled recomputation began. */
    OWN_WRITE("on a cycle and written since"),

    /**
     * The query is on a cycle, and an evaluation or a promotion wrote a read of its last settled
     * recomputation after that recomputation began.
     */
    READ_EVALUATED("on a cycle with a read an evaluation or a promotion wrote since"),

    /**
     * The query is on a cycle, and nothing but the write-backs of settlement passes wrote the reads
     * of its last settled recomputation after that recomputation began.
     */
    READ_SETTLED("on a cycle with reads only settlement passes wrote since"),

    /**
     * The query is on a cycle the pass is iterating to a fixpoint ({@link #iterate}): it reads a
     * query the previous step on that cycle wrote, or that step gave it a value that is not its own
     * recomputation's.
     */
    ITERATED("on a cycle the pass is iterating");

    /** The cause as the census line words it. */
    private final String description;

    RecomputationCause(String description) {
      this.description = description;
    }
  }

  /** What a settled recomputation did to the query's settled value, for the settlement census. */
  enum RecomputationOutcome {
    /** The recomputation returned the settled value. */
    SAME,

    /** The recomputation's value replaced the settled value. */
    REPLACED,

    /**
     * The settled value stays over a recomputation that differs from it: one that returned ⊥ over a
     * settled value that is not ⊥, or one a pass inside a generator guard made of a query of {@link
     * #joinSettled}. The census counts the second apart ({@link #guardHeldDiffering}).
     */
    KEPT
  }

  /** The settled recomputations by {@link RecomputationCause} and {@link RecomputationOutcome}. */
  private final long[][] recomputationsByCause =
      new long[RecomputationCause.values().length][RecomputationOutcome.values().length];

  /**
   * The replacements, by {@link RecomputationCause}, that gave a query back the value its previous
   * replacement by a settlement pass had replaced: a value moving back and forth between passes.
   */
  private final long[] restoredByCause = new long[RecomputationCause.values().length];

  /**
   * The hash code of the value each query's last replacement by a settlement pass replaced, or
   * {@code null} when the census does not count restored values. The map holds a hash code and not
   * the value, so it keeps no replaced value alive, and a restored count is over by any two
   * different values with one hash code. It has an entry for every query a pass ever replaced, so
   * it is kept only when the census line is logged.
   */
  private Map<Object, Integer> replacedBySettlement =
      SETTLEMENT_CENSUS.isLoggable(Level.FINE) ? HashMapFactory.make() : null;

  /**
   * The settlement passes that replaced no value, and the time the sweeps' recomputations took. No
   * sweep runs inside another's recomputations, so no time is counted twice: {@link
   * #recomputeSettled} runs a transfer as an evaluation, a generator's query inside an evaluation
   * is a {@link #read}, and only {@link #demand} runs a solve.
   */
  private long unchangedSettlementPasses, sweepRecomputeNanos;

  /**
   * The walk census (wala/ML#1176): for the pure-cycle promotion and for the settlement sweep, the
   * walks, those over the whole graph, the queries the graph held at each walk, the queries their
   * reader closures held, the queries their condensations emitted and the time the closure and the
   * condensation took; the time of the staleness scans; and the outermost solves and their time.
   * Neither a walk nor a scan evaluates a transfer, so no nested solve is inside the times they
   * record. The condensed queries are the restricted pass's in either mode, but in the check mode
   * of {@link #CHECK_SKIPS_PROPERTY} a walk's time also holds the whole-graph condensation it is
   * checked against.
   */
  private long promotionWalks,
      promotionWholeWalks,
      promotionGraphQueries,
      promotionClosureQueries,
      promotionCondensedQueries,
      promotionWalkNanos,
      sweepWalks,
      sweepWholeWalks,
      sweepGuardedWholeWalks,
      sweepGraphQueries,
      sweepClosureQueries,
      sweepCondensedQueries,
      sweepWalkNanos,
      staleScanNanos,
      outerSolves,
      outerSolveNanos;

  /**
   * The read-set census of the settlement sweep's target selection (wala/ML#1176): the queries it
   * tested for having read a changed query, the sizes of their read sets and of the changed set
   * summed over those tests, the containment probes the tests made and the time the selection took.
   * Besides the read tests, the selection's time holds the settled-input checks of cyclic queries.
   * It evaluates no transfer, so no nested solve is inside it, except in the check mode of {@link
   * #CHECK_SKIPS_PROPERTY}, where it also holds the recomputations that check each skip.
   */
  private long sweepReadTests, sweepReadsSum, sweepChangedSum, sweepReadProbes, sweepSelectNanos;

  /** How many solves are running, so only the outermost one is timed. */
  private int solveDepth;

  /** When the walk census was last logged, so a long analysis logs it as it goes. */
  private long walkCensusLoggedAt = System.nanoTime();

  /** How long an analysis runs between interim walk census lines. */
  private static final long WALK_CENSUS_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(5);

  /** The logger of the settlement census line, apart from the engine's own. */
  private static final Logger SETTLEMENT_CENSUS =
      Logger.getLogger(WorklistTypeResolver.class.getName() + ".settlement");

  /** Membership view of {@link #evaluating}, the inline-recursion cycle break. */
  private final Set<Object> inStack = HashSetFactory.make();

  /**
   * The queries held in the pending stacks of the {@link #unwinding} frames, with their
   * multiplicities. They are the evaluations an unbounded stack would still have in progress, so a
   * read of one is a cycle read, like a read of a query in {@link #inStack}.
   */
  private final Map<Object, Integer> unwindPending = HashMapFactory.make();

  /** Queries whose first (inline) evaluation has completed. */
  private final Set<Object> evaluated = HashSetFactory.make();

  /** Whether the post-settlement replay diagnostic is running; see {@link #replaySettled}. */
  private boolean replaying;

  /**
   * Whether an on-stack read (a cycle) occurred since the last promotion pass. The SCC promotion
   * needs a Tarjan pass over every recorded edge, and each key's hash recursively walks its calling
   * context (WALA computes those uncached), so the pass runs only when a cycle can actually exist:
   * without one, every value came from a single inline evaluation and promotion has nothing to do.
   */
  private boolean cycleSeen;

  private WorklistTypeResolver() {}

  /**
   * Returns the builder's active engine, or {@code null} outside an analysis run's resolution.
   *
   * @param builder The builder whose engine to return.
   * @return The active engine or {@code null}.
   */
  static WorklistTypeResolver active(PropagationCallGraphBuilder builder) {
    return ACTIVE.get(builder);
  }

  /**
   * Installs an engine for the builder's resolution run.
   *
   * @param builder The builder to resolve for.
   * @return The installed engine.
   */
  static WorklistTypeResolver install(PropagationCallGraphBuilder builder) {
    WorklistTypeResolver engine = new WorklistTypeResolver();
    ACTIVE.put(builder, engine);
    return engine;
  }

  /**
   * Uninstalls the builder's engine and returns its end-of-analysis census (the successor of the
   * retired wala/ML#591 recorder's summary; see wala/ML#727). Besides the map sizes, the census
   * counts the queries that sit in a nontrivial strongly connected component of the final
   * query-dependency graph, and among them the slice-result generator queries: one Tarjan pass over
   * the settled graph, after the fixpoint and outside it, reading the graph without writing state.
   * A slice query in a cycle is the engine's view of a loop-carried slice, whether the loop is
   * written directly or carried through a helper (wala/ML#916).
   *
   * @param builder The builder whose engine to remove.
   * @param complete Whether the analysis ran to its end; carried into the census so a reader can
   *     tell the counts of a partial solve from a result.
   * @return The census, or {@code null} when no engine was installed for the builder.
   */
  static PythonTensorAnalysisEngine.ResolverCensus uninstall(
      PropagationCallGraphBuilder builder, boolean complete) {
    WorklistTypeResolver engine = ACTIVE.remove(builder);
    if (engine == null) return null;
    SETTLEMENT_CENSUS.fine(() -> "Settlement census: " + engine.settlementCensus() + ".");
    SETTLEMENT_CENSUS.fine(() -> "Walk census: " + engine.walkCensus() + ".");
    SETTLEMENT_CENSUS.fine(
        () ->
            "Tracked values: "
                + engine.trackedComputations
                + " computed, "
                + engine.trackedHits
                + " answered from a kept value ("
                + engine.checkedTracked
                + " checked).");
    int cyclicQueries = 0;
    Map<Object, String> cyclicSliceGenerators = HashMapFactory.make();
    for (List<Object> scc : engine.tarjan()) {
      boolean cyclic =
          scc.size() > 1
              || engine.reads.getOrDefault(scc.get(0), Collections.emptySet()).contains(scc.get(0));
      if (!cyclic) continue;
      String composition = null;
      for (Object key : scc) {
        cyclicQueries++;
        Object generator = sliceGeneratorOf(key);
        if (generator == null || cyclicSliceGenerators.containsKey(generator)) continue;
        if (composition == null) composition = describeComponent(scc);
        cyclicSliceGenerators.put(generator, composition);
      }
    }
    int cyclicSliceQueries = cyclicSliceGenerators.size();
    List<String> descriptions = new ArrayList<>();
    for (Map.Entry<Object, String> generator : cyclicSliceGenerators.entrySet())
      descriptions.add(describeGenerator(generator.getKey()) + " | " + generator.getValue());
    Collections.sort(descriptions);
    // Every slice generator the resolver evaluated, cyclic or not, with what its evaluations read:
    // the denominator that makes a zero above a reading rather than an absence, and the record of
    // where a query chain stops for a site that was expected to cycle and did not.
    Set<Object> sliceGenerators = HashSetFactory.make();
    for (Object key : engine.state.keySet()) {
      Object generator = sliceGeneratorOf(key);
      if (generator != null) sliceGenerators.add(generator);
    }
    List<String> evaluated = new ArrayList<>();
    for (Object generator : sliceGenerators) {
      StringBuilder line = new StringBuilder(describeGenerator(generator));
      for (String tag : new String[] {"shape-eval", "dtype-eval"}) {
        Object evaluation = Pair.make(tag, generator);
        Set<Object> reads = engine.reads.get(evaluation);
        if (reads != null)
          line.append(" | ")
              .append(tag)
              .append(" reads ")
              .append(describeComponent(new ArrayList<>(reads)));
        Set<Object> dependents = engine.dependents.get(evaluation);
        if (dependents != null)
          line.append(" | ")
              .append(tag)
              .append(" read by ")
              .append(describeComponent(new ArrayList<>(dependents)));
      }
      evaluated.add(line.toString());
    }
    Collections.sort(evaluated);
    PythonTensorAnalysisEngine.ResolverCensus census =
        new PythonTensorAnalysisEngine.ResolverCensus(
            complete,
            engine.state.size(),
            engine.reads.values().stream().mapToInt(Set::size).sum(),
            engine.versions.values().stream().mapToInt(Integer::intValue).sum(),
            cyclicQueries,
            cyclicSliceQueries,
            Collections.unmodifiableList(descriptions),
            sliceGenerators.size(),
            Collections.unmodifiableList(evaluated),
            engine.caughtQueryExceptions);
    LOGGER.fine(
        () ->
            "Query census"
                + (census.complete() ? ": " : " (analysis did not complete): ")
                + census.queries()
                + " queries, "
                + census.dependencyEdges()
                + " dependency edges, "
                + census.valueGrowths()
                + " value growths, "
                + census.cyclicQueries()
                + " in cycles, "
                + census.cyclicSliceQueries()
                + " slice generators in cycles, "
                + census.caughtQueryExceptions()
                + " query evaluations floored by a caught exception, "
                + engine.depthUnwinds
                + " unwinds at the inline depth bound.");
    return census;
  }

  /**
   * Describes a strongly connected component by its size and the kinds of query in it, so a reader
   * can tell a cycle that runs through a value read of the receiver from one made only of a
   * generator's own evaluations and delegations (which would be an artefact of the counter rather
   * than a dependency in the program).
   *
   * @param scc The component's keys.
   * @return {@code "scc=N: k1 x kind1, k2 x kind2, ..."} with kinds sorted.
   */
  private static String describeComponent(List<Object> scc) {
    Map<String, Integer> kinds = new java.util.TreeMap<>();
    for (Object key : scc) {
      String kind;
      if (key instanceof Pair && ((Pair<?, ?>) key).fst instanceof String) {
        kind = (String) ((Pair<?, ?>) key).fst;
        Object generator = sliceGeneratorOf(key);
        if (generator != null) kind = kind + "(slice)";
        else if (((Pair<?, ?>) key).snd instanceof Pair
            && ((Pair<?, ?>) ((Pair<?, ?>) key).snd).fst instanceof Class)
          kind =
              kind
                  + "("
                  + ((Class<?>) ((Pair<?, ?>) ((Pair<?, ?>) key).snd).fst).getSimpleName()
                  + ")";
      } else if (key instanceof Pair) kind = "dtype-read";
      else kind = key.getClass().getSimpleName();
      kinds.merge(kind, 1, Integer::sum);
    }
    StringBuilder description = new StringBuilder("scc=").append(scc.size()).append(':');
    for (Map.Entry<String, Integer> kind : kinds.entrySet())
      description
          .append(' ')
          .append(kind.getValue())
          .append(" x ")
          .append(kind.getKey())
          .append(',');
    return description.substring(0, description.length() - 1);
  }

  /**
   * Describes a generator by its anchor so a census reader can find the program site: the anchoring
   * node's method, the value number, and the source file and line of the value's defining
   * instruction where the method carries positions. Falls back to the anchor's own rendering.
   *
   * @param evaluationKey A generator evaluation key, {@code Pair(class, Pair(anchor,
   *     discriminator))}.
   * @return A one-line description.
   */
  private static String describeGenerator(Object evaluationKey) {
    Object anchor =
        evaluationKey instanceof Pair && ((Pair<?, ?>) evaluationKey).snd instanceof Pair
            ? ((Pair<?, ?>) ((Pair<?, ?>) evaluationKey).snd).fst
            : evaluationKey;
    if (anchor instanceof ReturnValueKey) {
      // A generator anchored on a summary's return value (the slice builtin's `do()`): name the
      // call site that created the context, since the summary node itself carries no position.
      CGNode node = ((ReturnValueKey) anchor).getNode();
      StringBuilder description =
          new StringBuilder("return of ").append(node.getMethod().getSignature());
      Context context = node.getContext();
      if (context instanceof CallerSiteContext || context instanceof AnchoredCallerSiteContext) {
        // The site is a program counter within the method it belongs to: the caller node's method
        // for a caller-site context, the calling method for a context the selector anchored on
        // another node.
        IMethod caller =
            context instanceof AnchoredCallerSiteContext anchored
                ? anchored.getCallerMethod()
                : ((CallerSiteContext) context).getCaller().getMethod();
        int pc =
            context instanceof AnchoredCallerSiteContext anchored
                ? anchored.getCallSite().getProgramCounter()
                : ((CallerSiteContext) context).getCallSite().getProgramCounter();
        description.append(" called from ").append(caller.getSignature());
        if (caller instanceof AstMethod) {
          Position position = ((AstMethod) caller).debugInfo().getInstructionPosition(pc);
          if (position != null) {
            String file = position.getURL() == null ? "?" : position.getURL().getPath();
            description
                .append(" @ ")
                .append(file.substring(file.lastIndexOf('/') + 1))
                .append(':')
                .append(position.getFirstLine());
          }
        }
      } else if (context instanceof CallerContext) {
        description
            .append(" called from ")
            .append(((CallerContext) context).getCaller().getMethod().getSignature());
      } else if (context instanceof CallStringContext) {
        // A call-string context names the innermost call site and its method directly.
        CallString callString = (CallString) context.get(CallStringContextSelector.CALL_STRING);
        if (callString.getMethods().length > 0 && callString.getCallSiteRefs().length > 0) {
          IMethod caller = callString.getMethods()[0];
          int pc = callString.getCallSiteRefs()[0].getProgramCounter();
          description.append(" called from ").append(caller.getSignature());
          if (caller instanceof AstMethod) {
            Position position = ((AstMethod) caller).debugInfo().getInstructionPosition(pc);
            if (position != null) {
              String file = position.getURL() == null ? "?" : position.getURL().getPath();
              description
                  .append(" @ ")
                  .append(file.substring(file.lastIndexOf('/') + 1))
                  .append(':')
                  .append(position.getFirstLine());
            }
          }
        }
      }
      return description.toString();
    }
    if (!(anchor instanceof LocalPointerKey)) return brief(anchor);
    LocalPointerKey local = (LocalPointerKey) anchor;
    CGNode node = local.getNode();
    int vn = local.getValueNumber();
    StringBuilder description =
        new StringBuilder(node.getMethod().getSignature()).append(" vn=").append(vn);
    SSAInstruction def = node.getDU() == null ? null : node.getDU().getDef(vn);
    if (def != null && def.iIndex() >= 0 && node.getMethod() instanceof AstMethod) {
      Position position =
          ((AstMethod) node.getMethod()).debugInfo().getInstructionPosition(def.iIndex());
      if (position != null) {
        String file = position.getURL() == null ? "?" : position.getURL().getPath();
        description
            .append(" @ ")
            .append(file.substring(file.lastIndexOf('/') + 1))
            .append(':')
            .append(position.getFirstLine());
      }
    }
    return description.toString();
  }

  /**
   * The slice-result generator a query key evaluates, or {@code null} when the key is not a slice
   * generator's evaluation. A generator evaluation is keyed {@code Pair(tag, evaluationKey)} with
   * the tag {@code "shape-eval"} or {@code "dtype-eval"} and {@code evaluationKey} the pair {@code
   * TensorGenerator.evaluationKey} builds, whose first component is the generator's class; the
   * returned value is that inner key, so the shape and dtype evaluations of one generator count
   * once. Delegation and value-read keys have other shapes and are not slice queries.
   *
   * @param key A query key.
   * @return The {@link SliceBuiltinOperation} evaluation key inside {@code key}, or {@code null}.
   */
  private static Object sliceGeneratorOf(Object key) {
    if (!(key instanceof Pair)) return null;
    Pair<?, ?> tagged = (Pair<?, ?>) key;
    if (!"shape-eval".equals(tagged.fst) && !"dtype-eval".equals(tagged.fst)) return null;
    if (!(tagged.snd instanceof Pair)) return null;
    Object generatorClass = ((Pair<?, ?>) tagged.snd).fst;
    return generatorClass instanceof Class
            && SliceBuiltinOperation.class.isAssignableFrom((Class<?>) generatorClass)
        ? tagged.snd
        : null;
  }

  /**
   * Returns whether a query evaluation is currently running, i.e. nested reads must divert.
   *
   * @return {@code true} iff inside {@link #evaluate}.
   */
  boolean isEvaluating() {
    return !this.evaluating.isEmpty();
  }

  /**
   * Registers a check a bound evaluation must pass once the iteration converges (wala/ML#1071).
   * Deciding at convergence rather than per evaluation makes the outcome a property of the
   * fixpoint, not of the order queries happened to evaluate in.
   *
   * @param binding The binding the check guards.
   * @param check Whether the converged dependency graph shows the binding applied as required.
   * @param root A query whose evaluation relies on the binding, re-evaluated if the check fails.
   */
  void registerBindingCheck(Object binding, Predicate<WorklistTypeResolver> check, Object root) {
    BindingCheck registered =
        this.bindingChecks.computeIfAbsent(
            binding, b -> new BindingCheck(check, new LinkedHashSet<>()));
    if (registered.roots().add(root)) this.bindingChecksPending = true;
  }

  /**
   * Whether a binding's check failed, so that evaluations relying on it decline.
   *
   * @param binding The binding.
   * @return {@code true} iff the binding was declined.
   */
  boolean isBindingDeclined(Object binding) {
    return this.declinedBindings.contains(binding);
  }

  /**
   * The registered queries memoized under a binding (wala/ML#1071).
   *
   * @param binding The binding.
   * @return The queries wrapped with it.
   */
  Set<Object> boundQueries(Object binding) {
    Set<Object> ret = new LinkedHashSet<>();
    for (Object key : this.transfers.keySet())
      if (key instanceof TensorGenerator.BoundKey bound && bound.binding().equals(binding))
        ret.add(key);
    return ret;
  }

  /**
   * The queries a query's evaluations read.
   *
   * @param key The query.
   * @return Its forward dependency edges; empty when it read none or was never evaluated.
   */
  Set<Object> readsOf(Object key) {
    return Collections.unmodifiableSet(this.reads.getOrDefault(key, Collections.emptySet()));
  }

  /**
   * Runs the registered binding checks against the converged dependency graph. A failed check
   * declines its binding and re-evaluates the queries relying on it, whose declined evaluation
   * joins in the unknown-marked element, and forgets their settled recomputations, so the next
   * settlement pass recomputes them against the decline instead of skipping them.
   *
   * @return {@code true} iff a binding was declined, so the iteration must continue.
   */
  private boolean validateBindings() {
    this.bindingChecksPending = false;
    boolean declined = false;
    for (Map.Entry<Object, BindingCheck> entry : this.bindingChecks.entrySet()) {
      if (this.declinedBindings.contains(entry.getKey())) continue;
      if (entry.getValue().check().test(this)) continue;
      this.declinedBindings.add(entry.getKey());
      LOGGER.fine(() -> "Declined binding " + entry.getKey() + " (wala/ML#1071).");
      for (Object root : entry.getValue().roots()) {
        // The decline is an input of the root that is not a query it reads, so a settled
        // recomputation of the root no longer stands, even when the declined evaluation's join
        // leaves the root's state as it was; and a restricted settlement pass must walk the root.
        this.settledAt.remove(root);
        this.settledReads.remove(root);
        this.markDirty(root);
        this.enqueue(root);
      }
      declined = true;
    }
    return declined;
  }

  /**
   * A tracked computation in progress: the queries it has read itself, not through an evaluation it
   * started, each with the stamp its value had when read.
   */
  private static final class Tracking {

    /** How many evaluations were in progress when the computation began. */
    private final int depth;

    /** Per query read, the {@link #stamps} entry at the read; 0 for a value never written. */
    private final Map<Object, Long> stamps = new LinkedHashMap<>();

    /**
     * Whether the computation's value may not be a function of the values recorded: it read one
     * query at two stamps, or deferred an arm for a settled recomputation.
     */
    private boolean unstable;

    private Tracking(int depth) {
      this.depth = depth;
    }
  }

  /**
   * A tracked computation's value with the stamps of the queries it read.
   *
   * @param value The value, which may be {@code null}.
   * @param stamps Per query read, the stamp its value had.
   */
  private record TrackedValue(Object value, Map<Object, Long> stamps) {}

  /** The tracked computations in progress, innermost first. */
  private final Deque<Tracking> trackings = new ArrayDeque<>();

  /** The values of the tracked computations that completed, by their keys. */
  private final Map<Object, TrackedValue> trackedValues = HashMapFactory.make();

  /** How many tracked computations ran, and how many were answered from a kept value. */
  private long trackedComputations, trackedHits;

  /** Whether each kept value returned is also recomputed and checked; see {@link #tracked}. */
  private boolean checkTracked = Boolean.getBoolean(CHECK_TRACKED_PROPERTY);

  /** The system property that turns on {@link #checkTracked}. */
  static final String CHECK_TRACKED_PROPERTY = "ariadne.typeResolution.checkTrackedValues";

  /** How many kept values the check mode verified. */
  private long checkedTracked;

  /**
   * Computes a value that is not a query's but is computed from queries' values, once for as long
   * as those values stand (wala/ML#1272). The value is kept with the stamp of each query the
   * computation read, and is returned without recomputing while every one of those queries still
   * has the stamp it was read at; a stamp moves on every write of a query's state, by an
   * evaluation, a promotion or the settlement pass. Returning a kept value replays the reads it
   * stands for, so the evaluation in progress has the edges, the consumed versions and the cycle
   * flag it would have had recomputing, and a tracked computation around this one records them.
   *
   * <p>A value is not kept when the computation read one query at two stamps, when it deferred an
   * arm for a settled recomputation, when a query it read was written again before it ended, or
   * when the caller's own conditions, asked after the computation, do not hold. The computation
   * must otherwise depend only on what is final when generators run.
   *
   * @param key The computation's key, distinct from every other computation's.
   * @param computation The computation.
   * @param storable Whether the value just computed may be kept, asked after the computation.
   * @return The value.
   */
  Object tracked(Object key, Supplier<Object> computation, BooleanSupplier storable) {
    TrackedValue stored = this.trackedValues.get(key);
    if (stored != null) {
      if (this.stand(stored.stamps())) {
        this.trackedHits++;
        for (Object read : stored.stamps().keySet()) this.replay(read);
        if (this.checkTracked) this.checkTracked(key, stored, computation, storable);
        return stored.value();
      }
      this.trackedValues.remove(key);
    }
    this.trackedComputations++;
    Tracking tracking = new Tracking(this.evaluating.size());
    Object value = this.track(tracking, computation);
    if (!tracking.unstable && this.stand(tracking.stamps) && storable.getAsBoolean())
      this.trackedValues.put(key, new TrackedValue(value, tracking.stamps));
    return value;
  }

  /**
   * Runs a computation with its reads recorded, and hands the reads on to the tracked computation
   * around it, if they are made for the same evaluation.
   *
   * @param tracking The record of the computation's reads.
   * @param computation The computation.
   * @return The computation's value.
   */
  private Object track(Tracking tracking, Supplier<Object> computation) {
    this.trackings.push(tracking);
    try {
      return computation.get();
    } finally {
      this.trackings.pop();
      Tracking outer = this.trackings.peek();
      if (outer != null && outer.depth == tracking.depth) {
        for (Map.Entry<Object, Long> read : tracking.stamps.entrySet()) {
          Long earlier = outer.stamps.putIfAbsent(read.getKey(), read.getValue());
          if (earlier != null && !earlier.equals(read.getValue())) outer.unstable = true;
        }
        if (tracking.unstable) outer.unstable = true;
      }
    }
  }

  /**
   * The check mode: recomputes a kept value that was just returned and fails if the recomputation,
   * one the value could have been kept from, gives another value or reads other queries.
   *
   * @param key The computation's key.
   * @param stored The kept value.
   * @param computation The computation.
   * @param storable Whether the value just computed may be kept, asked after the computation.
   * @throws AssertionError if the recomputation differs from the kept value.
   */
  private void checkTracked(
      Object key, TrackedValue stored, Supplier<Object> computation, BooleanSupplier storable) {
    Tracking tracking = new Tracking(this.evaluating.size());
    Object value = this.track(tracking, computation);
    // A recomputation that deferred, wrote a query the kept value stands on, or fails the caller's
    // conditions is not one the value would be kept from, so it says nothing about the kept one.
    if (tracking.unstable || !this.stand(stored.stamps()) || !storable.getAsBoolean()) return;
    if (!Objects.equals(value, stored.value()))
      throw new AssertionError(
          "A kept value differs: "
              + key
              + " kept "
              + stored.value()
              + " but recomputes to "
              + value
              + ".");
    if (!tracking.stamps.keySet().equals(stored.stamps().keySet()))
      throw new AssertionError(
          "A kept value's recomputation reads other queries: "
              + key
              + " kept reading "
              + stored.stamps().keySet()
              + " but recomputes reading "
              + tracking.stamps.keySet()
              + ".");
    this.checkedTracked++;
  }

  /** Turns on the check mode of {@link #CHECK_TRACKED_PROPERTY} for this engine. */
  void enableTrackedCheck() {
    this.checkTracked = true;
  }

  /**
   * How many kept values the check mode verified.
   *
   * @return The count.
   */
  long checkedTracked() {
    return this.checkedTracked;
  }

  /**
   * Whether every query still has the stamp recorded for it.
   *
   * @param stamps Per query, the recorded stamp.
   * @return {@code true} iff none was written since.
   */
  private boolean stand(Map<Object, Long> stamps) {
    for (Map.Entry<Object, Long> read : stamps.entrySet())
      if (!read.getValue().equals(this.stamps.getOrDefault(read.getKey(), 0L))) return false;
    return true;
  }

  /**
   * Records a read of a query for the tracked computation in progress, if it made the read itself.
   *
   * @param key The query read.
   */
  private void track(Object key) {
    Tracking tracking = this.trackings.peek();
    if (tracking == null || tracking.depth != this.evaluating.size()) return;
    Long now = this.stamps.getOrDefault(key, 0L);
    Long earlier = tracking.stamps.putIfAbsent(key, now);
    if (earlier != null && !earlier.equals(now)) tracking.unstable = true;
  }

  /**
   * Replays a read of a query a kept value stands for: what {@link #read} records for the
   * evaluation in progress, without evaluating anything, since the query was evaluated when the
   * value was computed.
   *
   * @param key The query read.
   */
  private void replay(Object key) {
    READS_ON_THREAD.get()[0]++;
    Object reader = this.evaluating.peek();
    this.recordEdge(reader, key);
    if (reader != null && (this.inStack.contains(key) || this.unwindPending.containsKey(key)))
      this.cycleSeen = true;
    this.recordConsumedVersion(reader, key);
    this.track(key);
  }

  /**
   * How many tracked computations ran.
   *
   * @return The count.
   */
  long trackedComputations() {
    return this.trackedComputations;
  }

  /**
   * How many tracked computations were answered from a kept value.
   *
   * @return The count.
   */
  long trackedHits() {
    return this.trackedHits;
  }

  /**
   * Records the edge of a read for the evaluation making it, if any.
   *
   * @param reader The evaluating query, or {@code null} outside any evaluation.
   * @param key The query read.
   */
  private void recordEdge(Object reader, Object key) {
    if (this.settling && reader != null && reader.equals(key)) this.selfReadInRecomputation = true;
    if (reader != null && !reader.equals(key)) {
      Set<Object> readerReads = this.reads.get(reader);
      if (readerReads == null) {
        readerReads = HashSetFactory.make();
        this.reads.put(reader, readerReads);
        this.readsOrder.put(reader, this.readsOrder.size());
      }
      if (readerReads.add(key)) {
        this.newEdgeReaders.add(reader);
        this.promotionEdgeReaders.add(reader);
      }
      this.dependents.computeIfAbsent(key, k -> HashSetFactory.make()).add(reader);
    }
  }

  /**
   * Records the version of the value a reader consumes (wala/ML#758).
   *
   * @param reader The evaluating query, or {@code null} outside any evaluation.
   * @param key The query read.
   */
  private void recordConsumedVersion(Object reader, Object key) {
    if (reader != null && !reader.equals(key))
      this.readVersions
          .computeIfAbsent(reader, k -> HashMapFactory.make())
          .put(key, this.versions.getOrDefault(key, 0));
  }

  /**
   * How many reads and demands each thread has made of any engine, so a computation can tell
   * whether it read a value the fixpoint may still revise ({@link #readsOnThread}).
   */
  private static final ThreadLocal<long[]> READS_ON_THREAD =
      ThreadLocal.withInitial(() -> new long[1]);

  /**
   * How many reads and demands this thread has made of any engine. A computation that left this
   * unchanged read no query, so it depends only on what is final when generators run.
   *
   * @return The count.
   */
  static long readsOnThread() {
    return READS_ON_THREAD.get()[0];
  }

  /**
   * Reads a query's current value from inside another query's evaluation: records the edge,
   * registers and schedules the dependency on first sight, and returns the state value or the
   * kind-appropriate bottom without recursing.
   *
   * @param key The query being read.
   * @param transfer The query's transfer, used if this is its first sighting.
   * @param shapeKind {@code true} for a {@link ShapeResult}-valued query, {@code false} for dtypes.
   * @return The current value.
   */
  Object read(Object key, Supplier<Object> transfer, boolean shapeKind) {
    READS_ON_THREAD.get()[0]++;
    Object reader = this.evaluating.peek();
    this.recordEdge(reader, key);
    if (!this.transfers.containsKey(key)) this.register(key, transfer, shapeKind);
    // Hybrid scheduling: a first-sighted query evaluates inline, so recursion gives the acyclic
    // region its natural (reverse-topological) order and each such query runs once; only an
    // on-stack read (a cycle) returns the current value and defers to the worklist, which then
    // converges just the cycle-affected keys.
    // Past the inline depth bound, a first-sighted query unwinds the evaluations in progress (see
    // DepthUnwind), so the longest acyclic dependency chain is not the Java stack's depth.
    boolean inProgress = this.inStack.contains(key) || this.unwindPending.containsKey(key);
    if (!this.evaluated.contains(key) && !inProgress) {
      if (this.evaluating.size() >= INLINE_DEPTH_BOUND) {
        this.depthUnwinds++;
        throw new DepthUnwind(key, this.evaluating);
      }
      this.evaluate(key);
    } else if (inProgress) this.cycleSeen = true;
    // Record the version of the value the reader consumes (wala/ML#758): recorded after any inline
    // evaluation so it matches the value returned below, and an on-stack (interim) read records the
    // pre-completion version, which lags as soon as the read value grows.
    this.recordConsumedVersion(reader, key);
    this.track(key);
    Object value = this.state.get(key);
    if (value == NULL_DTYPE) return null;
    if (value != null) return value;
    return shapeKind ? ShapeResult.bottom() : EnumSet.noneOf(DType.class);
  }

  /**
   * Returns whether an evaluation running now could observe an interim (still-converging) value.
   * Interim values are observable only through an on-stack read (a cycle, which sets the cycle
   * flag) or a read of an enqueued key (a nonempty worklist); without either, hybrid inline
   * scheduling makes every read final on return.
   *
   * @return {@code true} iff a value read now may later grow.
   */
  boolean mayObserveInterim() {
    return this.cycleSeen || !this.worklist.isEmpty();
  }

  /**
   * Returns whether the post-fixpoint canonicalization pass is running, i.e. every read observes a
   * settled (final) value and finality-dependent arms may fire.
   *
   * @return {@code true} iff inside {@link #canonicalize}.
   */
  boolean isSettling() {
    return this.settling;
  }

  /**
   * Registers the currently evaluating query for a recomputation against the settled state in
   * {@link #canonicalize}. A transfer whose arm must not fire on a possibly-interim observation
   * (wala/ML#717's contract seed under the wala/ML#758 clause-1 contract) calls this and withholds
   * the arm; the canonicalization recomputes the query with {@link #isSettling} holding, where the
   * arm can fire on a provably final observation, and replaces the settled value.
   */
  void requestSettledRecomputation() {
    Object reader = this.evaluating.peek();
    if (reader != null) this.settlementRequested.add(reader);
    this.deferTracked();
  }

  /**
   * Marks the tracked computation in progress, if it is the one deferring, as one whose value is
   * not to be kept: a deferring arm answers differently once its observation is final.
   */
  private void deferTracked() {
    Tracking tracking = this.trackings.peek();
    if (tracking != null && tracking.depth == this.evaluating.size()) tracking.unstable = true;
  }

  /**
   * Demands a query's stabilized value from outside any evaluation: schedules it if new, runs the
   * fixpoint (with SCC promotion passes), and returns the final value.
   *
   * @param key The demanded query.
   * @param transfer The query's transfer, used if this is its first sighting.
   * @param shapeKind {@code true} for a {@link ShapeResult}-valued query, {@code false} for dtypes.
   * @return The stabilized value.
   */
  Object demand(Object key, Supplier<Object> transfer, boolean shapeKind) {
    READS_ON_THREAD.get()[0]++;
    if (!this.transfers.containsKey(key)) this.register(key, transfer, shapeKind);
    this.unwinding(
        () -> {
          if (!this.evaluated.contains(key) && !this.inStack.contains(key)) this.evaluate(key);
          return null;
        });
    this.solve();
    this.track(key);
    Object value = this.state.get(key);
    if (value == NULL_DTYPE) return null;
    if (value != null) return value;
    return shapeKind ? ShapeResult.bottom() : EnumSet.noneOf(DType.class);
  }

  /**
   * Parses the wala/ML#756 perturbation seed from the {@code ariadne.typeResolution.shuffleCycles}
   * system property or the {@code ARIADNE_SHUFFLE_CYCLES} environment variable. An unparsable value
   * disables the perturbation with a logged warning rather than aborting the analysis: the knob is
   * diagnostic, so misconfiguration should be visible but harmless.
   *
   * @return The configured seed, or {@code null} when neither setting is present or the value does
   *     not parse as a {@code long}.
   */
  static Long parseCycleShuffleSeed() {
    String seed = System.getProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY);
    if (seed == null) seed = System.getenv(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_VARIABLE);
    return parseCycleShuffleSeed(seed);
  }

  /**
   * Parsing core of {@link #parseCycleShuffleSeed()}, taking the setting value directly so the
   * defensive behavior is testable without owning the shared system property (the suite runs test
   * methods in parallel).
   *
   * @param seed The configured value, or {@code null} when unset.
   * @return The seed, or {@code null} when unset or unparsable (logged).
   */
  static Long parseCycleShuffleSeed(String seed) {
    if (seed == null) return null;
    try {
      return Long.valueOf(seed);
    } catch (NumberFormatException e) {
      LOGGER.warning("Ignoring the unparsable cycle-shuffle seed: " + seed + ".");
      return null;
    }
  }

  /**
   * Creates the cycle-order perturbation source (wala/ML#756).
   *
   * @return A {@link Random} seeded per {@link #parseCycleShuffleSeed}, or {@code null} when the
   *     knob is off.
   */
  private static Random createCycleShuffle() {
    Long seed = parseCycleShuffleSeed();
    return seed == null ? null : new Random(seed);
  }

  private void enqueue(Object key) {
    if (!this.enqueued.add(key)) return;
    // Perturbation point 1 of the wala/ML#756 knob: enqueueing at a random end permutes the poll
    // order among the keys concurrently on the worklist, which are exactly the cycle-affected keys.
    if (this.cycleShuffle != null && !this.worklist.isEmpty() && this.cycleShuffle.nextBoolean())
      this.worklist.addFirst(key);
    else this.worklist.add(key);
  }

  /**
   * Orders a changed query's dependents for re-enqueueing: iteration order as-is normally, a seeded
   * shuffle under the wala/ML#756 perturbation knob (point 2: the relative order in which a grown
   * value's readers join the worklist).
   *
   * @param dependents The dependents to order.
   * @return The re-enqueue order.
   */
  private Iterable<Object> orderDependentsForReEnqueue(Set<Object> dependents) {
    if (this.cycleShuffle == null || dependents.size() < 2) return dependents;
    List<Object> shuffled = new ArrayList<>(dependents);
    Collections.shuffle(shuffled, this.cycleShuffle);
    return shuffled;
  }

  /**
   * Runs the fixpoint: chaotic iteration, a read-version staleness sweep, then SCC promotion,
   * repeating until none moves. The sweep runs before promotion so the pure-cycle step (whose
   * unknown mark is permanent under join) only fires on a staleness-drained state, and promotion's
   * value changes feed back through the sweep on the next round.
   */
  private void solve() {
    // Nothing queued, no cycle observed, and no settlement requested since the last pass: every
    // value came from a single inline evaluation and is already final, so skip the iteration and
    // (Tarjan-priced) promotion entirely. This is what makes repeated demands (the second seeding
    // pass) effectively free.
    if (this.worklist.isEmpty()
        && !this.cycleSeen
        && this.settlementRequested.isEmpty()
        && !this.bindingChecksPending) return;
    boolean outer = this.solveDepth++ == 0;
    long solveStart = outer ? System.nanoTime() : 0;
    try {
      boolean moved;
      do {
        this.iterate();
        long scanStart = System.nanoTime();
        moved = this.enqueueStaleReaders();
        this.staleScanNanos += System.nanoTime() - scanStart;
        if (!moved) moved = this.cycleSeen && this.promotePureCycles();
        if (!moved) moved = this.validateBindings();
      } while (moved);
      if (this.cycleSeen || !this.settlementRequested.isEmpty()) this.canonicalize();
      this.cycleSeen = false;
      this.dumpFiltered();
    } finally {
      this.solveDepth--;
      if (outer) {
        long now = System.nanoTime();
        this.outerSolves++;
        this.outerSolveNanos += now - solveStart;
        if (now - this.walkCensusLoggedAt >= WALK_CENSUS_INTERVAL_NANOS) {
          this.walkCensusLoggedAt = now;
          SETTLEMENT_CENSUS.fine(() -> "Walk census so far: " + this.walkCensus() + ".");
          SETTLEMENT_CENSUS.fine(
              () -> "Settlement census so far: " + this.settlementCensus() + ".");
        }
      }
    }
  }

  /**
   * Describes the settlement census.
   *
   * @return The description.
   */
  String settlementCensus() {
    return this.settlementPasses
        + " passes ("
        + this.guardedSettlementPasses
        + " inside a generator guard), "
        + this.settledRecomputations
        + " settled recomputations ("
        + this.guardedSettledRecomputations
        + " inside a generator guard, of which "
        + this.guardDecidedRecomputations
        + " a guard decided), "
        + this.settledSkips
        + " skipped ("
        + this.guardedSettledSkips
        + " inside a generator guard, "
        + this.checkedSkips
        + " checked, of which "
        + this.guardDecidedCheckedSkips
        + " a guard decided); "
        + this.unchangedSettlementPasses
        + " passes replaced nothing; "
        + this.iteratedComponents
        + " components iterated in "
        + this.iterationSteps
        + " further steps (the largest holds "
        + this.largestIteratedComponent
        + " queries, "
        + this.checkedFixpointQueries
        + " queries checked at a fixpoint), "
        + this.orbits
        + " orbits found ("
        + this.orbitMeets
        + " values set to a meet, "
        + this.orbitJoins
        + " to a join), "
        + this.joinedComponents
        + " components left to join, "
        + this.sightedWhileIterating
        + " queries first sighted in their steps ("
        + this.sightedOnCycle
        + " of them read the component), "
        + this.guardHeld
        + " recomputations of queries left to join held inside a generator guard ("
        + this.guardHeldDiffering
        + " differing from the settled value); recomputations took "
        + TimeUnit.NANOSECONDS.toMillis(this.sweepRecomputeNanos)
        + " ms; "
        + this.widenedQueries
        + " queries widened where they kept making new shapes; recomputations by cause: "
        + this.causeCensus();
  }

  /**
   * Describes the settled recomputations by cause and outcome, leaving out a cause with none.
   *
   * @return The description.
   */
  private String causeCensus() {
    List<String> causes = new ArrayList<>();
    for (RecomputationCause cause : RecomputationCause.values()) {
      long[] outcomes = this.recomputationsByCause[cause.ordinal()];
      if (Arrays.stream(outcomes).sum() == 0) continue;
      causes.add(
          cause.description
              + ": "
              + outcomes[RecomputationOutcome.SAME.ordinal()]
              + " the same, "
              + outcomes[RecomputationOutcome.REPLACED.ordinal()]
              + " replaced"
              + (this.replacedBySettlement == null
                  ? ""
                  : " (" + this.restoredByCause[cause.ordinal()] + " restored)")
              + ", "
              + outcomes[RecomputationOutcome.KEPT.ordinal()]
              + " kept");
    }
    return causes.isEmpty() ? "none" : String.join("; ", causes);
  }

  /**
   * Describes the walk census.
   *
   * @return The description.
   */
  String walkCensus() {
    return this.state.size()
        + " queries and "
        + this.reads.values().stream().mapToInt(Set::size).sum()
        + " dependency edges; "
        + this.outerSolves
        + " outermost solves in "
        + TimeUnit.NANOSECONDS.toMillis(this.outerSolveNanos)
        + " ms; promotion: "
        + this.promotionWalks
        + " walks ("
        + this.promotionWholeWalks
        + " whole-graph), "
        + this.promotionGraphQueries
        + " graph queries, "
        + this.promotionClosureQueries
        + " closure queries, "
        + this.promotionCondensedQueries
        + " condensed queries, "
        + TimeUnit.NANOSECONDS.toMillis(this.promotionWalkNanos)
        + " ms; sweep: "
        + this.sweepWalks
        + " walks ("
        + this.sweepWholeWalks
        + " whole-graph, "
        + this.sweepGuardedWholeWalks
        + " of them inside a generator guard), "
        + this.sweepGraphQueries
        + " graph queries, "
        + this.sweepClosureQueries
        + " closure queries, "
        + this.sweepCondensedQueries
        + " condensed queries, "
        + TimeUnit.NANOSECONDS.toMillis(this.sweepWalkNanos)
        + " ms; sweep target selection: "
        + this.sweepReadTests
        + " read tests over "
        + this.sweepReadsSum
        + " reads and "
        + this.sweepChangedSum
        + " changed queries, "
        + this.sweepReadProbes
        + " probes, "
        + TimeUnit.NANOSECONDS.toMillis(this.sweepSelectNanos)
        + " ms; staleness scans: "
        + TimeUnit.NANOSECONDS.toMillis(this.staleScanNanos)
        + " ms; "
        + this.hashCensus();
  }

  /**
   * Describes how the engine's query keys spread over hash buckets (wala/ML#1183): the distinct
   * hash codes of the queries, and how many queries share a bucket with 7 or more others in a hash
   * table holding them all, where {@link java.util.HashMap} turns a bucket into a tree.
   *
   * @return The description.
   */
  private String hashCensus() {
    int queries = this.state.size();
    int capacity = Integer.highestOneBit(Math.max(1, (int) (queries / 0.75f)) * 2 - 1);
    Set<Integer> hashes = new HashSet<>();
    int[] buckets = new int[capacity];
    for (Object query : this.state.keySet()) {
      int hash = query.hashCode();
      hashes.add(hash);
      buckets[(hash ^ (hash >>> 16)) & (capacity - 1)]++;
    }
    long crowded = 0;
    for (int bucket : buckets) if (bucket >= 8) crowded += bucket;
    return "query hashes: "
        + hashes.size()
        + " distinct of "
        + queries
        + ", "
        + crowded
        + " queries in buckets of 8 or more";
  }

  /**
   * Returns the walk census's counts: promotion walks, whole-graph promotion walks, promotion graph
   * queries, promotion closure queries, promotion condensed queries, sweep walks, whole-graph sweep
   * walks, guarded whole-graph sweep walks, sweep graph queries, sweep closure queries, sweep
   * condensed queries and outermost solves.
   *
   * @return The counts, in that order.
   */
  long[] walkCounts() {
    return new long[] {
      this.promotionWalks,
      this.promotionWholeWalks,
      this.promotionGraphQueries,
      this.promotionClosureQueries,
      this.promotionCondensedQueries,
      this.sweepWalks,
      this.sweepWholeWalks,
      this.sweepGuardedWholeWalks,
      this.sweepGraphQueries,
      this.sweepClosureQueries,
      this.sweepCondensedQueries,
      this.outerSolves
    };
  }

  /**
   * Returns the read-set census's counts: the read tests, the reads and the changed queries summed
   * over them, and the probes they made.
   *
   * @return The counts, in that order.
   */
  long[] readTestCounts() {
    return new long[] {
      this.sweepReadTests, this.sweepReadsSum, this.sweepChangedSum, this.sweepReadProbes
    };
  }

  /**
   * Counts the queries in the components of a condensation that lie in a region. In the check mode
   * of {@link #CHECK_SKIPS_PROPERTY} the condensation is the whole graph's, so either mode counts
   * the restricted pass's queries; a component lies wholly inside or outside the region.
   *
   * @param sccs The condensation.
   * @param region The region, or {@code null} for the whole graph.
   * @return The count.
   */
  private static long condensedQueries(List<List<Object>> sccs, Set<Object> region) {
    long count = 0;
    for (List<Object> scc : sccs)
      if (region == null || region.contains(scc.get(0))) count += scc.size();
    return count;
  }

  /**
   * Re-enqueues every query whose recorded read versions lag the read values' current versions
   * (wala/ML#758's read-version staleness criterion, exact where the change-driven re-enqueue in
   * {@link #reEnqueueStaleDependents} spares on-stack readers): such a reader consumed a value that
   * has since grown, so its contribution predates evidence it must incorporate. Queries already
   * enqueued or pinned at the evaluation cap are skipped; the pinned skip is what keeps repeated
   * sweeps from re-enqueueing a capped query forever.
   *
   * @return {@code true} iff any stale reader was re-enqueued.
   */
  private boolean enqueueStaleReaders() {
    // Only a candidate can be stale; when none is, the whole scan below would find nothing, and
    // skipping it is what keeps a demand from paying for every record of every reader.
    Set<Object> retained = HashSetFactory.make();
    boolean anyStale = false;
    for (Object key : this.staleCandidates)
      if (this.staleScanSkips(key)) retained.add(key);
      else if (this.isStale(key)) {
        anyStale = true;
        break;
      }
    if (this.checkSkips && !anyStale && !this.findStaleReaders(null).isEmpty())
      throw new AssertionError("A reader outside the staleness candidates is stale.");
    if (!anyStale) {
      this.staleCandidates.retainAll(retained);
      return false;
    }
    this.staleCandidates.clear();
    List<Object> stale = this.findStaleReaders(this.staleCandidates);
    this.staleCandidates.addAll(stale);
    if (!stale.isEmpty()) this.staleScanHits++;
    LOGGER.fine(() -> "Staleness sweep re-enqueueing " + stale.size() + " readers.");
    this.staleReaders.addAll(stale);
    // Perturbation point 3 of the wala/ML#756 knob: the sweep's enqueue order is the map's
    // (deterministic) iteration order unless the seeded shuffle perturbs it, so the sweep adds no
    // unseeded order dependence.
    if (this.cycleShuffle != null && stale.size() > 1)
      Collections.shuffle(stale, this.cycleShuffle);
    for (Object key : stale) this.enqueue(key);
    return true;
  }

  /**
   * Scans every reader, in {@link #readVersions}' order, for a stale one.
   *
   * @param skipped Where to add the readers the scan skips, or {@code null}.
   * @return The stale readers, in scan order.
   */
  private List<Object> findStaleReaders(Set<Object> skipped) {
    List<Object> stale = new ArrayList<>();
    for (Object key : this.readVersions.keySet())
      if (this.staleScanSkips(key)) {
        if (skipped != null) skipped.add(key);
      } else if (this.isStale(key)) stale.add(key);
    return stale;
  }

  /**
   * Returns whether the staleness scan passes over a reader: it is already enqueued, or pinned at
   * the evaluation cap.
   *
   * @param key The reader.
   * @return {@code true} iff the scan skips the reader.
   */
  private boolean staleScanSkips(Object key) {
    return this.enqueued.contains(key)
        || this.evaluationCounts.getOrDefault(key, 0) > EVALUATION_CAP;
  }

  /**
   * Returns whether any version the reader's latest evaluation recorded lags the value's current
   * one.
   *
   * @param key The reader.
   * @return {@code true} iff the reader is stale.
   */
  private boolean isStale(Object key) {
    Map<Object, Integer> records = this.readVersions.get(key);
    if (records == null) return false;
    for (Map.Entry<Object, Integer> record : records.entrySet())
      if (this.versions.getOrDefault(record.getKey(), 0) > record.getValue()) return true;
    return false;
  }

  /**
   * Bumps a query's value version, making each reader that recorded the value a staleness
   * candidate.
   *
   * @param key The query whose value changed.
   */
  private void bumpVersion(Object key) {
    this.versions.merge(key, 1, Integer::sum);
    this.staleCandidates.addAll(this.dependents.getOrDefault(key, Collections.emptySet()));
  }

  /**
   * Recomputes cycle-affected queries once against the settled state, replacing their values
   * (wala/ML#753, wala/ML#748). The iteration to the fixpoint joins every recomputation into the
   * state, so join history rides into the settled values: a transfer that consumed an on-stack
   * interim read collapses to the unknown-marked element, {@link ShapeResult#union} disjoins the
   * mark permanently, and interim per-axis folds survive alongside their converged refinements.
   * Which artifacts persist depends on the order the cycle iterated in, so the settled states are
   * order-dependent exactly on cycle-affected keys. One post-fixpoint recomputation per key against
   * the settled (no longer interim) values, replacing instead of joining, makes each canonical
   * value a function of its dependencies' final values rather than of the join history.
   *
   * <p>The sweep runs over the SCC condensation in reverse-topological order (the Tarjan emission
   * order), so every recomputation reads dependencies that are already canonical; a key is
   * recomputed only if it sits in a cyclic SCC (where join history can originate), reads a key
   * whose canonical value changed (where it can propagate), ever consumed a since-grown value (the
   * {@link #staleReaders} criterion — its joins can hold artifacts of the stale consumption even
   * off-cycle), or requested a settled recomputation for a withheld finality-dependent arm ({@link
   * #requestSettledRecomputation}). Within a cyclic SCC the replacements are computed jointly
   * against the settled state before any is written back (a Jacobi step), so intra-SCC
   * recomputation order cannot influence the result. Two guards keep the replacement semantics
   * sound: a transfer that throws keeps the settled value, and a recomputation may not degrade a
   * non-bottom settled value to ⊥ &mdash; the settled evidence that the value is a tensor stands,
   * and the pure-cycle promotion's unknown-marked elements (whose baseless transfers recompute to
   * ⊥) must not be reclassified as "not a tensor."
   */
  private void canonicalize() {
    int replaced = 0;
    this.settlementPasses++;
    this.guardedSettlement = TensorGenerator.guardsInProgress();
    if (this.guardedSettlement) this.guardedSettlementPasses++;
    this.settling = true;
    try {
      Set<Object> changed = HashSetFactory.make();
      replaced += this.sweep(changed, true);
      // The closing step (wala/ML#1021): a query whose settled value is ⊥ and whose recomputation
      // deferred under a re-entered len fold is a member of a cycle the engine cannot see that no
      // evaluation gave a base; it takes the unknown-marked element, the pure-cycle policy, and
      // its readers are recomputed against the settled state by the same sweep, under the same
      // guards and in the same reverse-topological order. The step repeats until a round promotes
      // nothing: a recomputed reader may itself defer under a re-entry and stand at ⊥. It
      // terminates, since a promoted query never returns to ⊥ (a recomputation may not degrade a
      // non-⊥ settled value) and the queries are finitely many.
      while (true) {
        List<Object> promoted = new ArrayList<>();
        for (Object query : this.reentryDeferredAtSettlement) {
          // Only shape-valued queries promote, as in promotePureCycles; a bottom dtype set already
          // reads as "no info."
          if (!Boolean.TRUE.equals(this.shapeKinds.get(query))) continue;
          if (!isBottomValue(this.state.get(query))) continue;
          promoted.add(query);
        }
        int deferred = this.reentryDeferredAtSettlement.size();
        LOGGER.fine(
            () ->
                "Settled recomputations deferring under a re-entered len fold: "
                    + deferred
                    + ", of which settled at ⊥ and promoted to the unknown-marked element: "
                    + promoted.size()
                    + (promoted.isEmpty()
                        ? "."
                        : " " + promoted.stream().map(WorklistTypeResolver::brief).toList() + "."));
        this.reentryDeferredAtSettlement.clear();
        if (promoted.isEmpty()) break;
        Set<Object> promotedNow = HashSetFactory.make();
        for (Object query : promoted) {
          this.putState(query, ShapeResult.unknown());
          this.bumpVersion(query);
          this.stamp(query, true);
          promotedNow.add(query);
        }
        replaced += promoted.size();
        replaced += this.sweep(promotedNow, false);
      }
    } finally {
      this.settling = false;
      // Both target sets are served by this pass; a replaced value propagates through the changed
      // set above, and later solves must not re-target (and re-replace) on stale membership.
      this.staleReaders.clear();
      this.settlementRequested.clear();
      this.reentryDeferredAtSettlement.clear();
    }
    if (replaced == 0) this.unchangedSettlementPasses++;
    if (replaced > 0) {
      int count = replaced;
      LOGGER.fine(() -> "Canonicalization replaced " + count + " settled values.");
    }
  }

  /**
   * Whether a query read any of a set of queries, for the sweep's target selection.
   *
   * @param key The query.
   * @param queries The queries.
   * @return Whether {@code key} read any of {@code queries}.
   */
  private boolean readsAny(Object key, Set<Object> queries) {
    Set<Object> keyReads = this.reads.getOrDefault(key, Collections.emptySet());
    this.sweepReadTests++;
    this.sweepReadsSum += keyReads.size();
    this.sweepChangedSum += queries.size();
    return this.intersects(keyReads, queries);
  }

  /**
   * Whether two sets share an element. It iterates the smaller set and looks each element up in the
   * larger, and answers at once when either is empty, so a test costs the smaller set's size. The
   * sweep tests a query's few reads against a changed set that can hold much of the graph, and
   * {@link Collections#disjoint} chooses which set to iterate by whether its first argument is a
   * {@link Set} rather than by size (wala/ML#1176).
   *
   * @param first A set.
   * @param second Another set.
   * @return Whether the sets share an element.
   */
  boolean intersects(Set<?> first, Set<?> second) {
    if (first.isEmpty() || second.isEmpty()) return false;
    Set<?> iterated = first.size() <= second.size() ? first : second;
    Set<?> probed = iterated == first ? second : first;
    for (Object element : iterated) {
      this.sweepReadProbes++;
      if (probed.contains(element)) return true;
    }
    return false;
  }

  /**
   * One sweep of {@link #canonicalize} over the SCC condensation in reverse-topological order: a
   * key is recomputed when it reads a key in {@code changed}, and in the full sweep also when it
   * sits in a cyclic SCC, is an ever-stale reader or requested a settled recomputation; a
   * replacement that differs from the settled value is written back and its key added to {@code
   * changed}, so later SCCs see it.
   *
   * @param changed The keys whose settled values changed; grown by this sweep.
   * @param full Whether the cyclic, stale and requested criteria apply (the pass's first sweep) or
   *     only the changed-read criterion (the promotion rounds).
   * @return The number of settled values replaced.
   */
  private int sweep(Set<Object> changed, boolean full) {
    int replaced = 0;
    String probeFilter = System.getProperty("ariadne.typeResolution.canonProbe");
    // The region a sweep can target: the readers, transitively, of the changed queries and, in a
    // full sweep with the skip on, of the stale, requested and possibly-unskippable cyclic queries
    // and of every reader that gained an edge since the last full sweep (where a new cycle can
    // be). Outside it no query is targeted and none is written, so walking it changes nothing. A
    // pass inside a generator guard is restricted like any other: a recomputation a guard decided
    // leaves no record, so its query is possibly unskippable.
    long walkStart = System.nanoTime();
    this.sweepWalks++;
    this.sweepGraphQueries += this.state.size();
    Set<Object> region = null;
    if (this.restrictSettlement && !(full && !this.skipEnabled)) {
      Set<Object> seeds = HashSetFactory.make();
      seeds.addAll(changed);
      if (full) {
        seeds.addAll(this.staleReaders);
        seeds.addAll(this.settlementRequested);
        seeds.addAll(this.dirtyCyclic);
        seeds.addAll(this.newEdgeReaders);
      }
      region = this.readerClosure(seeds);
      this.sweepClosureQueries += region.size();
    } else {
      this.sweepWholeWalks++;
      if (this.guardedSettlement) this.sweepGuardedWholeWalks++;
    }
    if (full) this.newEdgeReaders.clear();
    List<List<Object>> sccs;
    if (region == null) sccs = this.tarjan();
    else if (this.checkSkips) {
      sccs = this.tarjan();
      this.checkRestriction(sccs, region);
    } else sccs = this.tarjan(region);
    this.sweepWalkNanos += System.nanoTime() - walkStart;
    this.sweepCondensedQueries += condensedQueries(sccs, region);
    if (region != null && this.checkSkips) this.checkedSweepRegion = region;
    // The cause of each target of the component in hand, for the census.
    Map<Object, RecomputationCause> causes = HashMapFactory.make();
    try {
      for (List<Object> scc : sccs) {
        boolean cyclic =
            scc.size() > 1
                || this.reads.getOrDefault(scc.get(0), Collections.emptySet()).contains(scc.get(0));
        if (cyclic) this.noteCyclic(scc, region, full);
        List<Object> targets = new ArrayList<>();
        if (!causes.isEmpty()) causes.clear();
        // Beyond the cyclic-SCC and changed-read criteria, an ever-stale reader is targeted (its
        // join history can hold artifacts of the since-grown consumption even after re-evaluation
        // caught it up), and so is a settlement-requested query (its transfer withheld a
        // finality-dependent arm that must now fire against the settled state). wala/ML#758.
        long selectStart = System.nanoTime();
        for (Object key : scc) {
          RecomputationCause cause;
          if (full && this.staleReaders.contains(key)) cause = RecomputationCause.STALE;
          else if (full && this.settlementRequested.contains(key))
            cause = RecomputationCause.REQUESTED;
          else if (this.readsAny(key, changed)) cause = RecomputationCause.CHANGED_READ;
          // A cyclic query whose inputs have not changed since its last settled recomputation
          // would recompute to the value it settled at, which the write-back below would leave
          // in place.
          else if (full && cyclic) cause = this.cyclicCause(key);
          else continue;
          if (cause != null) {
            targets.add(key);
            causes.put(key, cause);
          } else {
            this.settleDirty.remove(key);
            this.dirtyCyclic.remove(key);
            this.settledSkips++;
            if (this.guardedSettlement) this.guardedSettledSkips++;
            if (this.checkSkips) this.checkSkip(key);
          }
        }
        this.sweepSelectNanos += System.nanoTime() - selectStart;
        if (full && probeFilter != null)
          for (Object key : scc) {
            Object settled = this.state.get(key);
            if (!(settled instanceof ShapeResult)) continue;
            ShapeResult shapes = (ShapeResult) settled;
            if (!shapes.members().isEmpty() || !shapes.hasUnknown()) continue;
            if (!String.valueOf(key).contains(probeFilter)) continue;
            boolean targeted = targets.contains(key);
            boolean inCycle = cyclic;
            LOGGER.fine(
                () ->
                    "CANON-PROBE "
                        + brief(key)
                        + " cyclic="
                        + inCycle
                        + " sccSize="
                        + scc.size()
                        + " targeted="
                        + targeted
                        + " edges="
                        + this.reads.getOrDefault(key, Collections.emptySet()).size());
          }
        // The check mode selects over the whole graph, so a target outside the region is a query
        // the restricted pass alone would have left unrecomputed.
        if (region != null && this.checkSkips)
          for (Object key : scc)
            if (!region.contains(key) && targets.contains(key))
              throw new AssertionError("A settlement target lies outside the region: " + key + ".");
        if (targets.isEmpty()) continue;
        boolean iterating = cyclic && this.iterationEnabled && !this.guardedSettlement;
        if (iterating) this.sightedInSteps = new ArrayList<>();
        List<Object[]> writes = this.step(targets, causes, changed, probeFilter, null);
        replaced += writes.size();
        // A pass inside a generator guard makes its one step: a guard can cut a recomputation at a
        // frame that is gone by the next pass, and a value moving for that reason is not an orbit.
        if (iterating && !writes.isEmpty()) {
          List<Object> sighted = this.sightedInSteps;
          replaced += this.iterate(scc, targets, writes, changed, probeFilter);
          this.countSighted(scc, sighted);
        }
        this.sightedInSteps = null;
      }
    } finally {
      this.checkedSweepRegion = null;
      this.sightedInSteps = null;
    }
    return replaced;
  }

  /**
   * Counts the queries first sighted during an iterated component's steps, and those of them that
   * read the component, directly or through another of them. Each was sighted by a read the
   * component made, so one that reads the component is on its cycle, and {@link #iterate} did not
   * recompute it: a step made after its evaluation can have rewritten a query it read.
   *
   * @param scc The component.
   * @param sighted The queries first sighted during its steps.
   */
  private void countSighted(List<Object> scc, List<Object> sighted) {
    List<Object> rest = sighted;
    this.sightedWhileIterating += rest.size();
    Set<Object> onCycle = HashSetFactory.make(scc);
    boolean found = true;
    while (found) {
      found = false;
      List<Object> unplaced = new ArrayList<>();
      for (Object key : rest)
        if (Collections.disjoint(this.reads.getOrDefault(key, Collections.emptySet()), onCycle))
          unplaced.add(key);
        else {
          onCycle.add(key);
          this.sightedOnCycle++;
          found = true;
        }
      rest = unplaced;
    }
  }

  /**
   * One step of a settlement sweep on a component: recomputes the targets against the state, and
   * only then writes back each value that differs from the settled one, so the order of the targets
   * cannot influence the values (a Jacobi step). A recomputation that returns ⊥ over a settled
   * value that is not ⊥ writes nothing. A query of {@link #joinSettled} is written the join of its
   * settled value and its recomputation, or nothing once it is pinned.
   *
   * <p>A step of {@link #iterate} also watches for a query that keeps making shapes no query of the
   * component holds. At the {@link #GROWTH_WIDENING_LIMIT}th such recomputation the query enters
   * {@link #widenedAtSettlement} and is written the widening of those recomputations, and every
   * later write widens its settled value by the recomputation. Every recomputation is taken without
   * the members a shape a widening made covers ({@link #absorb}).
   *
   * <p>A pass inside a generator guard writes no query of {@link #joinSettled}. What such a pass
   * recomputes can hold a guard's cut, made in the recomputation's own run or read from a query
   * that took one in this pass or in an earlier pass inside a guard. A query that is replaced gives
   * the cut back when a later pass recomputes it, and a query that only grows would keep it for
   * good. The recomputation leaves no record ({@link #recordSettledRecomputation}), so the next
   * pass recomputes the query.
   *
   * @param targets The queries to recompute.
   * @param causes The cause of each target, or {@code null} when each is {@link
   *     RecomputationCause#ITERATED}.
   * @param changed The keys whose settled values changed; grown by this step.
   * @param probeFilter The text a probed query's name contains, or {@code null}.
   * @param growth The shapes the component has held and the writes that made new ones, or {@code
   *     null} outside {@link #iterate}.
   * @return The writes, each a query, the value it held and the value written.
   */
  private List<Object[]> step(
      List<Object> targets,
      Map<Object, RecomputationCause> causes,
      Set<Object> changed,
      String probeFilter,
      Growth growth) {
    Object[] values = new Object[targets.size()];
    long recomputeStart = System.nanoTime();
    for (int i = 0; i < values.length; i++) {
      Object key = targets.get(i);
      // The clock at the entry of the attempt that completes: an unwound attempt reruns, and the
      // writes the pending chain makes before the rerun precede every read the rerun records.
      long[] entry = {0};
      values[i] =
          this.unwinding(
              () -> {
                entry[0] = this.clock;
                return this.recomputeSettled(key);
              });
      this.settledRecomputations++;
      if (this.guardedSettlement) {
        this.guardedSettledRecomputations++;
        if (this.recomputationGuardDecided) this.guardDecidedRecomputations++;
      }
      this.recordSettledRecomputation(key, entry[0]);
    }
    this.sweepRecomputeNanos += System.nanoTime() - recomputeStart;
    List<Object[]> writes = new ArrayList<>();
    for (int i = 0; i < values.length; i++) {
      Object key = targets.get(i);
      Object settled = this.state.get(key);
      Object recomputed = values[i];
      if (probeFilter != null && String.valueOf(key).contains(probeFilter))
        LOGGER.fine(() -> "CANON-PROBE recompute " + brief(key) + " := " + brief(recomputed));
      Object value = this.absorb(recomputed);
      int cause = (causes == null ? RecomputationCause.ITERATED : causes.get(key)).ordinal();
      if (this.guardedSettlement && this.joinSettled.contains(key)) {
        RecomputationOutcome outcome =
            value == null || value.equals(settled)
                ? RecomputationOutcome.SAME
                : RecomputationOutcome.KEPT;
        this.recomputationsByCause[cause][outcome.ordinal()]++;
        this.guardHeld++;
        if (outcome == RecomputationOutcome.KEPT) this.guardHeldDiffering++;
        continue;
      }
      if (value != null && this.joinSettled.contains(key)) {
        if (this.pinnedAtSettlement.contains(key)) value = settled;
        else if (this.widenedAtSettlement.contains(key))
          value = this.widen(key, this.generalizing(settled, value));
        else {
          Object joined = joinCovering(settled, value);
          value = this.widen(key, joined);
          if (!value.equals(joined)) this.pinnedAtSettlement.add(key);
        }
      } else if (growth != null && growth.makesShape(settled, value)) {
        // Not the value the query holds, which is no recomputation that made a shape when the
        // query's transfer gave its shapes up since, or handed another query's value on.
        Object grown = growth.grow(key, value, this::generalizing);
        if (growth.keepsGrowing(key)) {
          this.joinSettled.add(key);
          this.widenedAtSettlement.add(key);
          this.widenedQueries++;
          Object widened = this.widen(key, grown);
          LOGGER.fine(
              () ->
                  "Widening "
                      + key
                      + ", which keeps making new shapes: "
                      + settled
                      + " by "
                      + recomputed
                      + " gives "
                      + widened
                      + ".");
          value = widened;
        }
      }
      if (value == null || value.equals(settled)) {
        this.recomputationsByCause[cause][RecomputationOutcome.SAME.ordinal()]++;
        continue;
      }
      if (isBottomValue(value) && !isBottomValue(settled)) {
        this.recomputationsByCause[cause][RecomputationOutcome.KEPT.ordinal()]++;
        continue;
      }
      this.recomputationsByCause[cause][RecomputationOutcome.REPLACED.ordinal()]++;
      if (this.replacedBySettlement != null
          && Integer.valueOf(value.hashCode())
              .equals(this.replacedBySettlement.put(key, Objects.hashCode(settled))))
        this.restoredByCause[cause]++;
      this.putState(key, value);
      if (probeFilter != null && String.valueOf(key).contains(probeFilter)) {
        Object written = value;
        LOGGER.fine(() -> "CANON-PROBE write " + brief(key) + " := " + brief(written));
      }
      this.stamp(key, false);
      changed.add(key);
      writes.add(new Object[] {key, settled, value});
    }
    // Only now, so that every query whose recomputation made a shape in this step made a new one.
    if (growth != null) growth.note(writes);
    return writes;
  }

  /**
   * The shapes a component's queries hold during one {@link #iterate} call, and how many
   * recomputations of each query held a shape that none of them held. A value that only moves
   * around the component, or that its queries gather from one another, is made of shapes the
   * component holds. A shape it does not hold was made by the transfer of the query recomputed,
   * from what it read.
   *
   * <p>The shapes are those held now, and not every shape held since the iteration began. The solve
   * joined at every update, so it can leave a component that makes shapes holding the shapes it
   * will make again, and how many it leaves depends on the order the solve took the queries in. The
   * iteration replaces those values, and once no query holds a shape, a recomputation that holds it
   * made it.
   *
   * <p>A query that has made enough shapes is widened over the recomputations that made them, which
   * are kept here widened by one another, and not over the value it holds then. A transfer need not
   * be monotone, so that value can be the unknown shape the transfer gave its shapes up for, or a
   * value another query handed it, and neither says what the query made.
   */
  private static final class Growth {

    /** The number of the component's queries that hold each shape. */
    private final Map<List<Dimension<?>>, Integer> held = HashMapFactory.make();

    /** The recomputations of each query that held a shape the component did not hold. */
    private final Map<Object, Integer> made = HashMapFactory.make();

    /** The widening of those recomputations of each query, in the order the query made them. */
    private final Map<Object, Object> grown = HashMapFactory.make();

    /**
     * Notes a value a query of the component holds.
     *
     * @param value The value.
     */
    void hold(Object value) {
      if (value instanceof ShapeResult)
        for (List<Dimension<?>> member : ((ShapeResult) value).members())
          this.held.merge(member, 1, Integer::sum);
    }

    /**
     * Notes the writes of a step: each query holds the value written and no longer the one it held.
     *
     * @param writes The writes, each a query, the value it held and the value written.
     */
    void note(List<Object[]> writes) {
      for (Object[] write : writes) {
        if (write[1] instanceof ShapeResult)
          for (List<Dimension<?>> member : ((ShapeResult) write[1]).members())
            this.held.computeIfPresent(member, (shape, count) -> count == 1 ? null : count - 1);
        this.hold(write[2]);
      }
    }

    /**
     * Decides whether a recomputation holds a shape the component does not hold.
     *
     * @param settled The value the query holds.
     * @param value The value it recomputed to, or {@code null}.
     * @return Whether the recomputation made a shape.
     */
    boolean makesShape(Object settled, Object value) {
      if (!(settled instanceof ShapeResult) || !(value instanceof ShapeResult)) return false;
      return !this.held.keySet().containsAll(((ShapeResult) value).members());
    }

    /**
     * Counts a recomputation that made a shape, and widens the query's earlier ones by it.
     *
     * @param key The query.
     * @param value The value it recomputed to.
     * @param widening Widens a value by a later one.
     * @return The widening of the query's recomputations that made a shape.
     */
    Object grow(Object key, Object value, BinaryOperator<Object> widening) {
      this.made.merge(key, 1, Integer::sum);
      return this.grown.merge(key, value, widening);
    }

    /**
     * Decides whether a query has made enough shapes to be widened.
     *
     * @param key The query.
     * @return Whether {@link #GROWTH_WIDENING_LIMIT} of its recomputations made a shape.
     */
    boolean keepsGrowing(Object key) {
      return this.made.getOrDefault(key, 0) >= GROWTH_WIDENING_LIMIT;
    }
  }

  /**
   * Iterates a cyclic component to a fixpoint inside the pass that wrote it (wala/ML#1293). One
   * step leaves every reader of a written query holding a value computed from the value the step
   * replaced, so the next pass recomputes those readers, and a value that only moves around the
   * cycle, sustained by nothing outside it, moves one query a pass for as long as passes run. What
   * a query then holds depends on how many passes ran before it is read. Here each further step
   * recomputes the component's readers of the queries the step before wrote, until a step writes
   * nothing. Every recomputation then stands: each reader of a written query was recomputed after
   * the write, so the next pass skips the component until an evaluation or a write outside it moves
   * an input.
   *
   * <p>The steps need not reach a fixpoint by themselves, since a value that only moves around the
   * cycle returns the component to a state it was in. That is an orbit, and it is found exactly: a
   * fingerprint of the component's state names an earlier step to compare with, and the writes
   * logged since that step decide whether the state is the same. Each query the orbit wrote then
   * takes the meet of the values the orbit gave it, and the iteration goes on from there, with
   * those queries as targets too. A value the component holds for a reason outside the orbit is in
   * every state of it, and the meet keeps it. A meet over single queries or single passes cannot
   * tell such a value from one in transit, which is why the orbit is of the whole component. Where
   * the meet is ⊥ the query takes the join of its orbit's values: a settled value that is not ⊥
   * does not return to ⊥, which is the rule a recomputation that returns ⊥ follows and the one the
   * promotion rounds of {@link #canonicalize} end by.
   *
   * <p>The value at the end is sound, and that needs neither monotone transfers nor the meet. The
   * iteration ends at a step that writes nothing, and each query of the component then holds at
   * least what its transfer returns from the state: that value where the recomputation was written
   * or returned the settled value, and a value above it where a recomputation to ⊥ was kept out,
   * where a join was written, where the query is pinned at the unknown-marked element, and where it
   * was widened, since the widened value covers the recomputation ({@link ShapeResult#widening}).
   * The transfers therefore do not take the state above itself. If each transfer is sound for the
   * program, such a state holds every value the program gives the queries: the program's own
   * meaning is monotone, and its least fixpoint is below every state it does not take above itself.
   * The meet decides only which such state the iteration ends at, and how soon. For monotone
   * transfers, at an orbit where no query takes the join, the meet is such a state and the
   * iteration from it descends: the transfers take each state of the orbit to the next, so they
   * take the meet of the states to a value below every state of the orbit, which is below the meet.
   * Where a query takes the join, or a transfer is not monotone, the iteration can rise from the
   * orbit's values and find another orbit.
   *
   * <p>The fixpoint is of the component as the sweep condensed it. A query first sighted during the
   * component's steps is evaluated where it is read and is not among the readers recomputed here,
   * so a later step can rewrite a query it read. The next pass finds it on the cycle, with no
   * settled recomputation standing, and recomputes it. The census counts the queries first sighted
   * in this way, and those of them that read the component.
   *
   * <p>A value that only moves around the cycle is one case, and a value the cycle makes anew on
   * each turn is the other: a loop that pads or batches what it is given yields a longer axis or a
   * higher rank each time around, with no end, and neither a fixpoint nor an orbit. Stepping such a
   * component until a cap would leave its queries holding the shapes of as many turns as the cap
   * allows. So each step notes the queries whose recomputation holds a shape that no query of the
   * component holds then, and a query that has done so {@link #GROWTH_WIDENING_LIMIT} times is
   * widened from then on ({@link #widenedAtSettlement}): the axis that grows becomes a size no one
   * computed, and a rank that grows past the ranks a widened value may hold ({@link
   * ShapeResult#WIDENED_RANKS}) makes the whole shape unknown. A widened value covers what its
   * transfer returns from it, so the growth ends at that query, and the others go on being
   * replaced. Each such shape is made at some query, so a component cannot go on making them
   * without one of its queries reaching the limit. The search for an orbit begins anew at the step
   * that widens a query, since the component moves by another transfer from then on, and each query
   * is widened once, so that happens no more often than the component has queries. Without it, a
   * component whose rank grew would be found in an orbit that is none: the widened query holds the
   * unknown shape, its readers take it, and that is the state the solve left them in. The orbit's
   * values would be the ranks the growth had reached at the limit, and the join of them would stand
   * as the component's. The shapes the growth made before the limit are still in the component, and
   * a cycle within it that hands a value on as it is keeps them. They are the members the widened
   * shape covers, and each recomputation that holds that shape is written without them ({@link
   * #absorb}).
   *
   * <p>A shape counts as made when no query of the component holds it at that step, whatever the
   * component held before ({@link Growth}). The solve joined at every update, so it can leave a
   * component that makes shapes holding the shapes of the turns it took, which are the shapes the
   * component will make again, and how many it leaves depends on the order the solve took the
   * queries in. Counting against what the component held when the iteration began would widen it
   * under one order and leave it to gather shapes under another. The iteration replaces those
   * values, and a query that makes them again is counted.
   *
   * <p>A query at the limit is widened over its recomputations that made a shape, and not over the
   * value it holds. A transfer need not be monotone: an operation that reads what it can resolve of
   * an operand and gives up past a number of shapes returns the unknown shape from three shapes and
   * one shape from the unknown shape. Such a query can hold the unknown shape at the limit, and the
   * widening of that would stay unknown. A reader that takes what it can resolve would then take
   * the shapes the component's other queries still hold for all there is.
   *
   * <p>A component that makes no shape can fail to settle all the same: its queries gain and lose a
   * shape together, and return to a state they were in with each holding the meet of its orbit's
   * values. Its queries are then left to join, and the join keeps no member another covers ({@link
   * #joinCovering}), so the members a more general one stands for do not gather beside it.
   *
   * <p>The iteration is bounded in two parts. While it replaces, it makes at most a budget of
   * recomputations: past the budget, or at an orbit whose meet moves no query, the component's
   * queries enter {@link #joinSettled}, and this pass and every later one join their
   * recomputations. The joins are not counted against the budget. They end because a value then
   * only grows: a shape is written at most once for each member up to the widening cap, once for
   * its unknown remainder and once to be pinned past the cap, and a dtype set at most once for each
   * dtype. A write makes one recomputation for each of the query's readers in the component, so the
   * joins make at most that many writes times the component's read edges, which is quadratic in the
   * component's size where the budget is linear.
   *
   * @param scc The component.
   * @param targets The queries the pass's own step recomputed.
   * @param first The writes of that step.
   * @param changed The keys whose settled values changed; grown by the iteration.
   * @param probeFilter The text a probed query's name contains, or {@code null}.
   * @return The number of settled values the iteration wrote.
   */
  private int iterate(
      List<Object> scc,
      List<Object> targets,
      List<Object[]> first,
      Set<Object> changed,
      String probeFilter) {
    this.iteratedComponents++;
    this.largestIteratedComponent = Math.max(this.largestIteratedComponent, scc.size());
    Map<Object, Integer> positions = HashMapFactory.make();
    for (int i = 0; i < scc.size(); i++) positions.put(scc.get(i), i);
    long budget = this.iterationBudgetFactor * scc.size() + this.iterationBudgetBase;
    long spent = 0;
    boolean joining = false;
    // The steps since the state the orbit search began at, and the step count at each fingerprint.
    List<List<Object[]>> log = new ArrayList<>();
    Map<Long, Integer> seen = HashMapFactory.make();
    long print = 0;
    seen.put(print, 0);
    log.add(first);
    print += printDelta(first);
    seen.put(print, 1);
    Set<Object> recomputed = this.checkSkips ? HashSetFactory.make(targets) : null;
    Growth growth = new Growth();
    for (Object key : scc) growth.hold(this.state.get(key));
    // The writes whose readers are still to recompute, and the queries among them that hold an
    // orbit's value and not their own recomputation's.
    List<Object[]> last = first;
    Set<Object> imposed = Collections.emptySet();
    int replaced = 0;
    while (true) {
      List<Object> readers = this.readersWithin(last, positions, imposed);
      if (readers.isEmpty()) break;
      spent += readers.size();
      if (!joining && spent > budget) {
        joining = true;
        if (this.joinSettled.addAll(scc)) this.joinedComponents++;
        if (probeFilter != null)
          LOGGER.fine("CANON-PROBE a component of " + scc.size() + " joins past its budget.");
      }
      if (recomputed != null) recomputed.addAll(readers);
      long widened = this.widenedQueries;
      List<Object[]> writes = this.step(readers, null, changed, probeFilter, growth);
      this.iterationSteps++;
      replaced += writes.size();
      if (writes.isEmpty()) break;
      last = writes;
      imposed = Collections.emptySet();
      if (joining) continue;
      if (this.widenedQueries != widened) {
        // A query is widened from this step on, so the component moves by another transfer than
        // the one that made the states logged so far, and meeting one of them again is no orbit.
        // It does meet one where a rank grew: the query is widened to the unknown shape, which is
        // what the solve left it and its cycle holding when the iteration began.
        log = new ArrayList<>();
        seen = HashMapFactory.make();
        print = 0;
        seen.put(print, 0);
        continue;
      }
      log.add(writes);
      print += printDelta(writes);
      Integer earlier = seen.put(print, log.size());
      if (earlier == null) continue;
      Map<Object, List<Object>> orbit = this.orbit(log, earlier);
      if (orbit == null) continue;
      this.orbits++;
      List<Object[]> settled = this.settleOrbit(orbit, changed, probeFilter);
      if (settled.isEmpty()) {
        // Every query already holds its orbit's value, and still the orbit goes on.
        joining = true;
        if (this.joinSettled.addAll(scc)) this.joinedComponents++;
        if (probeFilter != null)
          LOGGER.fine("CANON-PROBE a component of " + scc.size() + " joins at an orbit.");
        continue;
      }
      replaced += settled.size();
      growth.note(settled);
      // The readers of the step that closed the orbit are still to recompute too.
      last = new ArrayList<>(writes);
      last.addAll(settled);
      imposed = HashSetFactory.make();
      for (Object[] write : settled) imposed.add(write[0]);
      log = new ArrayList<>();
      seen = HashMapFactory.make();
      print = 0;
      seen.put(print, 0);
    }
    // The check mode: at a fixpoint reached by replacement, each query the pass recomputed on the
    // component recomputes to the value it holds.
    if (recomputed != null && !joining)
      for (Object key : scc)
        if (recomputed.contains(key)
            && this.cyclicCause(key) == null
            && this.checkStanding(key, true)) this.checkedFixpointQueries++;
    return replaced;
  }

  /**
   * The targets of an iteration step: the component's readers of the queries the step before wrote,
   * and each written query that read itself or was given a value that is not its recomputation's,
   * in the component's order.
   *
   * @param last The writes of the step before.
   * @param positions The position of each of the component's queries.
   * @param imposed The written queries that hold an orbit's value and not their recomputation's.
   * @return The targets.
   */
  private List<Object> readersWithin(
      List<Object[]> last, Map<Object, Integer> positions, Set<Object> imposed) {
    Set<Object> found = HashSetFactory.make();
    for (Object[] write : last) {
      Object key = write[0];
      if (imposed.contains(key) || this.settledSelfReaders.contains(key)) found.add(key);
      for (Object reader : this.dependents.getOrDefault(key, Collections.emptySet()))
        if (positions.containsKey(reader)) found.add(reader);
    }
    List<Object> readers = new ArrayList<>(found);
    readers.sort(Comparator.comparing(positions::get));
    return readers;
  }

  /**
   * Decides whether a component's state is the state it had after an earlier step, and if so
   * returns the orbit between the two.
   *
   * @param log The writes of each step.
   * @param earlier The number of steps the earlier state follows.
   * @return The values each query was written since the earlier state, in the order written, or
   *     {@code null} when the two states differ.
   */
  private Map<Object, List<Object>> orbit(List<List<Object[]>> log, int earlier) {
    Map<Object, Object> before = HashMapFactory.make();
    Map<Object, List<Object>> values = new LinkedHashMap<>();
    for (List<Object[]> writes : log.subList(earlier, log.size()))
      for (Object[] write : writes) {
        if (!before.containsKey(write[0])) before.put(write[0], write[1]);
        values.computeIfAbsent(write[0], k -> new ArrayList<>()).add(write[2]);
      }
    for (Map.Entry<Object, Object> held : before.entrySet())
      if (!Objects.equals(held.getValue(), this.state.get(held.getKey()))) return null;
    return values;
  }

  /**
   * Gives each query of an orbit the meet of the values the orbit gave it, or their join where the
   * meet is ⊥.
   *
   * @param orbit The values each query was written, as {@link #orbit} returns them.
   * @param changed The keys whose settled values changed; grown by the writes.
   * @param probeFilter The text a probed query's name contains, or {@code null}.
   * @return The writes, each a query, the value it held and the value written.
   */
  private List<Object[]> settleOrbit(
      Map<Object, List<Object>> orbit, Set<Object> changed, String probeFilter) {
    List<Object[]> writes = new ArrayList<>();
    for (Map.Entry<Object, List<Object>> written : orbit.entrySet()) {
      Object key = written.getKey();
      Object met = null;
      Object joined = null;
      for (Object value : written.getValue()) {
        met = met == null ? value : meet(met, value);
        joined = join(joined, value);
      }
      boolean disjoint = isBottomValue(met);
      Object value = disjoint ? this.widen(key, joined) : met;
      Object settled = this.state.get(key);
      if (value.equals(settled)) continue;
      if (disjoint) this.orbitJoins++;
      else this.orbitMeets++;
      if (this.replacedBySettlement != null)
        this.replacedBySettlement.put(key, Objects.hashCode(settled));
      this.putState(key, value);
      if (probeFilter != null && String.valueOf(key).contains(probeFilter)) {
        LOGGER.fine(
            () ->
                "CANON-PROBE orbit "
                    + (disjoint ? "join " : "meet ")
                    + brief(key)
                    + " := "
                    + brief(value));
      }
      this.stamp(key, false);
      changed.add(key);
      writes.add(new Object[] {key, settled, value});
    }
    return writes;
  }

  /**
   * The change a step's writes make to the fingerprint of a component's state: a sum over the
   * queries, so that it is the same for the same state however the state was reached.
   *
   * @param writes The writes, each a query, the value it held and the value written.
   * @return The change.
   */
  private static long printDelta(List<Object[]> writes) {
    long delta = 0;
    for (Object[] write : writes) delta += print(write[0], write[2]) - print(write[0], write[1]);
    return delta;
  }

  /**
   * A query's term of the fingerprint of a component's state.
   *
   * @param key The query.
   * @param value The value it holds.
   * @return The term.
   */
  private static long print(Object key, Object value) {
    long mixed = ((long) key.hashCode() << 32) ^ (Objects.hashCode(value) & 0xffffffffL);
    mixed ^= mixed >>> 33;
    mixed *= 0xff51afd7ed558ccdL;
    mixed ^= mixed >>> 33;
    mixed *= 0xc4ceb9fe1a85ec53L;
    return mixed ^ (mixed >>> 33);
  }

  /**
   * Recomputes a query's transfer once against the settled state for {@link #canonicalize}.
   *
   * @param key The query to recompute.
   * @return The recomputed (widened) value, or the settled value when the transfer is unknown or
   *     throws.
   */
  private Object recomputeSettled(Object key) {
    Supplier<Object> transfer = this.transfers.get(key);
    if (transfer == null) return this.state.get(key);
    Map<Object, Integer> records = this.readVersions.get(key);
    if (records != null) records.clear();
    Object result;
    this.recomputationThrew = false;
    this.selfReadInRecomputation = false;
    long decisions = TensorGenerator.guardDecisions();
    this.evaluating.push(key);
    this.inStack.add(key);
    try {
      result = transfer.get();
    } catch (IllegalArgumentException e) {
      LOGGER.log(
          Level.FINE, e, () -> "Canonicalization IAE for " + key + "; keeping the settled value.");
      this.recomputationThrew = true;
      return this.state.get(key);
    } finally {
      this.evaluating.pop();
      this.inStack.remove(key);
      this.recomputationGuardDecided = TensorGenerator.guardDecisions() != decisions;
    }
    if (result == null) result = this.shapeKinds.get(key) ? ShapeResult.unknown() : NULL_DTYPE;
    return this.widen(key, result);
  }

  /**
   * Records the stamps of a state write: every write moves {@link #stamps}, and a write other than
   * the settlement pass's write-back of the query's own recomputation also moves {@link
   * #selfStamps}.
   *
   * @param key The query written.
   * @param self Whether the write is an evaluation or a promotion, not a settlement write-back.
   */
  private void stamp(Object key, boolean self) {
    long now = ++this.clock;
    this.stamps.put(key, now);
    if (self) this.selfStamps.put(key, now);
    this.markDirty(key);
    for (Object reader : this.settledReaders.getOrDefault(key, Collections.emptySet()))
      this.markDirty(reader);
  }

  /**
   * Registers a first-sighted query.
   *
   * @param key The query.
   * @param transfer Its transfer.
   * @param shapeKind Whether it is shape-valued.
   */
  private void register(Object key, Supplier<Object> transfer, boolean shapeKind) {
    this.transfers.put(key, transfer);
    if (this.sightedInSteps != null) this.sightedInSteps.add(key);
    this.shapeKinds.put(key, shapeKind);
    // No settled recomputation stands for a new query.
    this.markDirty(key);
    if (shapeKind) {
      this.shapeBottoms.add(key);
      this.newShapeBottoms.add(key);
    }
  }

  /**
   * Writes a query's state, keeping {@link #shapeBottoms} and {@link #newShapeBottoms}.
   *
   * @param key The query.
   * @param value Its new value.
   */
  private void putState(Object key, Object value) {
    this.state.put(key, value);
    if (Boolean.TRUE.equals(this.shapeKinds.get(key))) {
      if (value instanceof ShapeResult && ((ShapeResult) value).isBottom()) {
        if (this.shapeBottoms.add(key)) this.newShapeBottoms.add(key);
      } else this.shapeBottoms.remove(key);
    }
  }

  /**
   * Marks a query as one whose settlement skip may no longer hold.
   *
   * @param key The query.
   */
  private void markDirty(Object key) {
    this.settleDirty.add(key);
    if (this.everCyclic.contains(key)) this.dirtyCyclic.add(key);
  }

  /**
   * Records a cyclic component found by a Tarjan pass. Under the check mode, a restricted pass also
   * walks the whole graph, so it sees components outside its region; it records none of them, as
   * the restricted pass alone would not see them. A full sweep's region holds every reader that
   * gained an edge since the last full sweep, so a cyclic component outside it was already cyclic,
   * and recorded, at that sweep; the check mode asserts so.
   *
   * @param scc The component.
   * @param region The region of the restricted pass in progress, or {@code null}.
   * @param recorded Whether a component outside the region must already be recorded.
   */
  private void noteCyclic(List<Object> scc, Set<Object> region, boolean recorded) {
    if (region != null && !region.contains(scc.get(0))) {
      if (recorded && !this.everCyclic.containsAll(scc))
        throw new AssertionError(
            "A cyclic component outside the region was not recorded: " + scc + ".");
      return;
    }
    for (Object key : scc)
      if (this.everCyclic.add(key) && this.settleDirty.contains(key)) this.dirtyCyclic.add(key);
  }

  /**
   * The closure of the given queries under {@link #dependents}: every query that reads one of them,
   * transitively. Closed under readers, it holds every component of each of its members whole.
   *
   * @param seeds The queries to close.
   * @return The closure.
   */
  private Set<Object> readerClosure(Set<Object> seeds) {
    Set<Object> region = HashSetFactory.make();
    Deque<Object> work = new ArrayDeque<>();
    for (Object seed : seeds) if (region.add(seed)) work.push(seed);
    while (!work.isEmpty())
      for (Object reader : this.dependents.getOrDefault(work.pop(), Collections.emptySet()))
        if (region.add(reader)) work.push(reader);
    return region;
  }

  /**
   * Checks the restriction against the whole-graph condensation: each component lies wholly inside
   * or outside the region, and the region's components are the restricted pass's, in its order.
   *
   * @param whole The whole-graph condensation.
   * @param region The region.
   */
  private void checkRestriction(List<List<Object>> whole, Set<Object> region) {
    List<List<Object>> inside = new ArrayList<>();
    for (List<Object> scc : whole) {
      long in = scc.stream().filter(region::contains).count();
      if (in != 0 && in != scc.size())
        throw new AssertionError("A component straddles the settlement region: " + scc + ".");
      if (in != 0) inside.add(scc);
    }
    if (!inside.equals(this.tarjan(region)))
      throw new AssertionError("The restricted condensation differs from the whole one's order.");
  }

  /**
   * Records a settled recomputation of a query for the next pass's skip decision, or forgets the
   * previous one when this one cannot stand for a recomputation: it caught an exception, deferred
   * under a re-entered fold (its value then depends on the promotion that follows), or ran inside a
   * generator guard that decided what it did. A guard's frames are state a transfer may read beside
   * its queries, and a frame that was open around the pass is gone by the next. A recomputation
   * that no guard decided read none of it ({@link TensorGenerator#guardDecisions}), so it is a
   * function of the values its reads returned, inside a guard as outside one. A recomputation of a
   * query of {@link #joinSettled} inside a guard leaves no record either: {@link #step} does not
   * write it, so the query does not hold what the record would stand for.
   *
   * @param key The query recomputed.
   * @param entry The clock at the entry of the attempt that completed.
   */
  private void recordSettledRecomputation(Object key, long entry) {
    // The next pass decides afresh whether this recomputation stands.
    this.markDirty(key);
    if (this.recomputationThrew
        || (this.guardedSettlement
            && (this.recomputationGuardDecided || this.joinSettled.contains(key)))
        || this.reentryDeferredAtSettlement.contains(key)) {
      this.settledAt.remove(key);
      this.settledReads.remove(key);
      return;
    }
    this.settledAt.put(key, entry);
    if (this.selfReadInRecomputation) this.settledSelfReaders.add(key);
    else this.settledSelfReaders.remove(key);
    Map<Object, Integer> records = this.readVersions.get(key);
    this.settledReads.put(
        key, records == null ? Collections.emptySet() : new HashSet<>(records.keySet()));
    for (Object read : this.settledReads.get(key))
      this.settledReaders.computeIfAbsent(read, k -> HashSetFactory.make()).add(key);
  }

  /**
   * Finds why the settlement pass in progress recomputes a query on a cycle, or that it may skip
   * the query: its last settled recomputation stands, nothing but the pass's own write-back of that
   * recomputation has written the query since, and none of the queries that recomputation read has
   * been written since it began. A transfer is a function of the values its reads return, given the
   * final call graph and pointer analysis and the generators' write-once caches, so the
   * recomputation would return the value the query settled at. That holds for a pass inside a
   * generator guard too, since a record stands only for a recomputation no guard around its pass
   * decided ({@link #recordSettledRecomputation}). There a recomputation made again could be cut at
   * a frame that is open around this pass, and the skip keeps the value the query settled at with
   * no such cut.
   *
   * @param key The query.
   * @return The cause, or {@code null} when the query's recomputation can be skipped.
   */
  private RecomputationCause cyclicCause(Object key) {
    if (!this.skipEnabled) return RecomputationCause.SKIP_OFF;
    Long at = this.settledAt.get(key);
    if (at == null) return RecomputationCause.NO_RECORD;
    // A query that read itself had its own write-back as an input; any other query only its writes
    // by an evaluation or a promotion.
    Map<Object, Long> own = this.settledSelfReaders.contains(key) ? this.stamps : this.selfStamps;
    if (own.getOrDefault(key, 0L) > at) return RecomputationCause.OWN_WRITE;
    boolean written = false;
    for (Object read : this.settledReads.get(key)) {
      if (this.stamps.getOrDefault(read, 0L) <= at) continue;
      // Every write moves `stamps`, and a write other than a settlement write-back `selfStamps`.
      if (this.selfStamps.getOrDefault(read, 0L) > at) return RecomputationCause.READ_EVALUATED;
      written = true;
    }
    return written ? RecomputationCause.READ_SETTLED : null;
  }

  /**
   * The check mode: recomputes a skipped query without writing anything back and fails if the
   * recomputation would have replaced the settled value, wrote any state or deferred. Inside a
   * generator guard, a recomputation that a guard decided is not the one the skip stands for: it
   * can be cut at a frame that was open around the pass. Its value and its deferral are then not
   * checked, and a state write still fails.
   *
   * @param key The skipped query.
   * @throws AssertionError if the recomputation differs from the settled value.
   */
  private void checkSkip(Object key) {
    if (this.checkStanding(key, false)) this.checkedSkips++;
    else this.guardDecidedCheckedSkips++;
  }

  /**
   * Recomputes a query whose settled value the pass in progress lets stand, without writing
   * anything back, and fails if the recomputation would have replaced the value, wrote any state or
   * deferred. A query of {@link #joinSettled} holds a join of its recomputations, so its
   * recomputation may be any value the join already holds, or that the widened value covers for a
   * query of {@link #widenedAtSettlement}.
   *
   * @param key The query.
   * @param iterated Whether the value stands at the fixpoint of {@link #iterate}, and not by the
   *     skip.
   * @return Whether the value was compared: {@code false} for a recomputation a guard decided.
   * @throws AssertionError if the recomputation differs from the settled value.
   */
  private boolean checkStanding(Object key, boolean iterated) {
    String what =
        iterated ? "A settled recomputation at a fixpoint" : "A skipped settled recomputation";
    Map<Object, Integer> records = this.readVersions.get(key);
    Map<Object, Integer> saved = records == null ? null : HashMapFactory.make(records);
    long before = this.clock;
    boolean deferredBefore = this.reentryDeferredAtSettlement.contains(key);
    Object value = this.absorb(this.unwinding(() -> this.recomputeSettled(key)));
    boolean decided = this.guardedSettlement && this.recomputationGuardDecided;
    if (saved != null) {
      this.readVersions.put(key, saved);
      this.staleCandidates.add(key);
    }
    // The pass did not recompute the query, so a deferral only the check noted is not the pass's.
    if (decided && !deferredBefore) this.reentryDeferredAtSettlement.remove(key);
    // A recomputation the skip stands for writes nothing and defers nothing; one that does breaks
    // the skip's premise even when its value passes the comparison below.
    if (this.clock != before || (!deferredBefore && this.reentryDeferredAtSettlement.contains(key)))
      throw new AssertionError(what + " of " + key + " wrote state or deferred.");
    if (decided) return false;
    Object settled = this.state.get(key);
    if (value != null
        && !value.equals(settled)
        && !(isBottomValue(value) && !isBottomValue(settled))
        && !(this.joinSettled.contains(key)
            && (this.pinnedAtSettlement.contains(key)
                || (this.widenedAtSettlement.contains(key)
                        ? widening(settled, value)
                        : joinCovering(settled, value))
                    .equals(
                        this.widenedAtSettlement.contains(key)
                            ? settled
                            : joinCovering(settled, settled)))))
      throw new AssertionError(
          what
              + " differs: "
              + key
              + " settled at "
              + brief(settled)
              + " but recomputes to "
              + brief(value)
              + ".");
    return true;
  }

  /**
   * Returns how many staleness scans found a stale reader.
   *
   * @return The count.
   */
  int staleScanHits() {
    return this.staleScanHits;
  }

  /**
   * Rewinds the version a reader recorded for a value by one, as if the reader had consumed the
   * value before its latest change, without making the reader a staleness candidate. For a test of
   * {@link #enqueueStaleReaders}'s check mode: no scheduling reaches this state.
   *
   * @param reader The reader.
   * @param key The value it read.
   */
  void lagRecordForTest(Object reader, Object key) {
    this.readVersions.get(reader).merge(key, -1, Integer::sum);
  }

  /**
   * Stamps a query as an evaluation's write does, without marking it or its settled readers as
   * queries whose skip may no longer hold. For a test of the check mode's walk outside a restricted
   * pass's region: no scheduling reaches this state.
   *
   * @param key The query.
   */
  void stampUnmarkedForTest(Object key) {
    long now = ++this.clock;
    this.stamps.put(key, now);
    this.selfStamps.put(key, now);
  }

  /**
   * Bumps a value's version without re-enqueueing its readers, as a change made while they are on
   * the stack does, so only the staleness scan can find them. For a test of {@link
   * #enqueueStaleReaders}.
   *
   * @param key The value.
   */
  void bumpVersionForTest(Object key) {
    this.bumpVersion(key);
  }

  /** Turns on the check mode of {@link #CHECK_SKIPS_PROPERTY} for this engine. */
  void enableSkipCheck() {
    this.checkSkips = true;
  }

  /** Turns off the settlement pass's skip for this engine, whatever the property says. */
  void disableSkip() {
    this.skipEnabled = false;
  }

  /** Turns off the settlement pass's iteration for this engine, whatever the property says. */
  void disableIteration() {
    this.iterationEnabled = false;
  }

  /**
   * Sets the recomputations the iteration may make on one component in one pass before it joins.
   *
   * @param factor The recomputations for each of the component's queries.
   * @param base The recomputations beyond those.
   */
  void setIterationBudget(long factor, long base) {
    this.iterationBudgetFactor = factor;
    this.iterationBudgetBase = base;
  }

  /**
   * Returns the steps the iteration made beyond each pass's own.
   *
   * @return The number of steps.
   */
  long iterationSteps() {
    return this.iterationSteps;
  }

  /**
   * Returns how many orbits the iteration found.
   *
   * @return The number of orbits.
   */
  long orbits() {
    return this.orbits;
  }

  /**
   * Returns how many values the iteration set to an orbit's meet.
   *
   * @return The number of values.
   */
  long orbitMeets() {
    return this.orbitMeets;
  }

  /**
   * Returns how many values the iteration set to an orbit's join.
   *
   * @return The number of values.
   */
  long orbitJoins() {
    return this.orbitJoins;
  }

  /**
   * Returns how many components the iteration left to join.
   *
   * @return The number of components.
   */
  long joinedComponents() {
    return this.joinedComponents;
  }

  /**
   * Returns how many queries the iteration widened.
   *
   * @return The number of queries.
   */
  long widenedQueries() {
    return this.widenedQueries;
  }

  /**
   * Returns how many queries were first sighted during an iterated component's steps.
   *
   * @return The number of queries.
   */
  long sightedWhileIterating() {
    return this.sightedWhileIterating;
  }

  /**
   * Returns how many of the queries first sighted during an iterated component's steps read the
   * component, directly or through another of them.
   *
   * @return The number of queries.
   */
  long sightedOnCycle() {
    return this.sightedOnCycle;
  }

  /**
   * Returns how many recomputations a pass inside a generator guard made of a query left to join,
   * none of which it wrote.
   *
   * @return The number of recomputations.
   */
  long guardHeld() {
    return this.guardHeld;
  }

  /**
   * Returns how many of the recomputations a pass inside a generator guard made of a query left to
   * join differ from the settled value the query keeps.
   *
   * @return The number of recomputations.
   */
  long guardHeldDiffering() {
    return this.guardHeldDiffering;
  }

  /**
   * Turns off the check mode for this engine, whatever {@link #CHECK_SKIPS_PROPERTY} says, for a
   * test of the skip itself.
   */
  void disableSkipCheck() {
    this.checkSkips = false;
  }

  /**
   * Sets whether the settlement census counts restored values for this engine, whatever is logged.
   *
   * @param count Whether to count them.
   */
  void countRestoredValues(boolean count) {
    this.replacedBySettlement = count ? HashMapFactory.make() : null;
  }

  /**
   * Returns how many settled recomputations had a cause and an outcome.
   *
   * @param cause The cause.
   * @param outcome The outcome.
   * @return The number of such recomputations.
   */
  long recomputations(RecomputationCause cause, RecomputationOutcome outcome) {
    return this.recomputationsByCause[cause.ordinal()][outcome.ordinal()];
  }

  /**
   * Returns how many replacements with a cause gave a query back the value its previous replacement
   * by a settlement pass had replaced.
   *
   * @param cause The cause.
   * @return The number of such replacements.
   */
  long restored(RecomputationCause cause) {
    return this.restoredByCause[cause.ordinal()];
  }

  /**
   * Returns how many skipped queries the check mode verified.
   *
   * @return The number of checked skips.
   */
  long checkedSkips() {
    return this.checkedSkips;
  }

  /**
   * Returns how many skipped queries the check mode recomputed inside a generator guard and did not
   * compare, because a guard decided what the recomputation did.
   *
   * @return The number of such skips.
   */
  long guardDecidedCheckedSkips() {
    return this.guardDecidedCheckedSkips;
  }

  /**
   * Returns how many settled recomputations made by passes inside a generator guard a guard
   * decided. Such a recomputation leaves no record.
   *
   * @return The number of such recomputations.
   */
  long guardDecidedRecomputations() {
    return this.guardDecidedRecomputations;
  }

  /**
   * Returns how many queries the settlement passes inside a generator guard skipped.
   *
   * @return The number of such skips.
   */
  long guardedSettledSkips() {
    return this.guardedSettledSkips;
  }

  /**
   * Returns the reads of a query's last settled recomputation, for tests.
   *
   * @param key The query.
   * @return The settled reads, empty if none stand.
   */
  Set<Object> settledReadsOf(Object key) {
    return this.settledReads.getOrDefault(key, Collections.emptySet());
  }

  /**
   * Evaluates a query outside settlement, as a later fixpoint would, for tests.
   *
   * @param key The query.
   */
  void evaluateForTest(Object key) {
    this.evaluate(key);
  }

  /**
   * Decides whether a state value is the ⊥ of its kind.
   *
   * @param value The state value, {@code null} meaning the iteration bottom.
   * @return {@code true} iff the value reads as "not a tensor."
   */
  private static boolean isBottomValue(Object value) {
    if (value == null) return true;
    if (value == NULL_DTYPE) return false;
    if (value instanceof ShapeResult) return ((ShapeResult) value).isBottom();
    if (value instanceof Set) return ((Set<?>) value).isEmpty();
    return false;
  }

  /**
   * Returns whether the post-settlement replay diagnostic is running, i.e. the per-member replay
   * logs in the aggregation bodies should fire.
   *
   * @return {@code true} iff inside {@link #replaySettled}.
   */
  boolean isReplaying() {
    return this.replaying;
  }

  /**
   * Diagnostic (wala/ML#753): re-runs the transfer of every settled pure-⊤ shape query (no members,
   * unknown remainder) whose key string contains any of the filter's semicolon-separated
   * substrings, logging the recomputed value against the settled one. A segment prefixed with
   * {@code value:} instead matches against the settled value's rendering and selects any shape
   * query regardless of its settled value, so a resolved-but-suspect member population (e.g. a
   * spurious self-consistent fixpoint's) can be replayed to its deriving transfer. The replay runs
   * after every demand has settled, so each nested read returns a final value and the recomputation
   * observes the aggregation's per-member results free of the evaluation-order effects an in-flight
   * probe would perturb; the recomputed values are logged, never written back.
   *
   * @param filter Semicolon-separated substrings selecting the queries to replay: plain segments
   *     match pure-⊤ queries by key string, {@code value:}-prefixed segments match any shape query
   *     by value string. The separator is a semicolon because dimension renderings contain commas.
   */
  void replaySettled(String filter) {
    // Blank segments (a stray separator or whitespace) would contain-match every key and replay
    // the whole pure-⊤ population; drop them.
    List<String> keyAlternatives = new ArrayList<>();
    List<String> valueAlternatives = new ArrayList<>();
    for (String alternative : filter.split(";")) {
      String trimmed = alternative.trim();
      if (trimmed.isEmpty()) continue;
      if (trimmed.startsWith("value:")) {
        String pattern = trimmed.substring("value:".length()).trim();
        if (!pattern.isEmpty()) valueAlternatives.add(pattern);
      } else keyAlternatives.add(trimmed);
    }
    if (keyAlternatives.isEmpty() && valueAlternatives.isEmpty()) return;
    List<Object> marked = new ArrayList<>();
    for (Map.Entry<Object, Object> entry : this.state.entrySet()) {
      Object value = entry.getValue();
      if (!(value instanceof ShapeResult)) continue;
      ShapeResult shapes = (ShapeResult) value;
      boolean selected = false;
      if (shapes.members().isEmpty() && shapes.hasUnknown()) {
        String keyString = String.valueOf(entry.getKey());
        for (String alternative : keyAlternatives)
          if (keyString.contains(alternative)) {
            selected = true;
            break;
          }
      }
      if (!selected && !valueAlternatives.isEmpty()) {
        String valueString = String.valueOf(value);
        for (String alternative : valueAlternatives)
          if (valueString.contains(alternative)) {
            selected = true;
            break;
          }
      }
      if (selected) marked.add(entry.getKey());
    }
    LOGGER.fine(() -> "REPLAY sweeping " + marked.size() + " settled shape queries: " + filter);
    this.replaying = true;
    try {
      for (Object key : marked) {
        Supplier<Object> transfer = this.transfers.get(key);
        if (transfer == null) continue;
        LOGGER.fine(() -> "REPLAY BEGIN " + brief(key));
        Object result;
        this.evaluating.push(key);
        this.inStack.add(key);
        try {
          result = this.unwinding(transfer);
        } catch (IllegalArgumentException e) {
          LOGGER.log(Level.FINE, e, () -> "REPLAY IAE for " + brief(key) + ".");
          continue;
        } finally {
          this.evaluating.pop();
          this.inStack.remove(key);
        }
        // The sweep selects only shape-kind queries; mirror evaluate's null normalization.
        Object recomputed = result == null ? ShapeResult.unknown() : result;
        Object settled = this.state.get(key);
        LOGGER.fine(
            () ->
                "REPLAY END "
                    + brief(key)
                    + " := "
                    + brief(recomputed)
                    + " (settled := "
                    + brief(settled)
                    + ")");
      }
    } finally {
      this.replaying = false;
    }
  }

  /**
   * Diagnostic: for queries whose key string contains the {@code ariadne.typeResolution.dumpFilter}
   * property value, logs the stabilized value and each dependency with its value. The engine's
   * dependency edges plus state are the ground truth a per-query diagnosis needs, with no cache or
   * ordering noise.
   */
  private void dumpFiltered() {
    String filter = System.getProperty("ariadne.typeResolution.dumpFilter");
    if (filter == null) filter = System.getenv("ARIADNE_WORKLIST_DUMP");
    if (filter == null) return;
    for (Map.Entry<Object, Object> entry : this.state.entrySet()) {
      String keyString = String.valueOf(entry.getKey());
      if (!keyString.contains(filter)) continue;
      LOGGER.fine(() -> "DUMP " + brief(entry.getKey()) + " := " + brief(entry.getValue()));
      for (Object dep : this.reads.getOrDefault(entry.getKey(), Collections.emptySet()))
        LOGGER.fine(() -> "DUMP     reads " + brief(dep) + " := " + brief(this.state.get(dep)));
    }
  }

  /**
   * Names where a caught exception came from: its throwing frame, and the first frame outside the
   * throwing class, since a shared helper's own line (the allocation-site extractor's throw, say)
   * names the mechanism but not the caller that has to change (wala/ML#925).
   *
   * @param e The caught exception.
   * @return {@code "<throwing frame> called from <first frame in another class>"}.
   */
  private static String thrownFrom(Throwable e) {
    StackTraceElement[] frames = e.getStackTrace();
    if (frames.length == 0) return "an unknown frame";
    String thrower = frames[0].getClassName();
    for (StackTraceElement frame : frames)
      if (!frame.getClassName().equals(thrower)) return frames[0] + " called from " + frame;
    return frames[0].toString();
  }

  /**
   * Renders a key or value for the diagnostic dump. The interesting record components of a query
   * key (the value number and the exactness mode) trail its node's rendering, whose nested calling
   * contexts exceed any reasonable truncation length, so they are hoisted in front of the truncated
   * remainder; without this, every dumped query reads as an anonymous node prefix (wala/ML#753's
   * localization sessions).
   *
   * @param o The key or value to render.
   * @return The rendering.
   */
  private static String brief(Object o) {
    String s = String.valueOf(o);
    int vnAt = s.lastIndexOf("valueNumber=");
    String prefix = "";
    if (vnAt >= 0) {
      int end = s.indexOf(']', vnAt);
      prefix = "{" + (end > vnAt ? s.substring(vnAt, end) : s.substring(vnAt)) + "} ";
    }
    return prefix + (s.length() > 400 ? s.substring(0, 400) : s);
  }

  /**
   * Thrown by {@link #read} when a first-sighted query would nest one evaluation deeper than {@link
   * #INLINE_DEPTH_BOUND}, to unwind the evaluations in progress to the nearest {@link #unwinding}
   * caller. It carries the query and the chain of evaluations above it, which that caller evaluates
   * deepest first, each from its own depth, before retrying. No reader sees a value of a query that
   * has not been evaluated, so on an acyclic chain the values are those an unbounded stack would
   * give. The unwound chain stays in progress while it is pending ({@link #unwindPending}), so a
   * cycle that crosses the bound closes where it would on an unbounded stack: a read of a pending
   * member is a cycle read of its current value. Were the pending members read as not in progress,
   * each retry of a cycle longer than the bound would start from its deepest member and walk the
   * bound's length around it before unwinding again, never reaching a member in progress. A
   * discarded evaluation is rerun, so its side effects repeat: the census's count of evaluations
   * floored by a caught exception, and its log lines, can count one query more than once on an
   * analysis that unwinds.
   *
   * <p>An {@link Error}, so that a transfer catching {@link Exception} cannot swallow it, and
   * without a stack trace, which would cost as much as the stack it unwinds.
   */
  private static final class DepthUnwind extends Error {

    private static final long serialVersionUID = 1L;

    /** The query whose evaluation would have exceeded the bound, then the chain, deepest first. */
    private final transient List<Object> pending;

    /**
     * @param query The first-sighted query being read.
     * @param evaluating The evaluations in progress, innermost first.
     */
    private DepthUnwind(Object query, Deque<Object> evaluating) {
      super(null, null, false, false);
      this.pending = new ArrayList<>(evaluating.size() + 1);
      this.pending.add(query);
      this.pending.addAll(evaluating);
    }
  }

  /**
   * Runs an action from the current stack depth, absorbing any {@link DepthUnwind} thrown above it:
   * the unwound chain's queries are evaluated deepest first from this depth, skipping those already
   * evaluated or still in progress below this frame, and the action is retried. A chain that
   * unwinds again pushes its own queries above the rest, so the pending queries are a stack held
   * here rather than on the Java stack. Each unwind leaves at least its first-sighted query to be
   * evaluated before the retry, so the action makes progress.
   *
   * @param <T> The action's result type.
   * @param action The action.
   * @return The action's result.
   */
  private <T> T unwinding(Supplier<T> action) {
    Deque<Object> pending = new ArrayDeque<>();
    try {
      while (true) {
        while (!pending.isEmpty()) {
          Object query = pending.peek();
          if (this.evaluated.contains(query) || this.inStack.contains(query)) {
            this.popPending(pending);
            continue;
          }
          try {
            this.evaluate(query);
            this.popPending(pending);
          } catch (DepthUnwind unwind) {
            this.pushPending(pending, unwind);
          }
        }
        try {
          return action.get();
        } catch (DepthUnwind unwind) {
          this.pushPending(pending, unwind);
        }
      }
    } finally {
      while (!pending.isEmpty()) this.popPending(pending);
    }
  }

  /**
   * Pushes an unwound chain onto a pending stack, outermost first, so that its query is on top.
   *
   * @param pending The pending stack.
   * @param unwind The unwind carrying the chain.
   */
  private void pushPending(Deque<Object> pending, DepthUnwind unwind) {
    for (int i = unwind.pending.size() - 1; i >= 0; i--) {
      Object query = unwind.pending.get(i);
      pending.push(query);
      this.unwindPending.merge(query, 1, Integer::sum);
    }
  }

  /**
   * Pops the top of a pending stack.
   *
   * @param pending The pending stack.
   */
  private void popPending(Deque<Object> pending) {
    this.unwindPending.computeIfPresent(pending.pop(), (k, n) -> n == 1 ? null : n - 1);
  }

  private void iterate() {
    while (!this.worklist.isEmpty()) {
      Object key = this.worklist.poll();
      this.enqueued.remove(key);
      int count = this.evaluationCounts.merge(key, 1, Integer::sum);
      // Ascent through the finite lattice bounds re-evaluations; a runaway count means a
      // non-monotone transfer or an unbounded value, so log the offender and stop ascending it.
      if (count > EVALUATION_CAP) {
        if (count == EVALUATION_CAP + 1)
          LOGGER.warning(
              "Worklist evaluation cap hit for "
                  + key
                  + "; pinning at its current value. Investigate the transfer's monotonicity.");
        continue;
      }
      this.unwinding(
          () -> {
            this.evaluate(key);
            return null;
          });
    }
  }

  private void evaluate(Object key) {
    Supplier<Object> transfer = this.transfers.get(key);
    if (transfer == null) return;
    if (this.checkedSweepRegion != null
        && this.reads.containsKey(key)
        && !this.checkedSweepRegion.contains(key))
      throw new AssertionError("A restricted sweep evaluates a query outside its region: " + key);
    // The consumed-version records reflect the latest evaluation only: a value the last run did
    // not read cannot stale this run's result (wala/ML#758's exactness over the edge heuristic).
    Map<Object, Integer> records = this.readVersions.get(key);
    if (records != null) records.clear();
    Object result;
    boolean unwound = false;
    this.evaluating.push(key);
    this.inStack.add(key);
    try {
      result = transfer.get();
    } catch (DepthUnwind e) {
      // An unwound evaluation leaves no value and stays unevaluated; it reruns once the chain under
      // it has been evaluated.
      unwound = true;
      throw e;
    } catch (IllegalArgumentException e) {
      // Demand-driven callers catch this variously; under the engine the conservative reading is
      // an unknown value of the query's kind.
      LOGGER.log(
          Level.FINE,
          e,
          () ->
              "Worklist transfer IAE for "
                  + key
                  + " ("
                  + e.getMessage()
                  + " at "
                  + thrownFrom(e)
                  + "); treating as ⊤.");
      result = this.shapeKinds.get(key) ? ShapeResult.unknown() : EnumSet.of(DType.UNKNOWN);
      this.caughtQueryExceptions++;
    } finally {
      this.evaluating.pop();
      this.inStack.remove(key);
      if (!unwound) this.evaluated.add(key);
    }
    // The legacy dtype convention allows a null transfer result meaning ⊤, and CALLERS null-check
    // it to run fallback arms. The state table cannot hold null, so store the sentinel and let
    // read/demand translate it back to null: normalizing to EnumSet.of(UNKNOWN) instead hides the
    // null from the reader's fallback arm and composes spurious unknown-dtype members (the
    // flag-on twins caught by testReshape2).
    if (result == null) result = this.shapeKinds.get(key) ? ShapeResult.unknown() : NULL_DTYPE;
    Object old = this.state.get(key);
    Object joined = join(old, result);
    joined = this.absorb(this.widen(key, joined));
    if (!joined.equals(old)) {
      this.putState(key, joined);
      this.bumpVersion(key);
      this.stamp(key, true);
      this.reEnqueueStaleDependents(key);
    }
  }

  /**
   * Re-enqueues the dependents whose latest evaluation consumed the key's just-grown value
   * (wala/ML#758's read-version staleness criterion, replacing the every-edge heuristic): a
   * dependent with no consumed-version record for the key did not read it in its latest run — its
   * first inline read (recorded after this update returns) or a run that took another arm — so its
   * result cannot depend on the growth. A consuming on-stack reader cannot rerun now, but it is
   * marked stale and {@link #enqueueStaleReaders} re-enqueues it once it pops (its recorded version
   * lags), which closes the sparing hole of the heuristic this criterion subsumes. A key still
   * re-enqueues itself: a changed self-loop needs another pass to stabilize.
   *
   * @param key The query whose value just changed (its version already bumped).
   */
  private void reEnqueueStaleDependents(Object key) {
    for (Object dependent :
        this.orderDependentsForReEnqueue(
            this.dependents.getOrDefault(key, Collections.emptySet()))) {
      if (dependent.equals(key)) {
        this.enqueue(dependent);
        continue;
      }
      Map<Object, Integer> records = this.readVersions.get(dependent);
      if (records == null || !records.containsKey(key)) continue;
      this.staleReaders.add(dependent);
      if (!this.inStack.contains(dependent)) this.enqueue(dependent);
    }
  }

  /**
   * Joins two query values of the same kind; {@code null} old means bottom.
   *
   * @param old The previous value or {@code null}.
   * @param next The newly computed value.
   * @return The join.
   */
  private static Object join(Object old, Object next) {
    if (old == null) return next;
    // The legacy null dtype (stored as the sentinel) is ⊤ and absorbs, like UNKNOWN below.
    if (old == NULL_DTYPE || next == NULL_DTYPE) return NULL_DTYPE;
    if (old instanceof ShapeResult && next instanceof ShapeResult)
      return ((ShapeResult) old).union((ShapeResult) next);
    if (old instanceof Set && next instanceof Set) {
      @SuppressWarnings("unchecked")
      Set<DType> a = (Set<DType>) old;
      @SuppressWarnings("unchecked")
      Set<DType> b = (Set<DType>) next;
      EnumSet<DType> union = EnumSet.noneOf(DType.class);
      union.addAll(a);
      union.addAll(b);
      // An UNKNOWN member joins as a member: it marks the union partial without erasing the
      // members that resolved (wala/ML#862).
      return union;
    }
    return next;
  }

  /**
   * Meets two query values of the same kind: the members both hold, with the unknown remainder only
   * when both have it. The legacy null dtype is ⊤, so the other value is the meet.
   *
   * @param first A value.
   * @param second Another value.
   * @return The meet.
   */
  private static Object meet(Object first, Object second) {
    if (first == NULL_DTYPE) return second;
    if (second == NULL_DTYPE) return first;
    if (first instanceof ShapeResult && second instanceof ShapeResult)
      return ((ShapeResult) first).intersection((ShapeResult) second);
    if (first instanceof Set && second instanceof Set) {
      EnumSet<DType> both = EnumSet.noneOf(DType.class);
      for (Object dtype : (Set<?>) first)
        if (((Set<?>) second).contains(dtype)) both.add((DType) dtype);
      return both;
    }
    return second;
  }

  /**
   * Widens a settled value by a recomputation of it: {@link ShapeResult#widening} for shapes, and
   * the join for any other kind, whose values are of a finite number.
   *
   * @param settled The settled value.
   * @param value The recomputation.
   * @return The widened value.
   */
  private static Object widening(Object settled, Object value) {
    if (settled instanceof ShapeResult && value instanceof ShapeResult)
      return ((ShapeResult) settled).widening((ShapeResult) value);
    return join(settled, value);
  }

  /**
   * Widens a settled value by a recomputation of it as {@link #widening} does, and notes in {@link
   * #generalized} each shape the widening made.
   *
   * @param settled The settled value.
   * @param value The recomputation.
   * @return The widened value.
   */
  private Object generalizing(Object settled, Object value) {
    Object widened = widening(settled, value);
    if (settled instanceof ShapeResult
        && value instanceof ShapeResult
        && widened instanceof ShapeResult)
      for (List<Dimension<?>> member : ((ShapeResult) widened).members())
        if (!((ShapeResult) settled).members().contains(member)
            && !((ShapeResult) value).members().contains(member)) this.generalized.add(member);
    return widened;
  }

  /**
   * Returns a value without the members that a shape a widening made covers, where the value holds
   * that shape ({@link #generalized}). The value is a settled recomputation, or the result of a
   * first evaluation: a query first evaluated after a widening reads the widened shape as any
   * reader does, and one evaluated before it would be recomputed without those members, so the
   * order of the two would otherwise show in what the query holds.
   *
   * @param value The value, or {@code null}.
   * @return The value to keep.
   */
  private Object absorb(Object value) {
    if (this.generalized.isEmpty() || !(value instanceof ShapeResult)) return value;
    return ((ShapeResult) value).withoutCoveredBy(this.generalized);
  }

  /**
   * Joins a settled value and a recomputation of it for a query of {@link #joinSettled}: {@link
   * #join}, and for shapes without the members another member covers.
   *
   * @param settled The settled value.
   * @param value The recomputation.
   * @return The join.
   */
  private static Object joinCovering(Object settled, Object value) {
    Object joined = join(settled, value);
    return joined instanceof ShapeResult ? ((ShapeResult) joined).withoutCovered() : joined;
  }

  private Object widen(Object key, Object value) {
    if (!(value instanceof ShapeResult)) return value;
    ShapeResult shapes = (ShapeResult) value;
    boolean oversize = shapes.members().size() > MEMBER_WIDENING_CAP;
    boolean overRank = shapes.members().stream().anyMatch(m -> m.size() > RANK_WIDENING_CAP);
    if (!oversize && !overRank) return value;
    LOGGER.fine(() -> "Widening " + key + ": " + shapes.members().size() + " members.");
    return ShapeResult.unknown();
  }

  /**
   * Promotes bottom-valued members of nontrivial SCCs to the unknown-marked element and re-enqueues
   * their dependents (the design's pure-cycle step).
   *
   * <p>A restricted pass walks only the reader closure of the queries that entered {@link
   * #shapeBottoms} since the previous pass and are still in it, and of the readers that gained an
   * edge since then (wala/ML#1176). That closure holds every query the whole-graph pass would
   * promote. Each pass promotes every candidate it finds, and a promoted query leaves {@link
   * #shapeBottoms}, so a candidate now was not one at the previous pass, or no pass has run since
   * it became one. Either it entered {@link #shapeBottoms} since then, and is a seed itself, or its
   * component became cyclic since then. A component becomes cyclic only through a new edge on the
   * cycle, and the cycle's members are readers, transitively, of that edge's reader, so the closure
   * of that reader holds the candidate's whole component. A state is written only through {@link
   * #putState} and a query is first sighted only through {@link #register}, which mark the entries,
   * and an edge is added only by {@link #read}, which marks its reader. Under the check mode of
   * {@link #CHECK_SKIPS_PROPERTY}, the pass condenses the whole graph and promotes from it, and an
   * assertion fails if a promotion lies outside the closure.
   *
   * @return whether any promotion occurred.
   */
  private boolean promotePureCycles() {
    boolean any = false;
    // Only a shape query whose state is absent or bottom promotes, and its component lies whole in
    // the closure of the seeds below under their readers.
    long walkStart = System.nanoTime();
    this.promotionWalks++;
    this.promotionGraphQueries += this.state.size();
    // The entries since the previous pass seed this one, and what this pass's own writes add (a
    // promotion only leaves the bottoms, and adds no edge) seeds the next.
    Set<Object> seeds = HashSetFactory.make();
    for (Object key : this.newShapeBottoms) if (this.shapeBottoms.contains(key)) seeds.add(key);
    seeds.addAll(this.promotionEdgeReaders);
    this.newShapeBottoms.clear();
    this.promotionEdgeReaders.clear();
    Set<Object> region = null;
    if (this.restrictSettlement) {
      region = this.readerClosure(seeds);
      this.promotionClosureQueries += region.size();
      if (region.isEmpty() && !this.checkSkips) {
        this.promotionWalkNanos += System.nanoTime() - walkStart;
        return false;
      }
    } else this.promotionWholeWalks++;
    List<List<Object>> sccs;
    if (region == null) sccs = this.tarjan();
    else if (this.checkSkips) {
      sccs = this.tarjan();
      this.checkRestriction(sccs, region);
    } else sccs = this.tarjan(region);
    this.promotionWalkNanos += System.nanoTime() - walkStart;
    this.promotionCondensedQueries += condensedQueries(sccs, region);
    for (List<Object> scc : sccs) {
      boolean cyclic =
          scc.size() > 1
              || this.reads.getOrDefault(scc.get(0), Collections.emptySet()).contains(scc.get(0));
      if (!cyclic) continue;
      this.noteCyclic(scc, region, false);
      for (Object key : scc) {
        // Only shape-valued queries promote; a bottom dtype set already reads as "no info."
        if (!this.shapeKinds.get(key)) continue;
        Object value = this.state.get(key);
        boolean bottomShapes =
            value == null || (value instanceof ShapeResult && ((ShapeResult) value).isBottom());
        if (!bottomShapes) continue;
        if (value == null && !(this.transfers.containsKey(key))) continue;
        if (region != null && !region.contains(key))
          throw new AssertionError("A promotion lies outside the region: " + key + ".");
        this.putState(key, ShapeResult.unknown());
        this.bumpVersion(key);
        this.stamp(key, true);
        this.reEnqueueStaleDependents(key);
        any = true;
      }
    }
    return any;
  }

  private List<List<Object>> tarjan() {
    return this.tarjan(null);
  }

  /**
   * The condensation of the query graph, or of a region of it closed under readers, in the
   * whole-graph emission order: a query outside such a region reads none inside, so the whole
   * pass's search never enters the region from outside it and its components emit in the same
   * relative order.
   *
   * @param region The region, or {@code null} for the whole graph.
   * @return The components, in reverse-topological order.
   */
  private List<List<Object>> tarjan(Set<Object> region) {
    Map<Object, Integer> index = HashMapFactory.make();
    Map<Object, Integer> low = HashMapFactory.make();
    Set<Object> onStack = HashSetFactory.make();
    Deque<Object> stack = new ArrayDeque<>();
    List<List<Object>> sccs = new ArrayList<>();
    int[] counter = {0};
    // A settlement-requested query with no reads of its own (its transfer observed emptiness
    // without an engine read) appears in no edge, so it roots explicitly; a read-free root emits
    // as a trivial SCC before any reader that reaches it, preserving the reverse-topological
    // emission order.
    List<Object> roots = new ArrayList<>();
    if (region == null) roots.addAll(this.reads.keySet());
    else {
      // The region's readers in the whole graph's order, without scanning every key.
      for (Object key : region) if (this.readsOrder.containsKey(key)) roots.add(key);
      roots.sort(Comparator.comparingInt(this.readsOrder::get));
    }
    for (Object requested : this.settlementRequested)
      if (!this.reads.containsKey(requested) && (region == null || region.contains(requested)))
        roots.add(requested);
    for (Object root : roots) {
      if (index.containsKey(root)) continue;
      Deque<Object[]> frames = new ArrayDeque<>();
      frames.push(new Object[] {root, this.successors(root, region).iterator()});
      index.put(root, counter[0]);
      low.put(root, counter[0]);
      counter[0]++;
      stack.push(root);
      onStack.add(root);
      while (!frames.isEmpty()) {
        Object[] frame = frames.peek();
        Object node = frame[0];
        @SuppressWarnings("unchecked")
        java.util.Iterator<Object> it = (java.util.Iterator<Object>) frame[1];
        if (it.hasNext()) {
          Object next = it.next();
          if (!index.containsKey(next)) {
            index.put(next, counter[0]);
            low.put(next, counter[0]);
            counter[0]++;
            stack.push(next);
            onStack.add(next);
            frames.push(new Object[] {next, this.successors(next, region).iterator()});
          } else if (onStack.contains(next))
            low.put(node, Math.min(low.get(node), index.get(next)));
        } else {
          frames.pop();
          if (!frames.isEmpty()) {
            Object parent = frames.peek()[0];
            low.put(parent, Math.min(low.get(parent), low.get(node)));
          }
          if (low.get(node).equals(index.get(node))) {
            List<Object> scc = new ArrayList<>();
            Object member;
            do {
              member = stack.pop();
              onStack.remove(member);
              scc.add(member);
            } while (!member.equals(node));
            sccs.add(scc);
          }
        }
      }
    }
    return sccs;
  }

  /**
   * A query's reads in their set order, within a region.
   *
   * @param key The query.
   * @param region The region, or {@code null} for the whole graph.
   * @return The reads.
   */
  private List<Object> successors(Object key, Set<Object> region) {
    List<Object> ret = new ArrayList<>();
    for (Object read : this.reads.getOrDefault(key, Collections.emptySet()))
      if (region == null || region.contains(read)) ret.add(read);
    return ret;
  }
}
