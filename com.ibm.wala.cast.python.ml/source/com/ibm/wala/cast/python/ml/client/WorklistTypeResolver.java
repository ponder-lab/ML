package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.WeakHashMap;
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
 * by inclusion with the {@code UNKNOWN}-absorbing normalization and the legacy null-⊤ preserved
 * through a sentinel so callers' null-check fallback arms behave as under recursion — via joins at
 * every update, so iteration reaches a fixpoint; a grown value re-enqueues the query's dependents,
 * and the per-key evaluation-count valve pins any oscillator at its current value.
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

  private final Deque<Object> worklist = new ArrayDeque<>();
  private final Set<Object> enqueued = HashSetFactory.make();
  private final Map<Object, Integer> evaluationCounts = HashMapFactory.make();

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

  /** Membership view of {@link #evaluating}, the inline-recursion cycle break. */
  private final Set<Object> inStack = HashSetFactory.make();

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
   * Uninstalls the builder's engine, logging its end-of-analysis census (the successor of the
   * retired wala/ML#591 recorder's summary; see wala/ML#727). The census reads only map sizes, so
   * it costs no key hashing.
   *
   * @param builder The builder whose engine to remove.
   */
  static void uninstall(PropagationCallGraphBuilder builder) {
    WorklistTypeResolver engine = ACTIVE.remove(builder);
    if (engine == null) return;
    LOGGER.fine(
        () ->
            "Query census: "
                + engine.state.size()
                + " queries, "
                + engine.reads.values().stream().mapToInt(Set::size).sum()
                + " dependency edges.");
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
    Object reader = this.evaluating.peek();
    if (reader != null && !reader.equals(key)) {
      this.reads.computeIfAbsent(reader, k -> HashSetFactory.make()).add(key);
      this.dependents.computeIfAbsent(key, k -> HashSetFactory.make()).add(reader);
    }
    if (!this.transfers.containsKey(key)) {
      this.transfers.put(key, transfer);
      this.shapeKinds.put(key, shapeKind);
    }
    // Hybrid scheduling: a first-sighted query evaluates inline, so recursion gives the acyclic
    // region its natural (reverse-topological) order and each such query runs once; only an
    // on-stack read (a cycle) returns the current value and defers to the worklist, which then
    // converges just the cycle-affected keys.
    if (!this.evaluated.contains(key) && !this.inStack.contains(key)) this.evaluate(key);
    else if (this.inStack.contains(key)) this.cycleSeen = true;
    Object value = this.state.get(key);
    if (value == NULL_DTYPE) return null;
    if (value != null) return value;
    return shapeKind ? ShapeResult.bottom() : EnumSet.noneOf(DType.class);
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
    if (!this.transfers.containsKey(key)) {
      this.transfers.put(key, transfer);
      this.shapeKinds.put(key, shapeKind);
    }
    if (!this.evaluated.contains(key) && !this.inStack.contains(key)) this.evaluate(key);
    this.solve();
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

  /** Runs the fixpoint: chaotic iteration, then SCC promotion, repeating until neither moves. */
  private void solve() {
    // Nothing queued and no cycle observed since the last pass: every value came from a single
    // inline evaluation and is already final, so skip the iteration and (Tarjan-priced) promotion
    // entirely. This is what makes repeated demands (the second seeding pass) effectively free.
    if (this.worklist.isEmpty() && !this.cycleSeen) return;
    boolean promoted;
    do {
      this.iterate();
      promoted = this.cycleSeen && this.promotePureCycles();
    } while (promoted);
    if (this.cycleSeen) this.canonicalize();
    this.cycleSeen = false;
    this.dumpFiltered();
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
   * recomputed only if it sits in a cyclic SCC (where join history can originate) or reads a key
   * whose canonical value changed (where it can propagate). Within a cyclic SCC the replacements
   * are computed jointly against the settled state before any is written back (a Jacobi step), so
   * intra-SCC recomputation order cannot influence the result. Two guards keep the replacement
   * semantics sound: a transfer that throws keeps the settled value, and a recomputation may not
   * degrade a non-bottom settled value to ⊥ &mdash; the settled evidence that the value is a tensor
   * stands, and the pure-cycle promotion's unknown-marked elements (whose baseless transfers
   * recompute to ⊥) must not be reclassified as "not a tensor."
   */
  private void canonicalize() {
    int replaced = 0;
    String probeFilter = System.getProperty("ariadne.typeResolution.canonProbe");
    Set<Object> changed = HashSetFactory.make();
    for (List<Object> scc : this.tarjan()) {
      boolean cyclic =
          scc.size() > 1
              || this.reads.getOrDefault(scc.get(0), Collections.emptySet()).contains(scc.get(0));
      List<Object> targets = new ArrayList<>();
      for (Object key : scc)
        if (cyclic
            || !Collections.disjoint(this.reads.getOrDefault(key, Collections.emptySet()), changed))
          targets.add(key);
      if (probeFilter != null)
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
      if (targets.isEmpty()) continue;
      Map<Object, Object> replacements = HashMapFactory.make();
      for (Object key : targets) replacements.put(key, this.recomputeSettled(key));
      for (Map.Entry<Object, Object> replacement : replacements.entrySet()) {
        Object key = replacement.getKey();
        Object value = replacement.getValue();
        Object settled = this.state.get(key);
        if (probeFilter != null && String.valueOf(key).contains(probeFilter))
          LOGGER.fine(() -> "CANON-PROBE recompute " + brief(key) + " := " + brief(value));
        if (value == null || value.equals(settled)) continue;
        if (isBottomValue(value) && !isBottomValue(settled)) continue;
        this.state.put(key, value);
        changed.add(key);
        replaced++;
      }
    }
    if (replaced > 0) {
      int count = replaced;
      LOGGER.fine(() -> "Canonicalization replaced " + count + " settled values.");
    }
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
    Object result;
    this.evaluating.push(key);
    this.inStack.add(key);
    try {
      result = transfer.get();
    } catch (IllegalArgumentException e) {
      LOGGER.log(
          Level.FINE, e, () -> "Canonicalization IAE for " + key + "; keeping the settled value.");
      return this.state.get(key);
    } finally {
      this.evaluating.pop();
      this.inStack.remove(key);
    }
    if (result == null) result = this.shapeKinds.get(key) ? ShapeResult.unknown() : NULL_DTYPE;
    return this.widen(key, result);
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
          result = transfer.get();
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

  private void iterate() {
    while (!this.worklist.isEmpty()) {
      Object key = this.worklist.poll();
      this.enqueued.remove(key);
      int count = this.evaluationCounts.merge(key, 1, Integer::sum);
      // Ascent through the finite lattice bounds re-evaluations; a runaway count means a
      // non-monotone transfer or an unbounded value, so log the offender and stop ascending it.
      if (count > 100) {
        if (count == 101)
          LOGGER.warning(
              "Worklist evaluation cap hit for "
                  + key
                  + "; pinning at its current value. Investigate the transfer's monotonicity.");
        continue;
      }
      this.evaluate(key);
    }
  }

  private void evaluate(Object key) {
    Supplier<Object> transfer = this.transfers.get(key);
    if (transfer == null) return;
    Object result;
    this.evaluating.push(key);
    this.inStack.add(key);
    try {
      result = transfer.get();
    } catch (IllegalArgumentException e) {
      // Demand-driven callers catch this variously; under the engine the conservative reading is
      // an unknown value of the query's kind.
      LOGGER.log(Level.FINE, e, () -> "Worklist transfer IAE for " + key + "; treating as ⊤.");
      result = this.shapeKinds.get(key) ? ShapeResult.unknown() : EnumSet.of(DType.UNKNOWN);
    } finally {
      this.evaluating.pop();
      this.inStack.remove(key);
      this.evaluated.add(key);
    }
    // The legacy dtype convention allows a null transfer result meaning ⊤, and CALLERS null-check
    // it to run fallback arms. The state table cannot hold null, so store the sentinel and let
    // read/demand translate it back to null: normalizing to EnumSet.of(UNKNOWN) instead hides the
    // null from the reader's fallback arm and composes spurious unknown-dtype members (the
    // flag-on twins caught by testReshape2).
    if (result == null) result = this.shapeKinds.get(key) ? ShapeResult.unknown() : NULL_DTYPE;
    Object old = this.state.get(key);
    Object joined = join(old, result);
    joined = this.widen(key, joined);
    if (!joined.equals(old)) {
      this.state.put(key, joined);
      for (Object dependent :
          this.orderDependentsForReEnqueue(
              this.dependents.getOrDefault(key, Collections.emptySet())))
        // An on-stack reader consumes this fresh value as its own evaluation completes (the
        // read returns after this update), so re-enqueueing it would only repeat the same
        // transfer; every first inline evaluation would otherwise schedule its whole reader
        // chain for a redundant second run. A key still re-enqueues itself: a changed
        // self-loop needs another pass to stabilize.
        if (dependent.equals(key) || !this.inStack.contains(dependent)) this.enqueue(dependent);
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
      if (union.contains(DType.UNKNOWN)) return EnumSet.of(DType.UNKNOWN);
      return union;
    }
    return next;
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
   * @return whether any promotion occurred.
   */
  private boolean promotePureCycles() {
    boolean any = false;
    for (List<Object> scc : this.tarjan()) {
      boolean cyclic =
          scc.size() > 1
              || this.reads.getOrDefault(scc.get(0), Collections.emptySet()).contains(scc.get(0));
      if (!cyclic) continue;
      for (Object key : scc) {
        // Only shape-valued queries promote; a bottom dtype set already reads as "no info."
        if (!this.shapeKinds.get(key)) continue;
        Object value = this.state.get(key);
        boolean bottomShapes =
            value == null || (value instanceof ShapeResult && ((ShapeResult) value).isBottom());
        if (!bottomShapes) continue;
        if (value == null && !(this.transfers.containsKey(key))) continue;
        this.state.put(key, ShapeResult.unknown());
        for (Object dependent :
            this.orderDependentsForReEnqueue(
                this.dependents.getOrDefault(key, Collections.emptySet()))) this.enqueue(dependent);
        any = true;
      }
    }
    return any;
  }

  private List<List<Object>> tarjan() {
    Map<Object, Integer> index = HashMapFactory.make();
    Map<Object, Integer> low = HashMapFactory.make();
    Set<Object> onStack = HashSetFactory.make();
    Deque<Object> stack = new ArrayDeque<>();
    List<List<Object>> sccs = new ArrayList<>();
    int[] counter = {0};
    for (Object root : new ArrayList<>(this.reads.keySet())) {
      if (index.containsKey(root)) continue;
      Deque<Object[]> frames = new ArrayDeque<>();
      frames.push(new Object[] {root, new ArrayList<>(this.reads.get(root)).iterator()});
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
            frames.push(
                new Object[] {
                  next,
                  new ArrayList<>(this.reads.getOrDefault(next, Collections.emptySet())).iterator()
                });
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
}
