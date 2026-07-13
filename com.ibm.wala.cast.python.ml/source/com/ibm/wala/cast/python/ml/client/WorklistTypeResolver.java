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
 * every nested query read to {@link #read}, which records the dependency edge, schedules the
 * dependency, and returns the state table's current value rather than recursing. Values ascend a
 * lattice — {@link ShapeResult} ordered by member inclusion with the unknown mark as a second
 * component, dtype sets by inclusion with the {@code UNKNOWN}-absorbing normalization — via joins
 * at every update, so iteration reaches a fixpoint; a strictly grown value re-enqueues the query's
 * dependents.
 *
 * <p>Cycles need no guards: reads never recurse. A dependency SCC with no external base stabilizes
 * at the iteration bottom, which would read as "not a tensor," so after each stabilization the
 * engine promotes bottom-valued members of nontrivial SCCs to the unknown-marked element and
 * re-solves, preserving the sound answer for genuinely baseless loop-carried values (the design's
 * promotion step). Widenings bound divergence: a member-set cardinality cap and a rank cap, both
 * logged when they fire.
 *
 * <p>Enabled by the {@code ariadne.typeResolution.worklist} system property; the default remains
 * the round-based resolution.
 */
final class WorklistTypeResolver {

  private static final Logger LOGGER = Logger.getLogger(WorklistTypeResolver.class.getName());

  /** Member-set cardinality beyond which a value widens to unknown-marked. */
  private static final int MEMBER_WIDENING_CAP = 16;

  /** Rank beyond which a member widens away. */
  private static final int RANK_WIDENING_CAP = 12;

  /** The per-builder active engine; non-null only while the flag-gated resolution runs. */
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

  /** The stack of currently-evaluating queries; reads record edges against the top. */
  private final Deque<Object> evaluating = new ArrayDeque<>();

  /** Membership view of {@link #evaluating}, the inline-recursion cycle break. */
  private final Set<Object> inStack = HashSetFactory.make();

  /** Queries whose first (inline) evaluation has completed. */
  private final Set<Object> evaluated = HashSetFactory.make();

  private WorklistTypeResolver() {}

  /**
   * Returns whether the flag-gated engine should run.
   *
   * @return {@code true} iff the {@code ariadne.typeResolution.worklist} property is set.
   */
  static boolean enabled() {
    return Boolean.getBoolean("ariadne.typeResolution.worklist")
        || "true".equals(System.getenv("ARIADNE_WORKLIST"));
  }

  /**
   * Returns the builder's active engine, or {@code null} when the round-based resolution runs.
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
   * Uninstalls the builder's engine.
   *
   * @param builder The builder whose engine to remove.
   */
  static void uninstall(PropagationCallGraphBuilder builder) {
    ACTIVE.remove(builder);
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
    Object value = this.state.get(key);
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
    if (value != null) return value;
    return shapeKind ? ShapeResult.bottom() : EnumSet.noneOf(DType.class);
  }

  private void enqueue(Object key) {
    if (this.enqueued.add(key)) this.worklist.add(key);
  }

  /** Runs the fixpoint: chaotic iteration, then SCC promotion, repeating until neither moves. */
  private void solve() {
    boolean promoted;
    do {
      this.iterate();
      promoted = this.promotePureCycles();
    } while (promoted);
    this.dumpFiltered();
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

  private static String brief(Object o) {
    String s = String.valueOf(o);
    return s.length() > 160 ? s.substring(0, 160) : s;
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
      result =
          Boolean.TRUE.equals(this.shapeKinds.get(key))
              ? ShapeResult.unknown()
              : EnumSet.of(DType.UNKNOWN);
    } finally {
      this.evaluating.pop();
      this.inStack.remove(key);
      this.evaluated.add(key);
    }
    // The legacy dtype convention allows a null transfer result meaning ⊤; normalize before the
    // join so the lattice sees a value of the query's kind.
    if (result == null)
      result =
          Boolean.TRUE.equals(this.shapeKinds.get(key))
              ? ShapeResult.unknown()
              : EnumSet.of(DType.UNKNOWN);
    Object old = this.state.get(key);
    Object joined = join(old, result);
    joined = this.widen(key, joined);
    if (!joined.equals(old)) {
      this.state.put(key, joined);
      for (Object dependent : this.dependents.getOrDefault(key, Collections.emptySet()))
        this.enqueue(dependent);
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
        Object value = this.state.get(key);
        boolean bottomShapes =
            value == null || (value instanceof ShapeResult && ((ShapeResult) value).isBottom());
        if (!bottomShapes) continue;
        // Only shape-valued queries promote; a bottom dtype set already reads as "no info."
        if (value == null && !(this.transfers.containsKey(key))) continue;
        this.state.put(key, ShapeResult.unknown());
        for (Object dependent : this.dependents.getOrDefault(key, Collections.emptySet()))
          this.enqueue(dependent);
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
