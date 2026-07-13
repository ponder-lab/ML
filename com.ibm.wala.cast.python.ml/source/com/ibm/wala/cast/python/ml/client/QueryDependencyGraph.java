package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.logging.Logger;

/**
 * Records the dependency edges among resolution queries as they evaluate (wala/ML#365, Phase 1).
 * Every memoized computation — a {@code (node, vn)} value read, a whole-generator evaluation, or a
 * producer delegation — {@linkplain #enter enters} the graph while it computes and {@linkplain
 * #access accesses} the queries it reads, producing the explicit query-dependency graph the
 * worklist engine (Phase 2) will iterate. Phase 1 is telemetry only: the graph validates the
 * design's cycle assumptions via the {@linkplain #summarize summary}'s strongly-connected-component
 * census and changes no behavior.
 *
 * <p>Evaluation is single-threaded within a builder's generator run; the per-builder instance
 * synchronizes defensively on itself.
 */
class QueryDependencyGraph {

  private static final Logger LOGGER = Logger.getLogger(QueryDependencyGraph.class.getName());

  /** Per-builder instances, weakly keyed like the sibling caches. */
  private static final Map<PropagationCallGraphBuilder, QueryDependencyGraph> INSTANCES =
      Collections.synchronizedMap(new WeakHashMap<>());

  /** Adjacency: query key → the query keys its computation read. */
  private final Map<Object, Set<Object>> edges = HashMapFactory.make();

  /** The stack of currently-computing queries. */
  private final Deque<Object> computing = new ArrayDeque<>();

  private QueryDependencyGraph() {}

  /**
   * Returns the builder's graph, creating it on first use.
   *
   * @param builder The builder whose graph to return.
   * @return The per-builder graph.
   */
  static QueryDependencyGraph of(PropagationCallGraphBuilder builder) {
    return INSTANCES.computeIfAbsent(builder, b -> new QueryDependencyGraph());
  }

  /**
   * Drops the builder's graph.
   *
   * @param builder The builder whose graph to drop.
   */
  static void clear(PropagationCallGraphBuilder builder) {
    INSTANCES.remove(builder);
  }

  /**
   * Records that the currently-computing query (if any) reads the given query.
   *
   * @param key The query being read.
   */
  synchronized void access(Object key) {
    Object reader = this.computing.peek();
    if (reader != null && !reader.equals(key))
      this.edges.computeIfAbsent(reader, k -> HashSetFactory.make()).add(key);
  }

  /**
   * Marks the query as computing (its reads become edges) until the matching {@link #exit}.
   *
   * @param key The query beginning its computation.
   */
  synchronized void enter(Object key) {
    this.edges.computeIfAbsent(key, k -> HashSetFactory.make());
    this.computing.push(key);
  }

  /**
   * Ends the query's computation.
   *
   * @param key The query ending its computation; must match the current top.
   */
  synchronized void exit(Object key) {
    Object top = this.computing.poll();
    if (top != null && !top.equals(key))
      LOGGER.warning("Query graph enter/exit mismatch: expected " + top + ", got " + key + ".");
  }

  /**
   * Logs the graph census at {@code FINE}: query and edge counts and the nontrivial
   * strongly-connected components (Tarjan), the cycles the Phase 2 worklist must converge.
   */
  synchronized void summarize() {
    int nodeCount = this.edges.size();
    int edgeCount = this.edges.values().stream().mapToInt(Set::size).sum();
    List<List<Object>> sccs = this.tarjan();
    List<List<Object>> nontrivial = new ArrayList<>();
    for (List<Object> scc : sccs)
      if (scc.size() > 1
          || (scc.size() == 1
              && this.edges.getOrDefault(scc.get(0), Collections.emptySet()).contains(scc.get(0))))
        nontrivial.add(scc);
    nontrivial.sort((a, b) -> Integer.compare(b.size(), a.size()));
    LOGGER.fine(
        () ->
            "Query graph census: "
                + nodeCount
                + " queries, "
                + edgeCount
                + " edges, "
                + nontrivial.size()
                + " nontrivial SCCs.");
    int rank = 0;
    for (List<Object> scc : nontrivial) {
      if (rank++ >= 10) break;
      final int size = scc.size();
      final Object sample = scc.get(0);
      LOGGER.fine(() -> "  SCC size " + size + ", e.g. " + sample);
    }
  }

  /**
   * Computes the strongly-connected components of the recorded graph (iterative Tarjan).
   *
   * @return The components, each a list of query keys.
   */
  private List<List<Object>> tarjan() {
    Map<Object, Integer> index = HashMapFactory.make();
    Map<Object, Integer> low = HashMapFactory.make();
    Set<Object> onStack = HashSetFactory.make();
    Deque<Object> stack = new ArrayDeque<>();
    List<List<Object>> sccs = new ArrayList<>();
    int[] counter = {0};

    for (Object root : new ArrayList<>(this.edges.keySet())) {
      if (index.containsKey(root)) continue;
      // Iterative DFS frame: [node, iterator-position] via an explicit frame stack.
      Deque<Object[]> frames = new ArrayDeque<>();
      frames.push(new Object[] {root, new ArrayList<>(this.edges.get(root)).iterator()});
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
                  new ArrayList<>(this.edges.getOrDefault(next, Collections.emptySet())).iterator()
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
