package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * The sources a {@link TensorGeneratorFactory} dispatch is resolving, and whether any dispatch on
 * them reached a source being resolved.
 *
 * <p>The factory answers a source already on the path with {@code null}, so the walk can try other
 * branches (wala/ML#435). A dispatch during which such a cut was made holds for this path only:
 * reached from elsewhere, the cut source would be resolved instead. Every other dispatch depends on
 * its source alone, since the call graph and the pointer analysis are final when generators are
 * dispatched, so the factory memoizes it per builder.
 *
 * <p>A dispatch can also start a walk of its own, through the public {@link
 * TensorGeneratorFactory#getGenerator(PointsToSetVariable,
 * com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder)}, on a fresh path. That walk
 * resolves a source the outer path is resolving instead of cutting it, and so does every later
 * walk, but a dispatch of the outer path that would replay its result could not: it would cut that
 * source. So a dispatch reaching a source that another path of its thread is resolving is not
 * memoized either.
 */
final class DispatchPath {

  /** The sources being resolved on each thread, on any path, with how many paths resolve each. */
  private static final ThreadLocal<Map<PointsToSetVariable, Integer>> RESOLVING =
      ThreadLocal.withInitial(HashMap::new);

  /** The sources on this path. */
  private final Set<PointsToSetVariable> sources = new HashSet<>();

  /** How many sources being resolved the dispatches on this path have reached. */
  private long reached;

  /** Whether dispatches along this path read and write the factory's memo. */
  private final boolean memoizes;

  /** A path whose dispatches are memoized. */
  DispatchPath() {
    this(true);
  }

  /**
   * A path.
   *
   * @param memoizes Whether dispatches along this path read and write the factory's memo.
   */
  DispatchPath(boolean memoizes) {
    this.memoizes = memoizes;
  }

  /**
   * Whether dispatches along this path read and write the factory's memo.
   *
   * @return {@code true} iff they do.
   */
  boolean memoizes() {
    return this.memoizes;
  }

  /**
   * Whether a source is on this path, so its dispatch is cut. Either way, a source being resolved
   * on this or another path of the thread keeps every dispatch now on this path from being
   * memoized.
   *
   * @param source The source.
   * @return {@code true} iff the source is on this path.
   */
  boolean cuts(PointsToSetVariable source) {
    if (RESOLVING.get().containsKey(source)) this.reached++;
    return this.sources.contains(source);
  }

  /**
   * Enters a source not on this path.
   *
   * @param source The source.
   * @return A mark to pass to {@link #leave}.
   */
  long enter(PointsToSetVariable source) {
    this.sources.add(source);
    RESOLVING.get().merge(source, 1, Integer::sum);
    return this.reached;
  }

  /**
   * Leaves a source.
   *
   * @param source The source, as passed to {@link #enter}.
   * @param mark What {@link #enter} returned for it.
   * @return {@code true} iff no dispatch made while the source was on the path reached a source
   *     being resolved, so the source's dispatch depends on the source alone.
   */
  boolean leave(PointsToSetVariable source, long mark) {
    this.sources.remove(source);
    RESOLVING.get().computeIfPresent(source, (s, paths) -> paths == 1 ? null : paths - 1);
    return this.reached == mark;
  }
}
