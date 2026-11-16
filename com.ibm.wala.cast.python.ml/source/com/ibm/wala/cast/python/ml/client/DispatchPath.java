package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory.Dispatch;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The sources a {@link TensorGeneratorFactory} dispatch is resolving, what each dispatch on them
 * depended on, and the dispatches the walk has completed.
 *
 * <p>The factory answers a source already on the path with {@code null}, so the walk can try other
 * branches (wala/ML#435). A dispatch during which such a cut was made at a source entered before
 * its own holds for this path only: reached from elsewhere, the cut source would be resolved
 * instead. Every other dispatch depends on its source alone, since the call graph and the pointer
 * analysis are final when generators are dispatched, so the factory memoizes it per builder.
 *
 * <p>The sources whose dispatches reach each other form a cycle, and a walk that computed each of
 * them again at every visit took every simple path through the cycle, a number exponential in the
 * cycle's size (wala/ML#1274). So the walk is a depth-first search for strongly connected
 * components. Each source entered is numbered in the order of entry, and its dispatch records the
 * lowest number it depended on: that of a path source it was cut at, or that of a completed
 * dispatch it replayed that depended on a source still on the path. A dispatch that depended on no
 * number below its own is that of the source the walk entered a cycle at, or of a source in no
 * cycle. It is kept for the rest of the walk and memoized per builder, and the dispatches kept
 * below it that depended on the path are dropped, since they hold only while their cycle is being
 * resolved. Any other completed dispatch is kept until then and replayed at its source's later
 * visits, so the walk computes each source of a cycle once.
 *
 * <p>A dispatch can also start a walk of its own, through the public {@link
 * TensorGeneratorFactory#getGenerator(PointsToSetVariable,
 * com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder)}, on a fresh path. That walk
 * resolves a source the outer path is resolving instead of cutting it, and so does every later
 * walk, but a dispatch of the outer path that would replay its result could not: it would cut that
 * source. So a dispatch reaching a source that another path of its thread is resolving is not
 * memoized per builder either. The other paths do not move while this one walks, so the dispatch is
 * still kept for the rest of this walk.
 */
final class DispatchPath {

  /** The sources being resolved on each thread, on any path, with how many paths resolve each. */
  private static final ThreadLocal<Map<PointsToSetVariable, Integer>> RESOLVING =
      ThreadLocal.withInitial(HashMap::new);

  /**
   * How many times a dispatch on each thread, on any path, reached a source being resolved or
   * replayed a dispatch that had, so a reading made of several walks can tell whether one of them
   * depended on a path ({@link #reachedOnThread}).
   */
  private static final ThreadLocal<long[]> REACHED_ON_THREAD =
      ThreadLocal.withInitial(() -> new long[1]);

  /** A source on the path. */
  private static final class Frame {

    /** The source. */
    private final PointsToSetVariable source;

    /** The source's number, in the order the walk entered its sources. */
    private final int index;

    /** The lowest number the source's dispatch has depended on so far, at most its own. */
    private int low;

    /** Whether the dispatch was cut, or replayed a dispatch that depended on a path. */
    private boolean reached;

    /** Whether the dispatch depended on a source that another path of the thread is resolving. */
    private boolean outside;

    private Frame(PointsToSetVariable source, int index) {
      this.source = source;
      this.index = index;
      this.low = index;
    }
  }

  /**
   * A completed dispatch of this walk.
   *
   * @param dispatch The dispatch.
   * @param index The source's number.
   * @param provisional Whether the dispatch depended on a source below its own on the path, so it
   *     holds until that source is left.
   * @param outside Whether the dispatch depended on a source that another path of the thread is
   *     resolving.
   */
  private record Kept(Dispatch dispatch, int index, boolean provisional, boolean outside) {}

  /** The sources on this path, with their frames. */
  private final Map<PointsToSetVariable, Frame> frames = new HashMap<>();

  /** The frames of this path, the last entered first. */
  private final Deque<Frame> path = new ArrayDeque<>();

  /** The completed dispatches of this walk, by source. */
  private final Map<PointsToSetVariable, Kept> kept = new HashMap<>();

  /** The sources of the provisional dispatches among {@link #kept}, the last completed first. */
  private final Deque<PointsToSetVariable> provisional = new ArrayDeque<>();

  /** How many sources this walk has entered. */
  private int entered;

  /** Whether dispatches along this path are kept, and read and write the factory's memo. */
  private final boolean memoizes;

  /** A path whose dispatches are memoized. */
  DispatchPath() {
    this(true);
  }

  /**
   * A path.
   *
   * @param memoizes Whether dispatches along this path are kept for the walk, and read and write
   *     the factory's memo.
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
   * A path holding this path's sources whose dispatches are computed afresh at every visit: none is
   * kept, and none reads or writes the factory's memo.
   *
   * @return The path.
   */
  DispatchPath unmemoized() {
    DispatchPath ret = new DispatchPath(false);
    List<Frame> entered = new ArrayList<>(this.path);
    for (int i = entered.size() - 1; i >= 0; i--) {
      Frame frame = new Frame(entered.get(i).source, ret.entered++);
      ret.frames.put(frame.source, frame);
      ret.path.push(frame);
    }
    return ret;
  }

  /**
   * Whether a source is on this path, so its dispatch is cut. The dispatch now on top of the path
   * then depends on the source's own, and holds until the source is left. A source that another
   * path of the thread is resolving is not cut, and keeps every dispatch now on this path from
   * being memoized per builder.
   *
   * @param source The source.
   * @return {@code true} iff the source is on this path.
   */
  boolean cuts(PointsToSetVariable source) {
    Frame entered = this.frames.get(source);
    if (entered != null) {
      REACHED_ON_THREAD.get()[0]++;
      // A source on the path means the path has a top.
      Frame top = this.path.element();
      top.low = Math.min(top.low, entered.index);
      top.reached = true;
      return true;
    }
    if (RESOLVING.get().containsKey(source)) {
      REACHED_ON_THREAD.get()[0]++;
      Frame top = this.path.peek();
      if (top != null) top.reached = top.outside = true;
    }
    return false;
  }

  /**
   * How many times a dispatch on this thread, on any path, has reached a source being resolved or
   * replayed a dispatch that had. A reading whose walks left this unchanged depends on what it read
   * alone.
   *
   * @return The count.
   */
  static long reachedOnThread() {
    return REACHED_ON_THREAD.get()[0];
  }

  /**
   * Sets {@link #reachedOnThread} back, after walks whose results are discarded.
   *
   * @param reached What {@link #reachedOnThread} returned before the walks.
   */
  static void forgetReachedSince(long reached) {
    REACHED_ON_THREAD.get()[0] = reached;
  }

  /**
   * The dispatch this walk completed for a source not on this path, for the dispatch now on top of
   * the path to replay. A provisional one makes that dispatch depend on what it depended on, as a
   * cut does; one that depended on another path's source keeps it from being memoized per builder
   * and adds nothing to what it depends on along this path, its cycle being complete.
   *
   * @param source The source.
   * @return The dispatch, or {@code null} when the walk keeps none for the source.
   */
  Dispatch kept(PointsToSetVariable source) {
    Kept kept = this.kept.get(source);
    if (kept == null) return null;
    if (kept.provisional() || kept.outside()) REACHED_ON_THREAD.get()[0]++;
    Frame top = this.path.peek();
    if (top != null) {
      if (kept.provisional()) top.low = Math.min(top.low, kept.index());
      top.reached |= kept.provisional() || kept.outside();
      top.outside |= kept.outside();
    }
    return kept.dispatch();
  }

  /**
   * Enters a source not on this path, for which the walk keeps no dispatch.
   *
   * @param source The source.
   */
  void enter(PointsToSetVariable source) {
    Frame frame = new Frame(source, this.entered++);
    this.frames.put(source, frame);
    this.path.push(frame);
    RESOLVING.get().merge(source, 1, Integer::sum);
  }

  /**
   * Leaves a source whose dispatch completed, keeping the dispatch for the rest of the walk, or,
   * when it depended on a source below its own on the path, until that source is left.
   *
   * @param source The source, as passed to {@link #enter}.
   * @param generator The dispatch's generator, or {@code null} when there is none.
   * @param failure What the dispatch threw, or {@code null} when it returned.
   * @return The dispatch when it depended on no source being resolved but its own, so the factory
   *     memoizes it per builder, and {@code null} otherwise.
   */
  Dispatch leave(
      PointsToSetVariable source, TensorGenerator generator, IllegalArgumentException failure) {
    Frame frame = this.pop(source);
    Frame parent = this.path.peek();
    boolean provisional = frame.low < frame.index;
    if (parent != null) {
      parent.reached |= frame.reached;
      parent.outside |= frame.outside;
      if (provisional) parent.low = Math.min(parent.low, frame.low);
    }
    // The dispatches that depended on this source, or on one of them, held while it was on the
    // path.
    if (!provisional) this.dropProvisionalSince(frame.index);
    if (!this.memoizes) return null;
    Dispatch dispatch =
        new Dispatch(generator == null ? null : generator.copy(), failure, !frame.reached);
    this.kept.put(source, new Kept(dispatch, frame.index, provisional, frame.outside));
    if (provisional) this.provisional.push(source);
    return provisional || frame.outside ? null : dispatch;
  }

  /**
   * Leaves a source whose dispatch did not complete. Nothing is kept for it, and no dispatch now on
   * this path is memoized per builder.
   *
   * @param source The source, as passed to {@link #enter}.
   */
  void abandon(PointsToSetVariable source) {
    Frame frame = this.pop(source);
    this.dropProvisionalSince(frame.index);
    Frame parent = this.path.peek();
    if (parent != null) parent.reached = parent.outside = true;
  }

  /**
   * Takes the source on top of the path off it.
   *
   * @param source The source.
   * @return Its frame.
   */
  private Frame pop(PointsToSetVariable source) {
    Frame frame = this.path.pop();
    assert frame.source == source : "Leaving " + source + " under " + frame.source + ".";
    this.frames.remove(source);
    RESOLVING.get().computeIfPresent(source, (s, paths) -> paths == 1 ? null : paths - 1);
    return frame;
  }

  /**
   * Drops the provisional dispatches of the sources entered after a source.
   *
   * @param index The source's number.
   */
  private void dropProvisionalSince(int index) {
    while (!this.provisional.isEmpty() && this.kept.get(this.provisional.peek()).index() > index)
      this.kept.remove(this.provisional.pop());
  }
}
