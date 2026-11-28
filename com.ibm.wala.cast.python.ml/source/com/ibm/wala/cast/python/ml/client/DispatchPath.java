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
 * The sources a {@link TensorGeneratorFactory} dispatch is resolving, the cycles their dispatches
 * form, and the dispatches the walk has completed.
 *
 * <p>The factory answers a source already on the path with {@code null}, so the walk can try other
 * branches (wala/ML#435). A dispatch during which such a cut was made holds for this path only:
 * reached from elsewhere, the cut source would be resolved instead. Every other dispatch depends on
 * its source alone, since the call graph and the pointer analysis are final when generators are
 * dispatched, so the factory memoizes it per builder.
 *
 * <p>The sources whose dispatches reach each other form a cycle. Computing each of them again at
 * every visit takes every simple path through the cycle, a number exponential in the cycle's size,
 * and gives a member the answer of the path it was reached from (wala/ML#1274). So the walk is a
 * depth-first search for strongly connected components. Each source entered is numbered in the
 * order of entry, and its dispatch records the lowest number it reached: that of a source on the
 * path it was cut at, or of a source whose dispatch completed in a component still open. A
 * completed dispatch that reached a number below its own belongs to a component still open, and its
 * source is cut at its later visits, as one on the path is. The source whose dispatch reached no
 * number below its own, where a later one reached its number, entered the component first. When it
 * is left, the component is complete ({@link #takeComponent}), and the factory solves its members'
 * dispatches together, reading each member's entry in a table while it does ({@link #beginSolve}).
 * What a member reads is then the same whichever member a walk entered first.
 *
 * <p>A dispatch can also start a walk of its own, through the public {@link
 * TensorGeneratorFactory#getGenerator(PointsToSetVariable,
 * com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder)}, on a fresh path. That walk
 * resolves a source the outer path is resolving instead of cutting it, and so does every later
 * walk, but a dispatch of the outer path that would replay its result could not: it would cut that
 * source. So a dispatch reaching a source that another path of its thread is resolving is not
 * memoized per builder either.
 */
final class DispatchPath {

  /** The sources being resolved on each thread, on any path, with how many paths resolve each. */
  private static final ThreadLocal<Map<PointsToSetVariable, Integer>> RESOLVING =
      ThreadLocal.withInitial(HashMap::new);

  /**
   * How many times a dispatch on each thread, on any path, reached a source being resolved, so a
   * reading made of several walks can tell whether one of them was cut ({@link #reachedOnThread}).
   */
  private static final ThreadLocal<long[]> REACHED_ON_THREAD =
      ThreadLocal.withInitial(() -> new long[1]);

  /** A source on the path. */
  private static final class Frame {

    /** The source. */
    private final PointsToSetVariable source;

    /** The source's number, in the order the walk entered its sources. */
    private final int index;

    /** The lowest number the source's dispatch has reached so far, at most its own. */
    private int low;

    /** Whether the dispatch reached a source being resolved, so it holds for this path only. */
    private boolean reached;

    /** Whether the dispatch depended on a source that another path of the thread is resolving. */
    private boolean outside;

    /**
     * The depth of the outermost component being solved whose entry the dispatch read, or {@link
     * Integer#MAX_VALUE} when it read none.
     */
    private int tableDepth = Integer.MAX_VALUE;

    /** Whether a dispatch reached this source's own number while it was on the path. */
    private boolean cycles;

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
   * @param outside Whether the dispatch depended on a source that another path of the thread is
   *     resolving.
   */
  private record Kept(Dispatch dispatch, boolean outside) {}

  /** The sources on this path, with their frames. */
  private final Map<PointsToSetVariable, Frame> frames = new HashMap<>();

  /** The frames of this path, the last entered first. */
  private final Deque<Frame> path = new ArrayDeque<>();

  /**
   * The sources whose dispatches completed in a component still open, by number, the last completed
   * first, with their numbers in {@link #open}.
   */
  private final Deque<PointsToSetVariable> stack = new ArrayDeque<>();

  /** The numbers of the sources in {@link #stack}. */
  private final Map<PointsToSetVariable, Integer> open = new HashMap<>();

  /** The completed dispatches of this walk that depend on their source alone, by source. */
  private final Map<PointsToSetVariable, Kept> kept = new HashMap<>();

  /** The tables of the components being solved, by depth, the outermost first. */
  private final List<Map<PointsToSetVariable, TensorGenerator>> tables = new ArrayList<>();

  /** The members of the component the last source left completed, or {@code null}. */
  private List<PointsToSetVariable> component;

  /** How many times a dispatch on this path read an entry of a component being solved. */
  private long tableReads;

  /** How many sources this walk has entered. */
  private int entered;

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
   * Whether a source is cut: it is on this path, or its dispatch completed in a component still
   * open. The dispatch now on top of the path then reached the source's number. Either way, a
   * source being resolved on this or another path of the thread keeps every dispatch now on this
   * path from being memoized per builder.
   *
   * @param source The source.
   * @return {@code true} iff the source is cut.
   */
  boolean cuts(PointsToSetVariable source) {
    Frame top = this.path.peek();
    Frame entered = this.frames.get(source);
    Integer completed = entered == null ? this.open.get(source) : null;
    if (entered != null || completed != null) {
      REACHED_ON_THREAD.get()[0]++;
      int index = entered != null ? entered.index : completed;
      if (top != null) {
        top.low = Math.min(top.low, index);
        top.reached = true;
      }
      if (entered != null) entered.cycles = true;
      return true;
    }
    if (RESOLVING.get().containsKey(source)) {
      REACHED_ON_THREAD.get()[0]++;
      if (top != null) top.reached = top.outside = true;
    }
    return false;
  }

  /**
   * How many times a dispatch on this thread, on any path, has reached a source being resolved. A
   * reading whose walks left this unchanged depends on what it read alone.
   *
   * @return The count.
   */
  static long reachedOnThread() {
    return REACHED_ON_THREAD.get()[0];
  }

  /**
   * Whether a source is a member of a component being solved, so its dispatch reads its entry.
   *
   * @param source The source.
   * @return {@code true} iff it is.
   */
  boolean solving(PointsToSetVariable source) {
    for (Map<PointsToSetVariable, TensorGenerator> table : this.tables)
      if (table.containsKey(source)) return true;
    return false;
  }

  /**
   * The entry of a member of a component being solved, for the dispatch now on top of the path to
   * read. That dispatch then holds only while the component is solved.
   *
   * @param source The member, for which {@link #solving} holds.
   * @return A copy of its entry, or {@code null} while it has none.
   */
  TensorGenerator entry(PointsToSetVariable source) {
    this.tableReads++;
    REACHED_ON_THREAD.get()[0]++;
    for (int depth = this.tables.size() - 1; depth >= 0; depth--) {
      Map<PointsToSetVariable, TensorGenerator> table = this.tables.get(depth);
      if (table.containsKey(source)) {
        Frame top = this.path.peek();
        if (top != null) {
          top.reached = true;
          top.tableDepth = Math.min(top.tableDepth, depth);
        }
        TensorGenerator entry = table.get(source);
        return entry == null ? null : entry.copy();
      }
    }
    throw new IllegalStateException(source + " is in no component being solved.");
  }

  /**
   * How many times a dispatch on this path has read an entry of a component being solved, so an arm
   * can tell whether a candidate it dispatched read one.
   *
   * @return The count.
   */
  long tableReads() {
    return this.tableReads;
  }

  /**
   * Starts solving a component, whose members' dispatches then read their entries in the table.
   *
   * @param table The members' entries, which the factory updates as it solves.
   * @return The solve's depth, to pass to {@link #leaveSolved}.
   */
  int beginSolve(Map<PointsToSetVariable, TensorGenerator> table) {
    this.tables.add(table);
    return this.tables.size() - 1;
  }

  /** Ends the innermost solve {@link #beginSolve} started. */
  void endSolve() {
    this.tables.remove(this.tables.size() - 1);
  }

  /**
   * The dispatch this walk completed for a source, which depended on the source alone, for the
   * dispatch now on top of the path to replay.
   *
   * @param source The source.
   * @return The dispatch, or {@code null} when the walk keeps none for the source.
   */
  Dispatch kept(PointsToSetVariable source) {
    Kept kept = this.kept.get(source);
    if (kept == null) return null;
    if (kept.outside()) {
      REACHED_ON_THREAD.get()[0]++;
      Frame top = this.path.peek();
      if (top != null) top.reached = top.outside = true;
    }
    return kept.dispatch();
  }

  /**
   * Keeps a solved member's dispatch for the rest of the walk.
   *
   * @param source The member.
   * @param dispatch Its dispatch.
   * @param outside Whether solving it depended on a source that another path of the thread is
   *     resolving.
   */
  void keep(PointsToSetVariable source, Dispatch dispatch, boolean outside) {
    this.kept.put(source, new Kept(dispatch, outside));
  }

  /**
   * Enters a source that is not cut and for which the walk keeps no dispatch.
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
   * Leaves a source whose dispatch completed or threw.
   *
   * @param source The source, as passed to {@link #enter}.
   * @param generator The dispatch's generator, or {@code null} when there is none.
   * @param failure What the dispatch threw, or {@code null} when it returned.
   * @return The dispatch when it depended on the source alone, so the factory memoizes it per
   *     builder, and {@code null} otherwise. When the source completed a component, {@link
   *     #takeComponent} gives its members.
   */
  Dispatch leave(
      PointsToSetVariable source, TensorGenerator generator, IllegalArgumentException failure) {
    Frame frame = this.pop(source);
    Frame parent = this.path.peek();
    boolean completes =
        frame.low == frame.index && (frame.cycles || this.hasOpenAbove(frame.index));
    if (parent != null) {
      // The cuts inside a component the source completes are solved with it, so they do not make
      // the parent's dispatch hold for this path only.
      parent.reached |= completes ? frame.outside : frame.reached;
      parent.outside |= frame.outside;
      parent.tableDepth = Math.min(parent.tableDepth, frame.tableDepth);
      parent.low = Math.min(parent.low, frame.low);
    }
    if (frame.low < frame.index) {
      // A member of a component still open: cut at its later visits until the component is.
      this.stack.push(source);
      this.open.put(source, frame.index);
      return null;
    }
    if (completes) {
      List<PointsToSetVariable> members = new ArrayList<>();
      while (!this.stack.isEmpty() && this.open.get(this.stack.peek()) > frame.index) {
        PointsToSetVariable member = this.stack.pop();
        this.open.remove(member);
        members.add(member);
      }
      members.add(source);
      this.component = members;
      return null;
    }
    // A dispatch that read a component's entry holds only while that component is solved.
    if (frame.tableDepth != Integer.MAX_VALUE) return null;
    Dispatch dispatch = new Dispatch(generator == null ? null : generator.copy(), failure);
    this.kept.put(source, new Kept(dispatch, frame.outside));
    return frame.reached ? null : dispatch;
  }

  /**
   * What a solved member's dispatch depended on beyond its own component.
   *
   * @param outside Whether it depended on a source that another path of the thread is resolving.
   * @param enclosing Whether it read an entry of an enclosing component being solved.
   */
  record Solved(boolean outside, boolean enclosing) {}

  /**
   * Leaves a member of a component being solved, whose dispatch the factory keeps itself.
   *
   * @param source The member, as passed to {@link #enter}.
   * @param depth The solve's depth, as {@link #beginSolve} returned it.
   * @return What its dispatch depended on beyond its own component.
   */
  Solved leaveSolved(PointsToSetVariable source, int depth) {
    Frame frame = this.pop(source);
    // What a member reached inside its own component is solved; the rest is the parent's.
    boolean enclosing = frame.tableDepth < depth;
    Frame parent = this.path.peek();
    if (parent != null) {
      parent.outside |= frame.outside;
      if (frame.outside || enclosing) parent.reached = true;
      if (enclosing) parent.tableDepth = Math.min(parent.tableDepth, frame.tableDepth);
    }
    return new Solved(frame.outside, enclosing);
  }

  /**
   * The members of the component the source last left completed, once.
   *
   * @return The members, the source that completed it last, or {@code null} when it completed none.
   */
  List<PointsToSetVariable> takeComponent() {
    List<PointsToSetVariable> ret = this.component;
    this.component = null;
    return ret;
  }

  /**
   * Whether a source numbered above the given number completed in a component still open.
   *
   * @param index The number.
   * @return {@code true} iff one did.
   */
  private boolean hasOpenAbove(int index) {
    return !this.stack.isEmpty() && this.open.get(this.stack.peek()) > index;
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
}
