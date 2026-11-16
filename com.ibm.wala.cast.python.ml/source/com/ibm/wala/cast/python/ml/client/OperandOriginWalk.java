package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.Pair;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The operands whose origins an {@link ElementWiseOperation} is classifying on a thread, what each
 * classification depended on, and the classifications the walk has completed.
 *
 * <p>An operand that is the result of a binary operator classifies through that operator's
 * operands, and one already being classified contributes no evidence, so the other operand decides
 * (wala/ML#599). Nothing was kept once an operand was classified, so an operand of two operators
 * was classified once for each, and so was everything below it: a chain of operators that share
 * operands took time exponential in its length (wala/ML#1275).
 *
 * <p>So a walk, from the first operand entered on the thread until that operand is left, keeps each
 * classification it completes and replays it at the operand's later visits. A classification that
 * was cut at an operand entered before its own holds only while that operand is being classified,
 * as a dispatch cut at a source on its path does (see {@link DispatchPath}, wala/ML#1274), so the
 * walk is the same depth-first search for strongly connected components: each operand entered is
 * numbered, its classification records the lowest number it depended on, and the classifications
 * that depended on a number below their own are dropped when the operand with that number is left.
 *
 * <p>Nothing is kept past the walk. An operand's origins are read while the analysis resolves types
 * ({@link TensorGenerator#readsNumpyDTypeTokens}), where what a generator reads may still change,
 * and a classification kept per builder would not change with it.
 */
final class OperandOriginWalk {

  /** The walk of each thread. */
  private static final ThreadLocal<OperandOriginWalk> WALKS =
      ThreadLocal.withInitial(OperandOriginWalk::new);

  /**
   * How many classifications were computed for each builder, rather than replayed. The counts hold
   * nothing of the builder, so its entry expires with it.
   */
  private static final Map<PropagationCallGraphBuilder, AtomicLong> COMPUTATIONS =
      new WeakHashMap<>();

  /** An operand being classified. */
  private static final class Frame {

    /** The operand. */
    private final Pair<CGNode, Integer> operand;

    /** The operand's number, in the order the walk entered its operands. */
    private final int index;

    /** The lowest number the operand's classification has depended on so far, at most its own. */
    private int low;

    private Frame(Pair<CGNode, Integer> operand, int index) {
      this.operand = operand;
      this.index = index;
      this.low = index;
    }
  }

  /**
   * A completed classification of this walk.
   *
   * @param origins The operand's origins, or {@code null} when the operand contributes no origin
   *     evidence.
   * @param index The operand's number.
   * @param provisional Whether the classification depended on an operand entered before its own, so
   *     it holds until that operand is left.
   */
  record Kept(Set<TensorOrigin> origins, int index, boolean provisional) {

    /**
     * The origins, in a set of the caller's own.
     *
     * @return A copy of {@link #origins}, or {@code null} when there are none.
     */
    Set<TensorOrigin> copy() {
      if (this.origins == null) return null;
      Set<TensorOrigin> ret = EnumSet.noneOf(TensorOrigin.class);
      ret.addAll(this.origins);
      return ret;
    }
  }

  /** The operands being classified, with their frames. */
  private final Map<Pair<CGNode, Integer>, Frame> frames = new HashMap<>();

  /** The frames of the operands being classified, the last entered first. */
  private final Deque<Frame> path = new ArrayDeque<>();

  /** The completed classifications of this walk, by operand. */
  private final Map<Pair<CGNode, Integer>, Kept> kept = new HashMap<>();

  /** The operands of the provisional classifications among {@link #kept}, the last first. */
  private final Deque<Pair<CGNode, Integer>> provisional = new ArrayDeque<>();

  /** How many operands this walk has entered. */
  private int entered;

  OperandOriginWalk() {}

  /**
   * The walk of the current thread.
   *
   * @return The walk.
   */
  static OperandOriginWalk current() {
    return WALKS.get();
  }

  /**
   * How many classifications were computed for a builder, rather than replayed.
   *
   * @param builder The builder.
   * @return The count.
   */
  static long computations(PropagationCallGraphBuilder builder) {
    synchronized (COMPUTATIONS) {
      AtomicLong computations = COMPUTATIONS.get(builder);
      return computations == null ? 0 : computations.get();
    }
  }

  /**
   * Whether an operand is being classified on this thread.
   *
   * @return {@code true} iff the walk has entered an operand it has not left.
   */
  boolean inProgress() {
    return !this.path.isEmpty();
  }

  /**
   * Whether an operand is being classified, so its classification is cut. The classification now on
   * top of the path then depends on the operand's own, and holds until the operand is left.
   *
   * @param operand The operand.
   * @return {@code true} iff the operand is being classified.
   */
  boolean cuts(Pair<CGNode, Integer> operand) {
    Frame entered = this.frames.get(operand);
    if (entered == null) return false;
    // An operand being classified means the path has a top.
    Frame top = this.path.element();
    top.low = Math.min(top.low, entered.index);
    return true;
  }

  /**
   * The classification this walk completed for an operand that is not being classified, for the
   * classification now on top of the path to replay. A provisional one makes that classification
   * depend on what it depended on, as a cut does.
   *
   * @param operand The operand.
   * @return The classification, or {@code null} when the walk keeps none for the operand.
   */
  Kept kept(Pair<CGNode, Integer> operand) {
    Kept kept = this.kept.get(operand);
    if (kept == null) return null;
    // A provisional classification is kept only while an operand below its own is on the path.
    if (kept.provisional()) {
      Frame top = this.path.element();
      top.low = Math.min(top.low, kept.index());
    }
    return kept;
  }

  /**
   * Enters an operand that is not being classified, for which the walk keeps no classification.
   *
   * @param builder The builder the operand is classified for.
   * @param operand The operand.
   */
  void enter(PropagationCallGraphBuilder builder, Pair<CGNode, Integer> operand) {
    synchronized (COMPUTATIONS) {
      COMPUTATIONS.computeIfAbsent(builder, b -> new AtomicLong()).incrementAndGet();
    }
    Frame frame = new Frame(operand, this.entered++);
    this.frames.put(operand, frame);
    this.path.push(frame);
  }

  /**
   * Leaves an operand whose classification completed, keeping the classification for the rest of
   * the walk, or, when it depended on an operand entered before its own, until that operand is
   * left.
   *
   * @param operand The operand, as passed to {@link #enter}.
   * @param origins The operand's origins, or {@code null} when it contributes no origin evidence.
   */
  void leave(Pair<CGNode, Integer> operand, Set<TensorOrigin> origins) {
    Frame frame = this.pop(operand);
    if (this.path.isEmpty()) return;
    boolean provisional = frame.low < frame.index;
    if (provisional) {
      Frame parent = this.path.element();
      parent.low = Math.min(parent.low, frame.low);
    } else
      // The classifications that depended on this operand, or on one of them, held while it was
      // being classified.
      this.dropProvisionalSince(frame.index);
    Set<TensorOrigin> copy = null;
    if (origins != null) {
      copy = EnumSet.noneOf(TensorOrigin.class);
      copy.addAll(origins);
    }
    this.kept.put(operand, new Kept(copy, frame.index, provisional));
    if (provisional) this.provisional.push(operand);
  }

  /**
   * Leaves an operand whose classification did not complete. Nothing is kept for it.
   *
   * @param operand The operand, as passed to {@link #enter}.
   */
  void abandon(Pair<CGNode, Integer> operand) {
    Frame frame = this.pop(operand);
    if (!this.path.isEmpty()) this.dropProvisionalSince(frame.index);
  }

  /**
   * Takes the operand on top of the path off it, and ends the walk when it was the first entered.
   *
   * @param operand The operand.
   * @return Its frame.
   */
  private Frame pop(Pair<CGNode, Integer> operand) {
    Frame frame = this.path.pop();
    assert frame.operand.equals(operand) : "Leaving " + operand + " under " + frame.operand + ".";
    this.frames.remove(operand);
    if (this.path.isEmpty()) {
      this.kept.clear();
      this.provisional.clear();
      this.entered = 0;
    }
    return frame;
  }

  /**
   * Drops the provisional classifications of the operands entered after an operand.
   *
   * @param index The operand's number.
   */
  private void dropProvisionalSince(int index) {
    while (!this.provisional.isEmpty() && this.kept.get(this.provisional.peek()).index() > index)
      this.kept.remove(this.provisional.pop());
  }
}
