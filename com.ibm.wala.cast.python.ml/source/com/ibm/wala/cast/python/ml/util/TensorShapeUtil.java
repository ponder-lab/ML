package com.ibm.wala.cast.python.ml.util;

import static java.lang.Math.max;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class TensorShapeUtil {

  /**
   * One possible reading of a matmul's operand flags (<a
   * href="https://github.com/wala/ML/issues/1152">wala/ML#1152</a>): whether {@code a} and {@code
   * b} enter the product with their last two axes swapped. {@code transpose_x} and {@code
   * adjoint_x} swap the same two axes (the adjoint also conjugates, which leaves the shape alone),
   * so each operand's reading is their disjunction.
   *
   * @param transposeA Whether {@code a}'s last two axes are swapped.
   * @param transposeB Whether {@code b}'s last two axes are swapped.
   */
  public record MatMulReading(boolean transposeA, boolean transposeB) {

    /** The reading of a call that sets no flag. */
    public static final MatMulReading PLAIN = new MatMulReading(false, false);
  }

  /**
   * Composes a matmul result's shapes over the possible readings of its operand flags (<a
   * href="https://github.com/wala/ML/issues/1152">wala/ML#1152</a>): each reading swaps the last
   * two axes of the operands it transposes, then composes as {@link #matmulShape} does.
   *
   * <p>When the flags allow more than one reading, a reading whose contracted axes are two unequal
   * numeric extents is dropped, since that product raises at run time; if that drops every reading,
   * all of them are kept, so the result never claims more than the flags show. A single reading is
   * composed as is, as {@link #matmulShape} always has been.
   *
   * @param aDims The first operand's dimensions.
   * @param bDims The second operand's dimensions.
   * @param readings The possible flag readings, never empty.
   * @return The composed dimensions, one per reading that composes, empty when none does.
   */
  public static Set<List<Dimension<?>>> matmulShapes(
      List<Dimension<?>> aDims, List<Dimension<?>> bDims, Set<MatMulReading> readings) {
    Set<List<Dimension<?>>> feasible = new LinkedHashSet<>();
    Set<List<Dimension<?>>> all = new LinkedHashSet<>();
    for (MatMulReading reading : readings) {
      List<Dimension<?>> a = reading.transposeA() ? swapLastTwo(aDims) : aDims;
      List<Dimension<?>> b = reading.transposeB() ? swapLastTwo(bDims) : bDims;
      List<Dimension<?>> composed = matmulShape(a, b);
      if (composed == null) continue;
      all.add(composed);
      if (readings.size() == 1 || contractible(a, b)) feasible.add(composed);
    }
    return feasible.isEmpty() ? all : feasible;
  }

  /**
   * Whether a matmul pair's contracted axes may agree: {@code false} only when both are numeric
   * extents that differ.
   *
   * @param a The first operand's dimensions, as read.
   * @param b The second operand's dimensions, as read.
   * @return Whether the pair can execute as far as its contracted axes show.
   */
  private static boolean contractible(List<Dimension<?>> a, List<Dimension<?>> b) {
    Dimension<?> k = a.get(a.size() - 1);
    Dimension<?> l = b.get(b.size() - 2);
    return !(k instanceof NumericDim && l instanceof NumericDim && !k.equals(l));
  }

  /**
   * The dimensions with their last two axes swapped, or the dimensions themselves below rank two.
   *
   * @param dims The dimensions, or {@code null}.
   * @return The swapped dimensions.
   */
  private static List<Dimension<?>> swapLastTwo(List<Dimension<?>> dims) {
    if (dims == null || dims.size() < 2) return dims;
    List<Dimension<?>> ret = new ArrayList<>(dims);
    int n = ret.size();
    ret.set(n - 2, dims.get(n - 1));
    ret.set(n - 1, dims.get(n - 2));
    return ret;
  }

  public static boolean areBroadcastable(List<Dimension<?>> xShape, List<Dimension<?>> yShape) {
    int xRank = xShape.size();
    int yRank = yShape.size();
    int maxRank = max(xRank, yRank);

    for (int i = 0; i < maxRank; i++) {
      Dimension<?> xDim = i < (maxRank - xRank) ? null : xShape.get(i - (maxRank - xRank));
      Dimension<?> yDim = i < (maxRank - yRank) ? null : yShape.get(i - (maxRank - yRank));

      // Either side missing (out of rank), a raw-null placeholder, a ragged dim, a dynamic dim,
      // an unresolved dim, or a symbolic dim is treated as broadcast-compatible—we lack the
      // precision to reason about the actual extent, so be permissive.
      // https://github.com/wala/ML/issues/544 introduced `RaggedDim`;
      // https://github.com/wala/ML/issues/545 introduced `DynamicDim` for the runtime-`None` case
      // (batch/placeholder/Keras `None`); wala/ML#721 introduced `UnresolvedDim` for the
      // fixed-but-uncomputed case; wala/ML#741 added `SymbolicDim` (the unknown-size placeholder,
      // e.g. an uninferable reshape `-1`). All join the same permissive branch.
      if (xDim == null
          || yDim == null
          || xDim instanceof RaggedDim
          || yDim instanceof RaggedDim
          || xDim instanceof DynamicDim
          || yDim instanceof DynamicDim
          || xDim instanceof UnresolvedDim
          || yDim instanceof UnresolvedDim
          || xDim instanceof SymbolicDim
          || yDim instanceof SymbolicDim) {
        continue;
      }

      if (xDim instanceof NumericDim && yDim instanceof NumericDim) {
        int xSize = ((NumericDim) xDim).value();
        int ySize = ((NumericDim) yDim).value();

        if (xSize != ySize && xSize != 1 && ySize != 1) return false; // Incompatible sizes
      } else return false; // Non-numeric dimensions are incompatible
    }

    return true; // All dimensions are compatible
  }

  public static List<Dimension<?>> getBroadcastedShapes(
      List<Dimension<?>> xShape, List<Dimension<?>> yShape) {
    List<Dimension<?>> ret = new java.util.ArrayList<>();

    int xRank = xShape.size();
    int yRank = yShape.size();
    int maxRank = max(xRank, yRank);

    for (int i = 0; i < maxRank; i++) {
      Dimension<?> xDim = i < (maxRank - xRank) ? null : xShape.get(i - (maxRank - xRank));
      Dimension<?> yDim = i < (maxRank - yRank) ? null : yShape.get(i - (maxRank - yRank));

      // Propagate raggedness, dynamic-ness, unresolvedness, or symbolic-ness when either side
      // carries them—broadcasting against anything (including a compatible-rank counterpart)
      // preserves the wider unknown semantics. https://github.com/wala/ML/issues/544 introduced
      // `RaggedDim`; https://github.com/wala/ML/issues/545 introduced `DynamicDim`; wala/ML#721
      // introduced `UnresolvedDim`; wala/ML#741 added `SymbolicDim`. When two markers meet on the
      // same axis, the wider unknown dominates: ragged (varies per row) over dynamic (runtime
      // `None`) over unresolved (fixed but uncomputed) over symbolic (unknown size with no
      // provenance claim).
      if (xDim instanceof RaggedDim) ret.add(xDim);
      else if (yDim instanceof RaggedDim) ret.add(yDim);
      else if (xDim instanceof DynamicDim) ret.add(xDim);
      else if (yDim instanceof DynamicDim) ret.add(yDim);
      else if (xDim instanceof UnresolvedDim) ret.add(xDim);
      else if (yDim instanceof UnresolvedDim) ret.add(yDim);
      else if (xDim instanceof SymbolicDim) ret.add(xDim);
      else if (yDim instanceof SymbolicDim) ret.add(yDim);
      else if (xDim == null) ret.add(yDim);
      else if (yDim == null) ret.add(xDim);
      else if (xDim instanceof NumericDim && yDim instanceof NumericDim) {
        int xSize = ((NumericDim) xDim).value();
        int ySize = ((NumericDim) yDim).value();

        if (xSize == ySize) ret.add(xDim); // Both sizes are equal
        else if (xSize == 1) ret.add(yDim); // x is broadcasted
        else if (ySize == 1) ret.add(xDim); // y is broadcasted
        else throw new IllegalArgumentException("Incompatible dimensions for broadcasting.");
      } else throw new IllegalArgumentException("Non-numeric dimensions cannot be broadcasted.");
    }

    return ret;
  }

  /**
   * Composes a matmul result's shape: {@code (..., m, k)} and {@code (..., k, n)} yield {@code
   * (..., m, n)}, with the leading batch axes BROADCAST pairwise from the right rather than taken
   * from one operand (<a href="https://github.com/wala/ML/issues/878">wala/ML#878</a>).
   *
   * <p>Taking the higher-rank operand's prefix is wrong wherever TensorFlow would broadcast. Equal
   * rank is the case it is least safe for, not the most: {@code (3, 1, m, k)} and {@code (1, 5, k,
   * n)} are both rank 4 and broadcast to {@code (3, 5, m, n)}, where taking the first operand's
   * prefix yields {@code (3, 1, m, n)}, a wrong extent on two axes rather than an unknown one.
   *
   * <p>This is the single implementation both the generator and the type-feed composition call, so
   * one operation cannot get two answers depending on which path resolved it. That mirroring was
   * introduced deliberately (wala/ML#877) and sharing the rule keeps it exact rather than
   * by-inspection.
   *
   * @param aDims The first operand's dimensions.
   * @param bDims The second operand's dimensions.
   * @return The composed dimensions, or {@code null} when the operation cannot be composed: either
   *     operand below rank two, or batch prefixes that are not broadcastable, which would fail at
   *     run time and so is declined rather than resolved to one side.
   */
  public static List<Dimension<?>> matmulShape(List<Dimension<?>> aDims, List<Dimension<?>> bDims) {
    if (aDims == null || bDims == null || aDims.size() < 2 || bDims.size() < 2) return null;
    List<Dimension<?>> aBatch = aDims.subList(0, aDims.size() - 2);
    List<Dimension<?>> bBatch = bDims.subList(0, bDims.size() - 2);
    List<Dimension<?>> ret = new java.util.ArrayList<>();
    // Equal prefixes need no broadcasting and compose to themselves whatever their dimension
    // kinds are. Taking this first keeps every shape that already resolved resolving: the defect
    // this rule fixes is prefixes that DIFFER, and a pairwise rule that cannot decide some kind
    // must not degrade the identical case on the way past.
    if (aBatch.equals(bBatch)) ret.addAll(aBatch);
    else {
      int aRank = aBatch.size();
      int bRank = bBatch.size();
      int rank = max(aRank, bRank);
      for (int i = 0; i < rank; i++) {
        Dimension<?> x = i < rank - aRank ? null : aBatch.get(i - (rank - aRank));
        Dimension<?> y = i < rank - bRank ? null : bBatch.get(i - (rank - bRank));
        // A missing axis on the shorter side behaves as 1, so the other side wins outright.
        if (x == null) ret.add(y);
        else if (y == null) ret.add(x);
        else if (x.equals(y)) ret.add(x);
        else if (x instanceof NumericDim && y instanceof NumericDim) {
          int xSize = ((NumericDim) x).value();
          int ySize = ((NumericDim) y).value();
          if (xSize == 1) ret.add(y);
          else if (ySize == 1) ret.add(x);
          // Two unequal extents, neither of them 1, would fail at run time. Declining is the
          // honest result; picking a side is what made this rule confidently wrong.
          else return null;
        }
        // Neither side decides: one axis carries a marker or a kind the pairwise rule cannot
        // compare. Degrade that axis rather than guess, keeping the wider unknown when the two
        // markers differ, in the wala/ML#544/#545/#721/#741 dominance order.
        else if (x instanceof RaggedDim || y instanceof RaggedDim)
          ret.add(x instanceof RaggedDim ? x : y);
        else if (x instanceof DynamicDim || y instanceof DynamicDim)
          ret.add(x instanceof DynamicDim ? x : y);
        else if (x instanceof UnresolvedDim || y instanceof UnresolvedDim)
          ret.add(UnresolvedDim.INSTANCE);
        // A placeholder size (`?`, wala/ML#741) against a numeric extent broadcasts to the
        // placeholder, as `getBroadcastedShapes` reads the same pair: the placeholder already says
        // the size is unknown, and the numeric side is either that size or 1. Read as an unresolved
        // size, the pair claimed a fixed extent it never established, and the claim reached a
        // batched value's own members: its concrete batch paired with its partial-batch sibling
        // minted a third batch extent the value never has.
        else if (x instanceof SymbolicDim || y instanceof SymbolicDim)
          ret.add(x instanceof SymbolicDim ? x : y);
        else ret.add(UnresolvedDim.INSTANCE);
      }
    }
    ret.add(aDims.get(aDims.size() - 2));
    ret.add(bDims.get(bDims.size() - 1));
    return ret;
  }
}
