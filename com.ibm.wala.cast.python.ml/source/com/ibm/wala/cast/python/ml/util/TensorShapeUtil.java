package com.ibm.wala.cast.python.ml.util;

import static java.lang.Math.max;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.DynamicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.RaggedDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import java.util.List;

public class TensorShapeUtil {

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
        else ret.add(UnresolvedDim.INSTANCE);
      }
    }
    ret.add(aDims.get(aDims.size() - 2));
    ret.add(bDims.get(bDims.size() - 1));
    return ret;
  }
}
