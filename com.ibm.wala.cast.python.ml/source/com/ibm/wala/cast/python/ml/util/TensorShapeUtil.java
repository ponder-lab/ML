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
}
