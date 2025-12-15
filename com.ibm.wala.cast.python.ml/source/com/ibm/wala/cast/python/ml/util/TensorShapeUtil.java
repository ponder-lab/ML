package com.ibm.wala.cast.python.ml.util;

import static java.lang.Math.max;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import java.util.List;

public class TensorShapeUtil {

  public static boolean areBroadcastable(List<Dimension<?>> xShape, List<Dimension<?>> yShape) {
    int xRank = xShape.size();
    int yRank = yShape.size();
    int maxRank = max(xRank, yRank);

    for (int i = 0; i < maxRank; i++) {
      Dimension<?> xDim = i < (maxRank - xRank) ? null : xShape.get(i - (maxRank - xRank));
      Dimension<?> yDim = i < (maxRank - yRank) ? null : yShape.get(i - (maxRank - yRank));

      if (xDim == null || yDim == null) {
        continue; // One of the dimensions is missing, treat as size 1
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

      if (xDim == null) ret.add(yDim);
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
