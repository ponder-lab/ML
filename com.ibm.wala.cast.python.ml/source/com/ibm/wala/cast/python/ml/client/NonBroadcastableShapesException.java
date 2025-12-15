package com.ibm.wala.cast.python.ml.client;

import static java.lang.String.format;

import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import java.util.List;

/**
 * An exception indicating that two shapes are not broadcastable for a given operation.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">NumPy Broadcasting</a>.
 */
public class NonBroadcastableShapesException extends RuntimeException {

  /** Serial version UID. */
  private static final long serialVersionUID = 805036824027449575L;

  /** The operation for which the shapes are not broadcastable. */
  private final transient Object op;

  /** The first shape. */
  private final transient List<Dimension<?>> xShape;

  /** The second shape. */
  private final transient List<Dimension<?>> yShape;

  /**
   * Constructs a new exception indicating that the given shapes are not broadcastable for the given
   * operation.
   *
   * @param op The operation for which the shapes are not broadcastable.
   * @param xShape The first shape.
   * @param yShape The second shape.
   */
  public NonBroadcastableShapesException(
      Object op, List<Dimension<?>> xShape, List<Dimension<?>> yShape) {
    this.op = op;
    this.xShape = xShape;
    this.yShape = yShape;
  }

  @Override
  public String getMessage() {
    return format("The shapes %s and %s are not broadcastable for %s.", xShape, yShape, op);
  }
}
