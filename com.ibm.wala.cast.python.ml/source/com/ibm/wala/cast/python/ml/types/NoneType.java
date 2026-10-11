package com.ibm.wala.cast.python.ml.types;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;

/**
 * The type of Python's {@code None} in a parameter's state, held where {@code None} reaches a
 * parameter that receives tensors in some context (<a
 * href="https://github.com/wala/ML/issues/1040">wala/ML#1040</a>). It records that the parameter's
 * values are not all tensors, so a client deriving a signature from the parameter's types can see
 * that a call passes {@code None} there.
 *
 * <p>It is not a tensor type. It is added to parameter states only after the tensor type analysis
 * has been solved, so no transfer function, dtype join or shape operation reads it. Its dtype is
 * {@link DType#UNKNOWN} and its dims are {@code null} only because every {@link TensorType} has
 * them; a client tests {@link #isNone()} before reading either. It never equals another {@link
 * TensorType}, since {@link TensorType#equals(Object)} compares classes.
 */
public final class NoneType extends TensorType {

  /** The one {@code None} member. */
  public static final NoneType INSTANCE = new NoneType();

  private NoneType() {
    super(DType.UNKNOWN, null);
  }

  @Override
  public boolean isNone() {
    return true;
  }

  @Override
  public String toString() {
    return "None";
  }

  @Override
  public String toCString(boolean useMarkdown) {
    return "None";
  }

  /**
   * Describes {@code None} as JSON Schema's {@code null} type, rather than as a tensor of unknown
   * shape and dtype.
   *
   * @return The schema.
   */
  @Override
  public JsonElement toJsonSchema() {
    JsonObject schema = new JsonObject();
    schema.addProperty("type", "null");
    return schema;
  }
}
