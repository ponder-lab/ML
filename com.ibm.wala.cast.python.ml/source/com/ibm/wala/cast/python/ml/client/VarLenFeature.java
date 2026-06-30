package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A generator for the SparseTensor a {@code tf.io.VarLenFeature(dtype)} represents: when a
 * variable-length feature is parsed (e.g. by {@code tf.io.parse_single_example}) it yields a {@code
 * tf.sparse.SparseTensor} whose values carry the feature's dtype and whose dense shape is dynamic
 * (⊤). Modeling it as that SparseTensor lets {@code parse_single_example} (a pass-through of its
 * feature dict) and a downstream {@code tf.sparse.to_dense} type the parsed value (wala/ML#645).
 *
 * <p>{@code dtype} is the sole argument and there is no {@code shape} argument, so the base {@link
 * TensorTypeAllocator} reads the dtype slot and falls to its ⊤-shape default.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/io/VarLenFeature">tf.io.VarLenFeature</a>.
 */
public class VarLenFeature extends TensorTypeAllocator {

  @Override
  protected boolean producesSparseTensor() {
    return true;
  }

  protected enum Parameters {
    DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public VarLenFeature(PointsToSetVariable source) {
    super(source);
  }

  /** No {@code shape} argument: a variable-length feature has a dynamic (⊤) dense shape. */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /** The dtype is the sole argument. */
  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
