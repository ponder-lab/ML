package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import java.util.Locale;

/**
 * A generator for the sparse identity matrix produced by {@code tf.sparse.eye}. The {@code
 * num_rows} &times; {@code num_columns} shape construction lives in {@link EyeBase}; this subclass
 * supplies the sparse signature's {@code dtype} parameter position (no {@code batch_shape}, unlike
 * {@link Eye}).
 *
 * @see <a href="https://www.tensorflow.org/api_docs/python/tf/sparse/eye">tf.sparse.eye API</a>.
 */
public class SparseEye extends EyeBase {

  protected enum Parameters {
    NUM_ROWS,
    NUM_COLUMNS,
    DTYPE,
    NAME;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public SparseEye(PointsToSetVariable source) {
    super(source);
  }

  public SparseEye(CGNode node) {
    super(node);
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }
}
