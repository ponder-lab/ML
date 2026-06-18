package com.ibm.wala.cast.python.ml.types;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import java.util.List;

/**
 * A {@link TensorType} whose tensor is stored sparsely (a {@code tf.sparse.SparseTensor} / {@code
 * tf.SparseTensor}), as opposed to dense. Sparseness is a tensor-level storage property orthogonal
 * to the dense shape, so it is modeled as this subtype rather than a {@link Dimension}; the dense
 * shape lives in {@link #getDims()} unchanged. {@link #layout()} reports {@link Layout#SPARSE} so a
 * consumer can emit the appropriate sparse spec. See <a
 * href="https://github.com/wala/ML/issues/588">wala/ML#588</a>.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SparseTensorType extends TensorType {

  /**
   * Constructs a sparse tensor type.
   *
   * @param dtype The tensor element type. Must not be null.
   * @param dims The dense dimensions of the tensor; may be null to indicate unknown rank (⊤ shape).
   */
  public SparseTensorType(DType dtype, List<Dimension<?>> dims) {
    super(dtype, dims);
  }

  @Override
  public Layout layout() {
    return Layout.SPARSE;
  }
}
