package com.ibm.wala.cast.python.ml.types;

/**
 * The library whose operation produced a tensor-typed value (wala/ML#724).
 *
 * <p>The tensor type analysis intentionally types numpy arrays as tensors: an ndarray is
 * tensor-convertible, so tracking what can flow into a tensor parameter requires typing it. The
 * resulting {@link TensorType} carries no record of which library produced the value, but some
 * consumers must distinguish a value produced by a numpy operation (convertible, but not yet a
 * TensorFlow computation) from one produced by a TensorFlow operation. This enum is that record; it
 * is seeded per dataflow source from the dispatched {@link
 * com.ibm.wala.cast.python.ml.client.TensorGenerator} and propagated along the same dataflow edges
 * as the tensor types, so a control-flow merge of both origins conservatively carries both
 * constants.
 *
 * <p>Classification is by the <em>runtime type of the produced value</em>, not the namespace of the
 * invoked API: a mixed binary operator ({@code ndarray + Tensor}) dispatches to TensorFlow and
 * yields a {@code tf.Tensor}, so it is {@link #TENSORFLOW}; {@code tf.keras.datasets}' {@code
 * load_data} returns ndarrays, so its results are {@link #NUMPY}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public enum TensorOrigin {

  /** The value is an ndarray produced by a numpy operation. */
  NUMPY,

  /** The value is a tensor produced by a TensorFlow operation. */
  TENSORFLOW
}
