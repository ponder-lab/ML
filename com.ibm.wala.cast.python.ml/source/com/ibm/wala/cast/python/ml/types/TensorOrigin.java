package com.ibm.wala.cast.python.ml.types;

/**
 * The provenance of a tensor-typed value: the library whose operation produced it, or the function
 * boundary it crossed (wala/ML#724, wala/ML#726).
 *
 * <p>The tensor type analysis intentionally types numpy arrays as tensors: an ndarray is
 * tensor-convertible, so tracking what can flow into a tensor parameter requires typing it. The
 * resulting {@link TensorType} carries no record of which library produced the value, but some
 * consumers must distinguish a value produced by a numpy operation (convertible, but not yet a
 * TensorFlow computation) from one produced by a TensorFlow operation. This enum is that record; it
 * is seeded per dataflow source from the dispatched {@link
 * com.ibm.wala.cast.python.ml.client.TensorGenerator} and propagated along the same dataflow edges
 * as the tensor types, so a control-flow merge of different origins conservatively carries all the
 * merged constants.
 *
 * <p>Classification is by the <em>runtime type of the produced value</em>, not the namespace of the
 * invoked API: a mixed binary operator ({@code ndarray + Tensor}) dispatches to TensorFlow and
 * yields a {@code tf.Tensor}, so it is {@link #TENSORFLOW}; {@code tf.keras.datasets}' {@code
 * load_data} returns ndarrays, so its results are {@link #NUMPY}.
 *
 * <p>Function parameters are the one exception to producer classification: a parameter reads {@link
 * #PARAMETER} regardless of what its call sites feed it (wala/ML#726).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public enum TensorOrigin {

  /** The value is an ndarray produced by a numpy operation. */
  NUMPY,

  /** The value is a tensor produced by a TensorFlow operation. */
  TENSORFLOW,

  /**
   * The value is a function parameter: the hybridization-frame origin (wala/ML#726). Under {@code
   * tf.function} tracing a tensor parameter is a symbolic tensor regardless of the library that
   * produced its eager feeds, so parameter provenance is first-class rather than inherited: the
   * parameter boundary blocks caller-side origin inflow, and a value derived from a parameter
   * carries this constant rather than the feeds' libraries.
   */
  PARAMETER
}
