package com.ibm.wala.cast.python.ml.client;

/**
 * An interface for tensor generators that delegate type and shape inference to an underlying
 * generator. Classes implementing this interface typically act as proxies or wrappers around
 * another generator.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public interface DelegatingTensorGenerator {

  /**
   * Retrieves the underlying generator to which this generator delegates.
   *
   * @return the underlying generator.
   */
  TensorGenerator getUnderlying();
}
