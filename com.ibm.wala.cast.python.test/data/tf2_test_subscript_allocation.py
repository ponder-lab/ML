# The result of subscripting an array or a tensor with an index tuple that holds `...`, `None`
# (`tf.newaxis`) or a slice is a new array or tensor (https://github.com/wala/ML/issues/400).
import numpy as np
import tensorflow as tf


def consume_ellipsis_newaxis(x):
    assert x.shape == (3, 2, 1) and x.dtype == np.float32


def consume_ellipsis_none(x):
    assert x.shape == (3, 2, 1) and x.dtype == np.float32


def consume_none_ellipsis(x):
    assert x.shape == (1, 3, 2) and x.dtype == np.float32


def consume_slice_ellipsis_newaxis(x):
    assert x.shape == (2, 2, 1) and x.dtype == np.float32


def consume_slice_none(x):
    assert x.shape == (3, 1) and x.dtype == np.float32


def consume_tensor_ellipsis_newaxis(x):
    assert x.shape == (3, 2, 1) and x.dtype == tf.float32


def consume_tensor_slice_ellipsis_newaxis(x):
    assert x.shape == (2, 2, 1) and x.dtype == tf.float32


a = np.ones((3, 2), dtype=np.float32)
consume_ellipsis_newaxis(a[..., tf.newaxis])
consume_ellipsis_none(a[..., None])
consume_none_ellipsis(a[None, ...])
consume_slice_ellipsis_newaxis(a[:2, ..., tf.newaxis])
consume_slice_none(a[:, 0, None])

t = tf.ones((3, 2))
consume_tensor_ellipsis_newaxis(t[..., tf.newaxis])
consume_tensor_slice_ellipsis_newaxis(t[:2, ..., tf.newaxis])
