import tensorflow as tf


def f(a):
    pass


# Broadcasting-ellipsis einsum: the generator doesn't model `...`, so the
# analysis soundly reports an unknown (⊤) shape while keeping the dtype
# precise. The shape assert below documents the Python runtime truth, not the
# analysis result; `testEinsumEllipsisFallback` pins the analysis-side ⊤ until
# wala/ML#705 models the ellipsis form.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
result = tf.einsum("...ij,...jk->...ik", a, b)
assert isinstance(result, tf.Tensor)
assert result.shape == (2, 2)
assert result.dtype == tf.float32
f(result)
