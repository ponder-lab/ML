import tensorflow as tf


def f(a):
    pass


# Diagonal einsum (a repeated label within one term): the generator doesn't
# model diagonal extraction, so the analysis soundly reports an unknown (⊤)
# shape while keeping the dtype precise. The shape assert below documents the
# Python runtime truth, not the analysis result; `testEinsumDiagonalFallback`
# pins the analysis-side ⊤ until wala/ML#705 models the diagonal form.
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
result = tf.einsum("ii->i", a)
assert isinstance(result, tf.Tensor)
assert result.shape == (2,)
assert result.dtype == tf.float32
f(result)
