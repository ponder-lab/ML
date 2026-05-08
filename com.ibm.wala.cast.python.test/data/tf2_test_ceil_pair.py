import tensorflow as tf


def f(a, b):
    pass


# 2-arg sink variant of `tf2_test_ceil.py` — same input and op, but with one combined sink
# `f(y, x)` instead of two separate single-arg sinks `f(y); g(x)`. This is the same shape as
# the wala/ML#495 multi-tensor-sink pattern, the difference being that #495 is specifically
# about dataset-loader outputs (`fashion_mnist`/`cifar100`/etc.) flowing through the
# `TensorGenerator.shapesFromSSAChain` fallback paths. For `ceil` on `tf.constant`, no
# fallback is involved, so the pattern works precisely today: the JUnit companion
# `testCeilPair` asserts the lattice-correct `(3,) float32` on both params. The fixture
# stands as a canary — if #495 ever generalizes beyond dataset loaders to per-op generators,
# this test will start failing.
x = tf.constant([1.0, 2.0, 3.0])
assert x.shape == (3,)
assert x.dtype == tf.float32
y = tf.math.ceil(x)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.float32
f(y, x)
