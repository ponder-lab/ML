# Regression guard for `tf.reshape(x, tf.shape(y))` shape inference (wala/ML#489).
#
# At runtime, `z` has shape (2, 3) of float32. Post the wala/ML#489 root-cause fix on this
# PR's `tensorflow.xml` (separating `tf.shape`'s allocation from `pass_through` aliasing),
# the malformed compound-dim shape that previously surfaced here is gone. The analyzer now
# produces a union of the input x's shape ((6,)) and a scalar — neither the precise (2, 3)
# answer nor a sound ⊤, but no longer malformed. Reshape doesn't have a try/catch around the
# helper (unlike BroadcastTo), so the exception propagates to upstream fallback paths.
#
# The corresponding JUnit test asserts the currently-observed shape with a TODO referencing
# #489; when Reshape gains a precise composer or a try/catch, the assertion will start
# failing and the TODO is the cue to tighten it.
import tensorflow as tf


def f(a):
    pass


x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # shape (2, 3)
z = tf.reshape(x, tf.shape(y))  # runtime: shape (2, 3) of float32
assert isinstance(z, tf.Tensor)
assert z.shape == (2, 3)
assert z.dtype == tf.float32
f(z)
