import tensorflow as tf


def f(a):
    pass


# Regression guard for the `tf.broadcast_to(x, tf.shape(y))` shape-arg
# pattern. At runtime, `z` has shape (2, 3) of float32. The analyzer's
# `getShapesFromShapeArgument` helper doesn't recognize a runtime-tensor
# `shape` arg (it expects a list/tuple/tf.constant/TensorSpec), so it
# throws `IllegalStateException`. PR ponder-lab/ML#245 localizes the
# tolerance to `BroadcastTo.getDefaultShapes` via a try/catch that
# returns null (lattice ⊤, "tensor of unknown shape") rather than
# aborting the analysis. This fixture fires that catch path; without
# the catch, analysis aborts and the JUnit test fails.
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
z = tf.broadcast_to(x, tf.shape(y))
assert isinstance(z, tf.Tensor)
assert z.shape == (2, 3)
assert z.dtype == tf.float32
f(z)
