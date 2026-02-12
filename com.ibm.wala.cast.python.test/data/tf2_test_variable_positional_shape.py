import tensorflow as tf


def f(v):
    pass


# initial_value, trainable, validate_shape, caching_device, name, variable_def, dtype, import_scope, constraint, synchronization, aggregation, shape
# explicit shape has unknown dimension [None, 2], initial_value is [2, 2].
# TensorFlow preserves the None dimension.
v1 = tf.Variable(
    [[1.0, 2.0], [3.0, 4.0]],
    True,
    True,
    None,
    "v1",
    None,
    tf.float32,
    None,
    None,
    tf.VariableSynchronization.AUTO,
    tf.VariableAggregation.NONE,
    [None, 2],
)
assert isinstance(v1, tf.Variable)
# Verify that the explicit shape argument was respected (it overrides the fully defined shape of initial_value)
assert v1.shape.as_list() == [None, 2]
assert v1.dtype == tf.float32

f(v1)
