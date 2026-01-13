import tensorflow as tf


def f(v):
    pass


# initial_value, trainable, validate_shape, caching_device, name, variable_def, dtype, import_scope, constraint, synchronization, aggregation, shape
v1 = tf.Variable(
    [1.0, 2.0, 3.0],
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
    [3],
)
assert isinstance(v1, tf.Variable)
assert v1.shape.as_list() == [3]
assert v1.dtype == tf.float32

f(v1)
