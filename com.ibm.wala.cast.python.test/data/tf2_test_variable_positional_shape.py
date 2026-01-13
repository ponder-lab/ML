import tensorflow as tf


def f(v):
    pass


# initial_value, trainable, validate_shape, caching_device, name, variable_def, dtype, import_scope, constraint, synchronization, aggregation, shape
# Note: This runtime call might fail due to shape mismatch checks in TensorFlow,
# but the static analysis should prioritize the explicit 'shape' argument [2, 2].
v1 = tf.Variable(
    [1.0, 2.0, 3.0, 4.0],
    True,
    False,
    None,
    "v1",
    None,
    tf.float32,
    None,
    None,
    tf.VariableSynchronization.AUTO,
    tf.VariableAggregation.NONE,
    [2, 2],
)

# If analysis works, it reports [2, 2].
f(v1)
