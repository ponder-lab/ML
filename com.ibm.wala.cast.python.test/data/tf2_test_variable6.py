import tensorflow as tf


def f(v):
    pass


# initial_value, trainable, validate_shape, caching_device, name, variable_def, dtype
# initial_value is int32, explicit dtype is float64. Should cast.
v1 = tf.Variable([1, 2], True, True, None, "v1", None, tf.float64)
assert isinstance(v1, tf.Variable)
assert v1.shape.as_list() == [2]
assert v1.dtype == tf.float64

f(v1)
