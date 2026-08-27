import tensorflow as tf


def consume_computed(x):
    pass


def consume_literal(x):
    pass


def consume_opaque(x, n):
    pass


num_gpus = 4
batch_size = 1024 * num_gpus
gpu_batch_size = int(batch_size / num_gpus)

y = tf.ones((batch_size,), dtype=tf.uint8)
assert y.shape == (4096,), y.shape

# Computed bounds: the slice length is a constant expression, and the result is shorter than the
# receiver. Carrying the receiver's extent forward would report 4096 here.
i = 0
computed = y[i * gpu_batch_size : (i + 1) * gpu_batch_size]
assert computed.shape == (1024,), computed.shape
consume_computed(computed)

# Literal bounds, as a control: the same shortening written with literals.
literal = y[0:1024]
assert literal.shape == (1024,), literal.shape
consume_literal(literal)
