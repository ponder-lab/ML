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


def consume_full(x):
    pass


def consume_dynamic(x):
    pass


# A full slice constrains nothing, so the source's extent is still the right answer and must be
# carried through rather than degraded.
full = y[:]
assert full.shape == (4096,), full.shape
consume_full(full)

# Slicing an axis that is already None-evidenced: the degraded extent stays None-evidenced rather
# than becoming a fixed-but-uncomputed size.
dyn = tf.keras.Input(shape=(3,), dtype=tf.uint8)
assert dyn.shape.as_list() == [None, 3], dyn.shape
dyn_sliced = dyn[i * gpu_batch_size : (i + 1) * gpu_batch_size]
assert dyn_sliced.shape.as_list() == [None, 3], dyn_sliced.shape
consume_dynamic(dyn_sliced)
