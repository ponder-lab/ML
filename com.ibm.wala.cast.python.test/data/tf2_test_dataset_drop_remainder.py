import tensorflow as tf


def consume_dropped(x):
    pass


def consume_kept(x):
    pass


def consume_padded_dropped(x):
    pass


def consume_padded_kept(x):
    pass


def consume_positional(x):
    pass


def consume_padded_positional(x):
    pass


# Ten elements batched by four: two full batches and a final partial batch of two.
ds = tf.data.Dataset.from_tensor_slices(tf.ones((10, 3)))

# `drop_remainder=True` discards the partial final batch, so every batch is full.
for dropped in ds.batch(4, drop_remainder=True):
    assert dropped.shape == (4, 3), dropped.shape
    consume_dropped(dropped)

# Without it the partial final batch reaches the consumer.
kept_shapes = []
for kept in ds.batch(4):
    kept_shapes.append(tuple(kept.shape))
    consume_kept(kept)

assert kept_shapes == [(4, 3), (4, 3), (2, 3)], kept_shapes

# `padded_batch` takes the same flag at a different argument position.
for padded_dropped in ds.padded_batch(4, drop_remainder=True):
    assert padded_dropped.shape == (4, 3), padded_dropped.shape
    consume_padded_dropped(padded_dropped)

padded_kept_shapes = []
for padded_kept in ds.padded_batch(4):
    padded_kept_shapes.append(tuple(padded_kept.shape))
    consume_padded_kept(padded_kept)

assert padded_kept_shapes == [(4, 3), (4, 3), (2, 3)], padded_kept_shapes

# Passed positionally the flag resolves by argument position rather than by keyword, and the two
# operations put it in different places: third for `batch`, fifth for `padded_batch`.
for positional in ds.batch(4, True):
    assert positional.shape == (4, 3), positional.shape
    consume_positional(positional)

for padded_positional in ds.padded_batch(4, None, None, True):
    assert padded_positional.shape == (4, 3), padded_positional.shape
    consume_padded_positional(padded_positional)
