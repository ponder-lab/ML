import tensorflow as tf


def consume_const_col(c):
    pass


def consume_batch_col(c):
    pass


# A trailing integer index with no following slice: `x[:, 0]` over a rank-2
# receiver drops the last axis entirely, leaving rank 1.
x = tf.ones((4, 5), dtype=tf.float32)
assert x.shape == (4, 5) and x.dtype == tf.float32

const_col = x[:, 0]
assert const_col.shape == (4,) and const_col.dtype == tf.float32
consume_const_col(const_col)

# The same pattern against a batched dataset element, which is how the column
# arrives in practice: the batch axis is unknown, the indexed axis is gone.
rows = tf.data.Dataset.from_tensor_slices(tf.ones((6, 5), dtype=tf.float32))
batched = rows.batch(2)

for batch in batched:
    assert batch.shape == (2, 5) and batch.dtype == tf.float32
    batch_col = batch[:, 0]
    assert batch_col.shape == (2,) and batch_col.dtype == tf.float32
    consume_batch_col(batch_col)
