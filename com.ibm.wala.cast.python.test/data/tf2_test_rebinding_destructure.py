import tensorflow as tf


def consume_plain_first(t):
    pass


def consume_plain_second(t):
    pass


def consume_rebound_first(t):
    pass


def consume_rebound_second(t):
    pass


# A destructuring assignment whose left-hand side does NOT mention the name being
# destructured. Both fields keep their types.
def plain(pair):
    first, second = pair
    assert first.shape == (2, 3) and first.dtype == tf.float32
    consume_plain_first(first)
    assert second.shape == (4, 5) and second.dtype == tf.float32
    consume_plain_second(second)


# The same destructuring, except that field 0's target rebinds the very name on the right.
# This is `gpt-2-tensorflow2.0`'s `train_dataset, test_dataset = train_dataset`.
def rebinding(pair):
    pair, second = pair
    assert pair.shape == (2, 3) and pair.dtype == tf.float32
    consume_rebound_first(pair)
    assert second.shape == (4, 5) and second.dtype == tf.float32
    consume_rebound_second(second)


a = tf.ones((2, 3), dtype=tf.float32)
b = tf.ones((4, 5), dtype=tf.float32)

plain((a, b))
rebinding((a, b))
