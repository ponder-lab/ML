import tensorflow as tf

# wala/ML#1099: `reversed` iterates its argument, so the loop variable over `reversed(y)` holds the
# elements of `y`. Each loop feeds its own consumer, which uses the element in its own body.


def scale_reversed(e):
    assert e.shape == () and e.dtype == tf.float32
    return e * 2.0


def scale_reversed_list(t):
    assert t.shape == (3,) and t.dtype == tf.float32
    return t * 2.0


def reversed_param(y):
    total = 0.0
    for e in reversed(y):
        total += scale_reversed(e)
    return total


def reversed_list(a, b):
    total = 0.0
    for t in reversed([a, b]):
        total += scale_reversed_list(t)
    return total


assert float(reversed_param(tf.constant([1.0, 2.0, 3.0]))) == 12.0
assert float(tf.reduce_sum(reversed_list(tf.ones([3]), tf.ones([3])))) == 12.0
