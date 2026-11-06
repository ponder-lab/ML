# Test https://github.com/wala/ML/issues/1214: `scan` calls its function with the state and the
# element, and yields the second member of each pair the function returns, threading the first as
# the next state.
import tensorflow as tf


def step(s, v):
    assert s.shape == ()
    assert s.dtype == tf.float32
    assert v.shape == (4,)
    assert v.dtype == tf.float32
    return s + 1.0, v


def grow(s, v):
    return s + v, s


def hold(s, v):
    return s + 1.0, s


def total(s, v):
    t = s + tf.reduce_sum(v)
    return t, t


def split(s, v):
    return s + 1.0, (v, v * 2.0)


def consume_stepped(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_grown(x):
    assert x.shape in ((), (4,))
    assert x.dtype == tf.float32


def consume_held(x):
    assert x.shape == ()
    assert x.dtype == tf.float32


def consume_held_batch(x):
    assert x.shape in ((3,), (2,))
    assert x.dtype == tf.float32


def consume_total_batch(x):
    assert x.shape == (2,)
    assert x.dtype == tf.float32


def consume_held_mapped(x):
    assert x.shape == ()
    assert x.dtype == tf.float32


def consume_split_first(x):
    assert x.shape == (4,)
    assert x.dtype == tf.float32


def consume_split_batch_first(x):
    assert x.shape == (2, 4)
    assert x.dtype == tf.float32


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))

for x in flat.scan(tf.constant(0.0), step):
    consume_stepped(x)

for x in flat.scan(tf.constant(0.0), grow):
    consume_grown(x)

for x in flat.scan(tf.constant(0.0), hold):
    consume_held(x)

for x in flat.scan(tf.constant(0.0), hold).batch(3):
    consume_held_batch(x)

for x in flat.scan(tf.constant(0.0), total).batch(2):
    consume_total_batch(x)

for x in flat.scan(tf.constant(0.0), hold).map(lambda h: h * 2.0):
    consume_held_mapped(x)

# A structured second member: its members are the tuple's own.
for x, y in flat.scan(tf.constant(0.0), split):
    consume_split_first(x)

for x, y in flat.scan(tf.constant(0.0), split).batch(2):
    consume_split_batch_first(x)
