# Test https://github.com/wala/ML/issues/967: an input passed by keyword to a pass-through-family
# op is fed from its dataflow type exactly as a positional one is.
import tensorflow as tf


def consume_seg_kw(t):
    assert t.shape == (3, 3)
    assert t.dtype == tf.float32


def consume_seg_pos(t):
    assert t.shape == (3, 3)
    assert t.dtype == tf.float32


def consume_cos_kw(t):
    assert t.shape == (4, 3)
    assert t.dtype == tf.float32


def consume_cos_pos(t):
    assert t.shape == (4, 3)
    assert t.dtype == tf.float32


# A concat over a list built in a loop is typed by dataflow alone: no points-to read reaches its
# elements, so an op reading it only gets a dtype through its feed.
parts = []
for k in range(2):
    parts.append(tf.ones((2, 3), dtype=tf.float32) * 2.0)
cat = tf.concat(parts, axis=0)
assert cat.shape == (4, 3) and cat.dtype == tf.float32

ids = tf.constant([0, 1, 1, 2], dtype=tf.int32)
consume_seg_kw(tf.math.unsorted_segment_sum(data=cat, segment_ids=ids, num_segments=3))
consume_seg_pos(tf.math.unsorted_segment_sum(cat, ids, 3))
consume_cos_kw(tf.math.cos(x=cat))
consume_cos_pos(tf.math.cos(cat))
