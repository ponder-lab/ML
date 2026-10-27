# An unsorted segment reduction whose `num_segments` is known only at run time is given no rank
# (wala/ML#1160).
import tensorflow as tf


def consume_direct(x):
    pass


def consume_chosen(x):
    pass


def consume_counted(x):
    pass


def consume_dynamic(x):
    pass


def consume_starred(x):
    pass


MESSAGES = tf.ones([4, 6])
SEGMENT_IDS = tf.constant([1, 2, 3, 4])


def aggregate_starred(first, second, num_nodes, extra):
    return tf.math.unsorted_segment_sum(MESSAGES, SEGMENT_IDS, num_nodes)


def aggregate(per_type, targets, num_nodes, aggr):
    if aggr == "sum":
        fn = tf.math.unsorted_segment_sum
    elif aggr == "max":
        fn = tf.math.unsorted_segment_max
    else:
        fn = tf.math.unsorted_segment_mean
    messages = tf.concat(per_type, axis=0)
    direct = tf.math.unsorted_segment_sum(
        messages, tf.concat(targets, axis=0), num_nodes
    )
    assert direct.shape == (5, 6) and direct.dtype == tf.float32
    consume_direct(direct)
    return fn(
        data=messages, segment_ids=tf.concat(targets, axis=0), num_segments=num_nodes
    )


x = tf.linalg.matmul(tf.ones([5, 3]), tf.ones([3, 6]))
edges = tf.constant([[0, 1], [1, 2], [2, 3], [3, 4]])
per_type = []
for adjacency in [edges]:
    per_type.append(tf.reshape(tf.ones([4]), [-1, 1]) * tf.gather(x, adjacency[:, 0]))
out = aggregate(per_type, [edges[:, 1]], tf.shape(x)[0], "sum")
assert out.shape == (5, 6)
chosen = tf.math.softmax(out, -1)
assert chosen.dtype == tf.float32
consume_chosen(chosen)

# A `num_segments` that is a Python `int` of unknown value: a fixed size.
xs = [1, 2, 3, 4, 5]
counted = tf.math.unsorted_segment_sum(tf.ones([4, 6]), edges[:, 1], len(xs))
assert counted.shape == (5, 6)
consume_counted(counted)

# A `num_segments` read off a tensor whose leading axis is dynamic: TensorFlow's static shape reports
# `None` there.
inputs = tf.keras.Input(shape=(6,))
dynamic = tf.math.unsorted_segment_sum(
    tf.ones([4, 6]), edges[:, 1], tf.shape(inputs)[0]
)
assert dynamic.shape.as_list() == [None, 6]
consume_dynamic(dynamic)

# A `num_segments` passed after a starred argument: `num_nodes` binds `tf.shape(x)[0]`, 5, while the
# call's fourth positional slot holds `tf.shape(y)[0]`, 7.
y = tf.ones([7, 2])
leading = (0, 0)
starred = aggregate_starred(*leading, tf.shape(x)[0], tf.shape(y)[0])
assert starred.shape == (5, 6)
consume_starred(starred)
