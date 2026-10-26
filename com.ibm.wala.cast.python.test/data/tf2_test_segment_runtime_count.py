# An unsorted segment reduction whose `num_segments` is known only at run time is given no rank
# (wala/ML#1160).
import tensorflow as tf


def consume_direct(x):
    pass


def consume_chosen(x):
    pass


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
