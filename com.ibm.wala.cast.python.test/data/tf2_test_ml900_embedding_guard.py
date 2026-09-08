# wala/ML#900 canary: the embedding guard-phi phantom in miniature.
#
# `embed`'s `input_ids` is a phi over a conditional `expand_dims`. When the input is rank 2 the
# guard fires and the phi's runtime value is rank 3, so the reshape that reads
# `get_shape_list(input_ids)` produces a rank-3 result. Without wala/ML#900 the `get_shape_list`
# parameter's raw points-to union retains the pre-`expand_dims` (rank-2) allocation, minting a
# spurious rank-2 member on the reshape output. With the fix the parameter resolves through the
# caller argument, whose phi feasibility prunes that arm, and only the rank-3 member remains.
import tensorflow as tf


def get_shape_list(tensor):
    return tensor.shape.as_list()


def embed(input_ids, table):
    if input_ids.shape.ndims == 2:
        input_ids = tf.expand_dims(input_ids, axis=[-1])
    flat = tf.reshape(input_ids, [-1])
    output = tf.gather(table, flat)
    input_shape = get_shape_list(input_ids)
    output = tf.reshape(output, input_shape[0:-1] + [input_shape[-1] * 8])
    return output


def consume(x):
    assert len(x.shape) == 3
    return x


def f():
    ids = tf.ones((16, 100), dtype=tf.int32)
    table = tf.ones((1000, 8))
    r = embed(ids, table)
    consume(r)


f()
