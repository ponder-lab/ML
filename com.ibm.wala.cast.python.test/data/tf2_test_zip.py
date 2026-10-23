# Tensors carried through `zip` inside functions keep their types: unpacked in a `for` loop,
# subscripted off `list(zip(...))`, and transposed by `zip(*rows)` before a concatenation. The
# transposed rows hold one tensor type, since the summary of `zip` does not track the iteration
# index a tuple came from. A `list` copy of a tensor holds the tensor's rows.
import tensorflow as tf


def consume_loop_first(t):
    pass


def consume_loop_second(t):
    pass


def consume_index_first(t):
    pass


def consume_index_second(t):
    pass


def consume_star_first(t):
    pass


def consume_star_second(t):
    pass


def consume_list_element(t):
    pass


def loop_zip():
    a_list = [tf.ones((1, 60, 2)), tf.ones((1, 60, 2))]
    b_list = [tf.zeros((1, 60, 2), tf.int32), tf.zeros((1, 60, 2), tf.int32)]
    for p, q in zip(a_list, b_list):
        consume_loop_first(p)
        assert p.shape == (1, 60, 2)
        assert p.dtype == tf.float32
        consume_loop_second(q)
        assert q.shape == (1, 60, 2)
        assert q.dtype == tf.int32


def index_zip():
    a_list = [tf.ones((1, 60, 2)), tf.ones((1, 60, 2))]
    b_list = [tf.zeros((1, 60, 2), tf.int32), tf.zeros((1, 60, 2), tf.int32)]
    pairs = list(zip(a_list, b_list))
    first = pairs[0]
    consume_index_first(first[0])
    assert first[0].shape == (1, 60, 2)
    assert first[0].dtype == tf.float32
    consume_index_second(first[1])
    assert first[1].shape == (1, 60, 2)
    assert first[1].dtype == tf.int32


def layer(x):
    return tf.ones((1, 60, 2)), tf.zeros((1, 60, 2))


def star_zip():
    layer_outputs = [layer(i) for i in range(2)]
    outputs = list(zip(*layer_outputs))
    first, second = [tf.concat(list(o), axis=1) for o in outputs]
    consume_star_first(first)
    assert first.shape == (1, 120, 2)
    assert first.dtype == tf.float32
    consume_star_second(second)
    assert second.shape == (1, 120, 2)
    assert second.dtype == tf.float32


def iterate_list(y):
    for e in list(y):
        consume_list_element(e)
        assert e.shape == (2,)
        assert e.dtype == tf.float32


loop_zip()
index_zip()
star_zip()
iterate_list(tf.ones((3, 2)))
