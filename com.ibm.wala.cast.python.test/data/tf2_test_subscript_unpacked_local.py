# Integer subscripts of a component unpacked, in the same function, from a tuple parameter: a
# constant index, a loop index, and a Keras model's `call`; and the reads that must stay as they
# are: a list component, and slice and tuple indexes.
import tensorflow as tf


def consume_const(x):
    assert x.shape == (4,)
    assert x.dtype == tf.int32


def const_index(inputs):
    a, gt = inputs
    consume_const(gt[0])


const_index((tf.ones((2, 3), tf.float32), tf.ones((3, 4), tf.int32)))


def consume_loop(x):
    assert x.shape == (4,)
    assert x.dtype == tf.int32


def loop_index(inputs):
    a, gt = inputs
    for i in range(2):
        consume_loop(gt[i])


loop_index((tf.ones((2, 3), tf.float32), tf.ones((3, 4), tf.int32)))


def consume_model(x):
    assert x.shape == (5,)
    assert x.dtype == tf.int32


def consume_model_masked(x):
    assert x.shape[0] <= 5
    assert x.dtype == tf.int32


class M(tf.keras.Model):

    def call(self, inputs):
        imgs, gt = inputs
        for i in range(2):
            consume_model(gt[i])
            consume_model_masked(tf.boolean_mask(gt[i], tf.not_equal(gt[i], 0)))
        return imgs


model = M()
model((tf.ones((2, 8, 8, 3), tf.float32), tf.ones((2, 5), tf.int32)))


def consume_list_component(x):
    assert x.shape == (2, 3)
    assert x.dtype == tf.float32


def list_component(inputs):
    a, xs = inputs
    consume_list_component(xs[1])


list_component(
    (
        tf.ones((2, 3), tf.float32),
        [tf.ones((2,), tf.int32), tf.ones((2, 3), tf.float32)],
    )
)


def consume_slice(x):
    assert x.shape == (2, 4)
    assert x.dtype == tf.int32


def slice_index(inputs, s):
    a, gt = inputs
    consume_slice(gt[s])


slice_index((tf.ones((2, 3), tf.float32), tf.ones((3, 4), tf.int32)), slice(0, 2))


def consume_tuple_index(x):
    assert x.shape == (4,)
    assert x.dtype == tf.int32


def tuple_index(inputs, s):
    a, gt = inputs
    consume_tuple_index(gt[s])


tuple_index((tf.ones((2, 3), tf.float32), tf.ones((3, 4), tf.int32)), (0, slice(None)))
