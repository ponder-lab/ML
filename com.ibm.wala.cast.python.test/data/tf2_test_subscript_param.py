# Integer subscripts of a parameter: a constant index, a loop index, a component unpacked from a
# tuple parameter, and a parameter holding a Python list of tensors.
import sys

import tensorflow as tf


def consume_const(x):
    assert x.shape == (4,)
    assert x.dtype == tf.int32


def const_index(x):
    consume_const(x[0])


const_index(tf.ones((3, 4), tf.int32))


def consume_loop(x):
    assert x.shape == (4,)
    assert x.dtype == tf.int32


def loop_index(x):
    for i in range(2):
        consume_loop(x[i])


loop_index(tf.ones((3, 4), tf.int32))


def consume_labels(x):
    assert x.shape == (5,)
    assert x.dtype == tf.int32


def consume_masked(x):
    assert x.shape[0] <= 5
    assert x.dtype == tf.int32


def build_targets(gt_boxes, gt_class_ids):
    for i in range(2):
        labels = gt_class_ids[i]
        consume_labels(labels)
        consume_masked(tf.boolean_mask(labels, tf.not_equal(labels, 0)))


class M(tf.keras.Model):

    def call(self, inputs):
        imgs, gt_boxes, gt_class_ids = inputs
        build_targets(gt_boxes, gt_class_ids)
        return imgs


model = M()
model(
    (
        tf.ones((2, 8, 8, 3), tf.float32),
        tf.ones((2, 5, 4), tf.float32),
        tf.ones((2, 5), tf.int32),
    )
)


def consume_list_element(x):
    assert x.shape == (2, 3)
    assert x.dtype == tf.float32


def list_index(xs):
    consume_list_element(xs[1])


list_index([tf.ones((2,), tf.int32), tf.ones((2, 3), tf.float32)])


def consume_join(x):
    assert x.shape in ((4,), (5,))


def join_index(x):
    consume_join(x[0])


def pick(flag):
    if flag:
        t = tf.ones((3, 4), tf.int32)
    else:
        t = tf.ones((2, 5), tf.float32)
    join_index(t)


pick(len(sys.argv) > 5)


def consume_dataset_component(x):
    assert x.shape == (2,)
    assert x.dtype in (tf.int32, tf.float32)


def step(ds):
    for x, y in ds:
        consume_dataset_component(x)


step(
    tf.data.Dataset.from_tensor_slices(
        (tf.ones((3, 2), tf.int32), tf.ones((3,), tf.int32))
    )
)
step(
    tf.data.Dataset.from_tensor_slices(
        (tf.ones((3, 2), tf.float32), tf.ones((3,), tf.int32))
    )
)


def consume_element_component(x):
    assert x.shape == (2,)
    assert x.dtype in (tf.int32, tf.float32)


def take_first(element):
    consume_element_component(element[0])


take_first(
    next(
        iter(
            tf.data.Dataset.from_tensor_slices(
                (tf.ones((3, 2), tf.int32), tf.ones((3,), tf.int32))
            )
        )
    )
)
take_first(
    next(
        iter(
            tf.data.Dataset.from_tensor_slices(
                (tf.ones((3, 2), tf.float32), tf.ones((3,), tf.int32))
            )
        )
    )
)


def consume_joined_element_component(x):
    assert x.shape == (2,)
    assert x.dtype in (tf.int32, tf.float32)


def take_first_joined(element):
    consume_joined_element_component(element[0])


def pick_element(flag):
    ints = tf.data.Dataset.from_tensor_slices(
        (tf.ones((3, 2), tf.int32), tf.ones((3,), tf.int32))
    )
    floats = tf.data.Dataset.from_tensor_slices(
        (tf.ones((3, 2), tf.float32), tf.ones((3,), tf.int32))
    )
    if flag:
        element = next(iter(ints))
    else:
        element = next(iter(floats))
    take_first_joined(element)


pick_element(len(sys.argv) > 5)
