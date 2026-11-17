# A dataset reassigned in a loop to what `Dataset.apply` returns for it (wala/ML#1290,
# wala/ML#1197).

import tensorflow as tf


def first(dataset):
    assert dataset.element_spec.shape == (23,)
    return dataset.batch(2, drop_remainder=True)


def second(dataset):
    assert dataset.element_spec.shape == (2, 23)
    return dataset.batch(3, drop_remainder=True)


def consume_looped(dataset):
    assert dataset.element_spec.shape == (3, 2, 23)


looped = tf.data.Dataset.from_tensor_slices(tf.ones([12, 23]))
for function in (first, second):
    looped = looped.apply(function)
consume_looped(looped)
