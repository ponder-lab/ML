# A `map` function whose returns pass different parameters through, so the mapped element is not
# one fixed component of the receiver's element (wala/ML#1088).
import tensorflow as tf

FIRST = bool(int("1"))


def consume(element):
    pass


def consume_first(component):
    pass


def either(wide, narrow):
    if FIRST:
        return wide
    return narrow


mapped = tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.ones([8, 3]))).map(
    either
)
for element in mapped:
    assert element.shape == (4,) and element.dtype == tf.float32
    consume(element)


def pair_or_label(pair, label):
    if FIRST:
        return pair
    return label


paired = tf.data.Dataset.from_tensor_slices(
    ((tf.ones([8, 4]), tf.ones([8, 3])), tf.ones([8, 5]))
).map(pair_or_label)
for element in paired:
    assert element[0].shape == (4,) and element[0].dtype == tf.float32
    consume_first(element[0])
