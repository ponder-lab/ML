# `tf.while_loop` calls its `body` with the loop variables unpacked positionally, so a function
# reached only through a loop body gets a call-graph node and typed parameters (wala/ML#942).
import tensorflow as tf


def consume_image(x):
    assert x.dtype == tf.float32
    assert x.shape == (4, 4, 3)


def consume_index(x):
    assert x.dtype == tf.int32
    assert x.shape == ()


def consume_boxes(x):
    assert x.dtype == tf.float32
    assert x.shape == (1, 4)


def erase(image):
    consume_image(image)
    return image * 0.5


def multiple_erase(image, iterations=3):
    i = tf.constant(0)
    condition = lambda i, _image: i < iterations
    body = lambda i, image: (i + 1, erase(image))
    _, image = tf.while_loop(condition, body, (i, image))
    return image


def step(i, image, boxes):
    consume_index(i)
    consume_boxes(boxes)
    return i + 1, image, boxes


multiple_erase(tf.ones((4, 4, 3)))
loop_body = step
tf.while_loop(
    lambda i, image, boxes: i < 2,
    loop_body,
    (tf.constant(0), tf.ones((4, 4, 3)), tf.zeros((1, 4))),
)
