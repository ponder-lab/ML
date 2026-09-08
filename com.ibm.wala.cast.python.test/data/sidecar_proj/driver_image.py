# Reduction of the wala/ML#876 image_transformation chain: an annotated array reaches an
# augmentation helper by TWO paths, directly and through a distorted random crop whose
# tf.slice bounds come from sample_distorted_bounding_box.
import numpy as np
import tensorflow as tf

img_array = np.load("image.npy")


def distorted_random_crop(image):
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.constant([[[0.0, 0.0, 1.0, 1.0]]]),
        min_object_covered=0.1,
    )
    return tf.slice(image, bbox_begin, bbox_size)


def consume(image):
    pass


def random_flip_left_right(image):
    consume(image)
    return tf.image.random_flip_left_right(image)


def transform_image(image):
    image = distorted_random_crop(image)
    image = random_flip_left_right(image)
    return image


def consume_decoded(image):
    pass


def decoded_chain():
    # A decoded JPEG has no static extents, so its axes carry run-time None-evidence and are
    # Dynamic rather than Unresolved. Slicing it must PRESERVE that distinction through the feed
    # (the wala/ML#721 convention), not flatten both sentinels together.
    decoded = tf.image.decode_jpeg(tf.constant("bytes"))
    consume_decoded(distorted_random_crop(decoded))


direct = random_flip_left_right(img_array)
chained = transform_image(img_array)
decoded_chain()


# wala/ML#901 negative case: a tensor parameter fed a value that never flows from the annotated
# `img_array`, so its evidence does not rest on the annotation. Its origins must read `{PARAMETER}`
# ALONE, with no `ANNOTATION`, distinguishing "the marker crosses where the evidence is" from "the
# marker crosses onto every parameter". A dedicated sink keeps it off the annotated path's union.
def plain_consume(t):
    pass


def unannotated_transform(plain_image):
    plain_consume(plain_image)


plain = tf.ones((2, 3))
unannotated_transform(plain)
