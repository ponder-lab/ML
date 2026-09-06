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


direct = random_flip_left_right(img_array)
chained = transform_image(img_array)
