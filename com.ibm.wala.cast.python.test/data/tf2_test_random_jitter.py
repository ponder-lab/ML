import tensorflow as tf
import numpy as np

# Mirrors `random_jitter` (and the `resize`/`random_crop` helpers it calls) from
# `TensorFlow2.0-Examples/.../Pix2Pix.py`, a real-world image-to-image
# translation utility (random resize/crop/mirror data augmentation), for
# tensor-type inference coverage.
IMG_WIDTH = 256
IMG_HEIGHT = 256


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]


def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)
    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    # random mirroring
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


input_image = tf.constant(np.ones((256, 256, 3), dtype=np.float32))
real_image = tf.constant(np.ones((256, 256, 3), dtype=np.float32))
jittered_input, jittered_real = random_jitter(input_image, real_image)
assert input_image.shape == (256, 256, 3) and input_image.dtype == tf.float32
assert real_image.shape == (256, 256, 3) and real_image.dtype == tf.float32
assert jittered_input.shape == (256, 256, 3) and jittered_input.dtype == tf.float32
