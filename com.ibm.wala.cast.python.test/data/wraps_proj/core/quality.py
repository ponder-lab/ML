import tensorflow as tf

from core.convert_type_decorator import convert_type


def consume(x):
    pass


@convert_type
def gaussian_noise(image, stddev=0.1):
    consume(image)
    return image + tf.random.normal(tf.shape(image), stddev=stddev)
