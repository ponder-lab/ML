# A decorator built on `functools.wraps`, in the shape of a library's own image-type decorator: the
# wrapper converts the image, calls the wrapped function, and converts the result back. The
# conversions are casts here; the library's own use `tf.image.convert_image_dtype`.
from functools import wraps

import tensorflow as tf


def convert_type(function):
    @wraps(function)
    def wrap(image, *args, **kwargs):
        image_type = image.dtype
        image = tf.cast(image, tf.float32)
        image = function(image, **kwargs)
        return tf.cast(image, image_type)

    return wrap
