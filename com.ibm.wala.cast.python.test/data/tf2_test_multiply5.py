# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/math/multiply#for_example/

import tensorflow as tf
from builtins import float


def f(a):
    pass


# Constructing a list with shape (2, 2, 2, 3)
# Interpretation: 2 images, 2 pixels high, 2 pixels wide, 3 color channels (RGB)


# Just for clarity: A single pixel with RGB values
pixel = [1.0, 1.0, 1.0]
assert len(pixel) == 3  # Confirming shape (3,)
assert pixel[0].__class__ == float  # Confirming dtype float32

images_list = [
    # Image 1
    [[pixel, pixel], [pixel, pixel]],  # Row 1  # Row 2
    # Image 2
    [[pixel, pixel], [pixel, pixel]],  # Row 1  # Row 2
]
assert len(images_list) == 2  # Confirming shape (2, 2, 2, 3)
assert len(images_list[0]) == 2  # Confirming shape (2, 2, 3)
assert len(images_list[0][0]) == 2  # Confirming shape (2, 3)
assert len(images_list[0][0][0]) == 3  # Confirming shape (3,)
assert images_list[0][0][0][0].__class__ == float  # Confirming dtype float32

# Weights: (3,) - One weight for R, one for G, one for B
weights = [0.5, 2.0, 10.0]
assert len(weights) == 3  # Confirming shape (3,)
assert weights[0].__class__ == float  # Confirming dtype float32

# TensorFlow automatically converts the lists to Tensors
result = tf.multiply(images_list, weights)
assert result.shape == (2, 2, 2, 3)
assert result.dtype == tf.float32

f(result)
