import numpy as np
import tensorflow as tf

# The image-augmentation chain in miniature: a distorted crop whose `begin`/`size` are runtime
# tensors from `tf.image.sample_distorted_bounding_box`, feeding shape-preserving `tf.image`
# helpers. `tf.slice` never changes the rank at run time, so the crop's result is rank 3 whatever
# the bounds resolve to, and every helper downstream receives a rank-3 image.


def consume_crop(x):
    pass


def consume_flip(x):
    pass


def consume_contrast(x):
    pass


def consume_union(x):
    pass


def consume_other_bounds(x):
    pass


img_array = np.ones((1332, 800, 3), dtype=np.uint8)


def distorted_random_crop(image):
    cropbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    sample = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=cropbox,
        min_object_covered=0.1,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, distort_bbox = sample
    cropped = tf.slice(image, bbox_begin, bbox_size)
    assert cropped.shape.rank == 3, cropped.shape
    assert cropped.shape[2] == 3, cropped.shape
    consume_crop(cropped)
    return cropped


def random_flip_left_right(image):
    assert image.shape.rank == 3, image.shape
    consume_flip(image)
    return tf.image.random_flip_left_right(image)


def random_contrast(image):
    assert image.shape.rank == 3, image.shape
    consume_contrast(image)
    r = tf.random.uniform([], minval=0.6, maxval=1.4)
    adjusted = tf.image.adjust_contrast(image, contrast_factor=r)
    return tf.cast(adjusted, tf.uint8)


out = random_contrast(random_flip_left_right(distorted_random_crop(img_array)))
assert out.shape.rank == 3, out.shape
assert out.dtype == tf.uint8, out.dtype


def random_brightness(image):
    assert len(image.shape) == 3, image.shape
    consume_union(image)
    return tf.image.adjust_brightness(image, 0.3)


# The two-call-site geometry: the same helper takes the full image directly and the crop's
# result, so its parameter is the union of the concrete shape and the rank-3 degraded one.
direct = random_brightness(img_array)
assert direct.shape == (1332, 800, 3), direct.shape
chained = random_brightness(distorted_random_crop(img_array))
assert chained.shape.rank == 3, chained.shape


def make_bounds():
    b = tf.random.uniform([3], maxval=2, dtype=tf.int32) * tf.constant([1, 1, 0])
    s = tf.constant([100, 100, 3]) - b
    return b, s


# A producer whose contract is NOT the documented crop's: the same destructuring shape over a
# different tuple source must keep the all-degraded answer rather than borrowing the channel
# contract.
other_begin, other_size = make_bounds()
other = tf.slice(img_array, other_begin, other_size)
assert other.shape.rank == 3, other.shape
consume_other_bounds(other)
