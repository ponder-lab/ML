# A tensor's `numpy()` result keeps a shape and dtype its receiver has only in the dataflow: the
# image is typed by the project's sidecar alone, its crop keeps the dtype and channel there, and the
# crop's `numpy()` must carry the same rather than only what the points-to substrate resolves.
import numpy as np
import tensorflow as tf

np.save("image.npy", np.zeros((4830, 2900, 3), dtype=np.uint8))
img_array = np.load("image.npy")


def distorted_random_crop(
    image,
    min_object_covered=0.1,
    aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
    area_range=(0.06, 1.0),
    max_attempts=100,
    scope=None,
):
    cropbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=cropbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
    )
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image


def consume_crop(image):
    pass


def consume_array(array):
    pass


cropped = distorted_random_crop(img_array)
consume_crop(cropped)
array = distorted_random_crop(img_array).numpy()
assert array.shape[2] == 3 and array.dtype == np.uint8
consume_array(array)
