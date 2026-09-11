# Channel-isolating witness for wala/ML#905: a sidecar-typed array is cropped by a `tf.slice` whose
# bounds are the destructured outputs of `sample_distorted_bounding_box`, mirroring the subject's
# `distorted_random_crop` form exactly (the result bound to a name, then unpacked; the box built by
# `tf.constant` with a `shape` keyword; every keyword passed). The crop's spatial extents are
# genuinely unknown; its channel is taken in full (`size[2] == -1`), so the sink's parameter must
# keep the 3 the annotation supplies. The crop's own parameter stays fully annotated: the two
# assertions together separate "the channel survives the crop" from "the input was never typed".
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


cropped = distorted_random_crop(img_array)
assert isinstance(cropped, tf.Tensor)
assert cropped.shape.rank == 3
assert cropped.shape[2] == 3
assert cropped.dtype == tf.uint8
consume_crop(cropped)
