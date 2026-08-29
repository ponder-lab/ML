import os
import tempfile

import tensorflow as tf

# The paired-image translation input pipeline in miniature, mirroring its subject shape: a JPEG
# decoded and split into halves at a computed width, jittered through resize/random_crop/flip, and
# mapped over `list_files`. The decoded image's static shape is (None, None, None) by TensorFlow's
# own contract (`decode_jpeg` reports rank 3 with every extent unknown until runtime), so the
# statically right answer for every image parameter below is rank 3, extents dynamic; the concrete
# sizes live in the data files, not the program.

IMG_WIDTH = 256
IMG_HEIGHT = 256


def consume_rgb(x):
    pass


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    assert image.shape.rank == 3, image.shape

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    assert input_image.shape.rank == 3, input_image.shape

    return input_image, real_image


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


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image


with tempfile.TemporaryDirectory() as tmp:
    jpg = os.path.join(tmp, "pair.jpg")
    tf.io.write_file(jpg, tf.io.encode_jpeg(tf.zeros([256, 512, 3], dtype=tf.uint8)))

    train_dataset = tf.data.Dataset.list_files(os.path.join(tmp, "*.jpg"))
    train_dataset = train_dataset.map(load_image_train)
    train_dataset = train_dataset.batch(1)

    seen = 0
    for input_image, real_image in train_dataset:
        assert input_image.shape == (1, 256, 256, 3), input_image.shape
        assert input_image.dtype == tf.float32, input_image.dtype
        seen += 1
    assert seen == 1

    # The channels refinement: a literal nonzero `channels` pins the channel axis, exactly as
    # TensorFlow's own static shape then reports it ((None, None, 3)).
    rgb = tf.image.decode_jpeg(tf.io.read_file(jpg), channels=3)
    assert rgb.shape.rank == 3, rgb.shape
    assert rgb.shape[2] == 3, rgb.shape
    assert rgb.dtype == tf.uint8, rgb.dtype
    consume_rgb(rgb)
