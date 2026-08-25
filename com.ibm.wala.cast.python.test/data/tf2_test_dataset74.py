# The reduced witness from the wala/ML#834 report, vendored: the `(x, y)` batch tuple's element
# types must reach the tuple's instance-field keys, where structure-consuming clients read
# per-position element evidence; reader-local types alone present the two positions as
# indistinguishable alternatives. Analysis-only as written: executing it needs an `images/`
# directory with class subfolders, which is not shipped.
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1.0 / 255)
generator = datagen.flow_from_directory(
    "images", target_size=(64, 64), batch_size=4, class_mode="categorical"
)


def step(inputs):
    images, labels = inputs
    return tf.reduce_mean(images) + tf.reduce_mean(labels)


dataset = iter(generator)
step(next(dataset))
