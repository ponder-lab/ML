# From https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/299fd6689f242d0f647a96b8844e86325e9fcb46/7-Utils/multi_gpu_train.py.

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


@tf.function
def distributed_train_step(dataset_inputs):
    pass


EPOCHS = 40
IMG_SIZE = 112  # Input Image Size
BATCH_SIZE = 512  # Total 4 GPU, 128 batch per GPU

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False
)

train_generator = train_datagen.flow_from_directory(
    "./mnist/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

for epoch in range(1, EPOCHS + 1):
    batchs_per_epoch = len(train_generator)
    train_dataset = iter(train_generator)

    for _ in range(batchs_per_epoch):
        dataset_inputs = next(train_dataset)

        assert isinstance(dataset_inputs, tuple)
        assert len(dataset_inputs) == 2

        images, labels = dataset_inputs

        assert isinstance(images, tf.Tensor)
        assert isinstance(labels, tf.Tensor)

        # Check shapes
        assert (
            images.shape.as_list() == [None, 112, 112, 3]
            or images.shape.as_list() == [1, 112, 112, 3]
            or images.shape.as_list() == [BATCH_SIZE, 112, 112, 3]
        )
        assert (
            labels.shape.as_list() == [None, 1]
            or labels.shape.as_list() == [1, 1]
            or labels.shape.as_list() == [BATCH_SIZE, 1]
        )

        # Check dtypes
        assert images.dtype == tf.float32
        assert labels.dtype == tf.float32

        batch_loss = distributed_train_step(dataset_inputs)
