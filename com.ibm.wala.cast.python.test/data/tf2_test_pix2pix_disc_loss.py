import numpy as np
import tensorflow as tf

# The paired-image GAN's discriminator loss at its subject shape: a functional Model over two
# declared Input(shape=[None, None, 3]) heads, concatenated and downsampled to a patch map, called
# with a LIST of two images. The declared inputs make the output's static shape
# (None, None, None, 1): batch and the convolved spatial axes dynamic, the final channel concrete.


def consume_real(x):
    pass


def consume_single(x):
    pass


def consume_pair(x):
    pass


def consume_generated(x):
    pass


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


def Discriminator():
    initializer = tf.random_normal_initializer(0.0, 0.02)
    inp = tf.keras.layers.Input(shape=[None, None, 3], name="input_image")
    tar = tf.keras.layers.Input(shape=[None, None, 3], name="target_image")

    x = tf.keras.layers.concatenate([inp, tar])

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    consume_real(disc_real_output)
    consume_generated(disc_generated_output)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    return real_loss + generated_loss


input_image = tf.constant(np.ones((1, 256, 256, 3), dtype=np.float32))
target = tf.constant(np.ones((1, 256, 256, 3), dtype=np.float32))
gen_output = tf.constant(np.ones((1, 256, 256, 3), dtype=np.float32))

disc_real_output = discriminator([input_image, target], training=True)
disc_generated_output = discriminator([input_image, gen_output], training=True)
assert disc_real_output.shape == (1, 30, 30, 1), disc_real_output.shape
assert disc_real_output.dtype == tf.float32, disc_real_output.dtype

disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
assert disc_loss.shape == (), disc_loss.shape

# The control that splits the cause: the same functional construction with a SINGLE declared
# input. If this composes while the list-input call does not, the loss is in the list form; if
# neither composes, it is the functional call result generally.
single_inp = tf.keras.layers.Input(shape=[None, None, 3], name="single_image")
single_out = tf.keras.layers.Conv2D(1, 4, strides=1)(single_inp)
single_model = tf.keras.Model(inputs=single_inp, outputs=single_out)
single_result = single_model(input_image)
assert single_result.shape == (1, 253, 253, 1), single_result.shape
consume_single(single_result)

# The second control: a LIST-input functional model with NO Sequential in its graph. If this
# composes, the list form is exonerated and the loss belongs to the `.add()`-built Sequential hop.
pair_a = tf.keras.layers.Input(shape=[None, None, 3], name="pair_a")
pair_b = tf.keras.layers.Input(shape=[None, None, 3], name="pair_b")
pair_cat = tf.keras.layers.concatenate([pair_a, pair_b])
pair_out = tf.keras.layers.Conv2D(1, 4, strides=1)(pair_cat)
pair_model = tf.keras.Model(inputs=[pair_a, pair_b], outputs=pair_out)
pair_result = pair_model([input_image, target])
assert pair_result.shape == (1, 253, 253, 1), pair_result.shape
consume_pair(pair_result)
