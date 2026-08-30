import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

batch_size = 128
noise_dim = 100


def consume_disc_fake(a):
    pass


def consume_disc_real(b):
    pass


def consume_gen_input(c):
    pass


def consume_gen_loss_input(d):
    pass


class Generator(Model):
    # Set layers.
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(7 * 7 * 128)
        self.bn1 = layers.BatchNormalization()
        self.conv2tr1 = layers.Conv2DTranspose(64, 5, strides=2, padding="SAME")
        self.bn2 = layers.BatchNormalization()
        self.conv2tr2 = layers.Conv2DTranspose(1, 5, strides=2, padding="SAME")

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 7, 7, 128)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = self.conv2tr1(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = self.conv2tr2(x)
        x = tf.nn.tanh(x)
        return x


# Generator Network
# Input: Noise, Output: Image
# Note that batch normalization has different behavior at training and inference time,
# we then use a placeholder to indicates the layer if we are training or not.
class Discriminator(Model):
    # Set layers.
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 5, strides=2, padding="SAME")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(128, 5, strides=2, padding="SAME")
        self.bn2 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024)
        self.bn3 = layers.BatchNormalization()
        self.fc2 = layers.Dense(2)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.conv1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn3(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        return self.fc2(x)


# Build neural network model.
generator = Generator()
discriminator = Discriminator()


# %%
# Losses.
def generator_loss(reconstructed_image):
    gen_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=reconstructed_image, labels=tf.ones([batch_size], dtype=tf.int32)
        )
    )
    return gen_loss


def discriminator_loss(disc_fake, disc_real):
    disc_loss_real = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)
        )
    )
    disc_loss_fake = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)
        )
    )
    return disc_loss_real + disc_loss_fake


# Bespoke minimal driver over the models and losses above, which are the subject's own source
# copied verbatim. The feed mirrors the subject's optimization step: a normal draw narrowed with
# `astype`, through the generator, then the discriminator, whose output reaches both loss
# parameters.
generator = Generator()
discriminator = Discriminator()

noise = np.random.normal(-1.0, 1.0, size=[batch_size, noise_dim]).astype(np.float32)
assert noise.shape == (batch_size, noise_dim), noise.shape

consume_gen_input(noise)

fake_images = generator(noise, is_training=True)
disc_fake = discriminator(fake_images, is_training=True)
disc_real = discriminator(fake_images, is_training=True)

assert disc_fake.shape == (batch_size, 2), disc_fake.shape

consume_disc_fake(disc_fake)
consume_disc_real(disc_real)

consume_gen_loss_input(disc_fake)

loss = discriminator_loss(disc_fake, disc_real)
gen_loss = generator_loss(disc_fake)
