import tensorflow as tf
from tensorflow.keras import Model, layers

# The GAN shape in miniature: a generator built from `Dense`, `BatchNormalization`,
# `Conv2DTranspose`, and reshapes; a discriminator from `Conv2D`, `BatchNormalization`,
# `Flatten`, and `Dense`; and loss functions whose parameters receive the discriminator's
# outputs for the fake and real arms.


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = layers.Dense(7 * 7 * 128)
        self.bn1 = layers.BatchNormalization()
        self.conv2tr1 = layers.Conv2DTranspose(64, 5, strides=2, padding="SAME")
        self.bn2 = layers.BatchNormalization()
        self.conv2tr2 = layers.Conv2DTranspose(1, 5, strides=2, padding="SAME")

    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.bn1(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = tf.reshape(x, shape=[-1, 7, 7, 128])
        x = self.conv2tr1(x)
        x = self.bn2(x, training=is_training)
        x = tf.nn.leaky_relu(x)
        x = self.conv2tr2(x)
        x = tf.nn.tanh(x)
        return x


class Discriminator(Model):
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


def generator_loss(reconstructed_image):
    return tf.reduce_mean(tf.cast(reconstructed_image, tf.float32))


def discriminator_loss(disc_fake, disc_real):
    return tf.reduce_mean(tf.cast(disc_fake, tf.float32)) + tf.reduce_mean(
        tf.cast(disc_real, tf.float32)
    )


generator = Generator()
discriminator = Discriminator()

noise = tf.ones((8, 100))
real_images = tf.ones((8, 28, 28))

fake_images = generator(noise, is_training=True)
assert fake_images.shape == (8, 28, 28, 1), fake_images.shape

disc_fake = discriminator(fake_images, is_training=True)
disc_real = discriminator(real_images, is_training=True)
assert disc_fake.shape == (8, 2), disc_fake.shape
assert disc_real.shape == (8, 2), disc_real.shape

gen_loss = generator_loss(disc_fake)
disc_loss = discriminator_loss(disc_fake, disc_real)
assert gen_loss.shape == (), gen_loss.shape
