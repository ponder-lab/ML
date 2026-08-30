import tensorflow as tf

# The feature-pyramid network at its subject shape, copied near-verbatim as a bespoke minimal
# driver: a ResNet backbone of loop-built Sequential stages feeding lateral and top-down paths
# whose merge helper reads its second argument's static shape.


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, in_channels, out_channels, strides=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=strides, padding="same", use_bias=False
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, kernel_size=3, strides=1, padding="same", use_bias=False
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        self.expansion * out_channels,
                        kernel_size=1,
                        strides=strides,
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        out += self.shortcut(x, training)
        return tf.nn.relu(out)


class FPN(tf.keras.Model):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_channels = 64

        self.conv1 = tf.keras.layers.Conv2D(64, 7, 2, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.top_layer = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")

        self.smooth1 = tf.keras.layers.Conv2D(256, 3, 1, padding="same")
        self.smooth2 = tf.keras.layers.Conv2D(256, 3, 1, padding="same")
        self.smooth3 = tf.keras.layers.Conv2D(256, 3, 1, padding="same")

        self.lateral_layer1 = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer2 = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")
        self.lateral_layer3 = tf.keras.layers.Conv2D(256, 1, 1, padding="valid")

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return tf.keras.Sequential(layers)

    def _upsample_add(self, x, y):
        _, H, W, C = y.shape
        return tf.image.resize(x, size=(H, W), method="bilinear")

    def call(self, x, training=False):
        p1 = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        p1 = tf.nn.max_pool2d(p1, ksize=3, strides=2, padding="SAME")

        p2 = self.layer1(p1, training=training)
        p3 = self.layer2(p2, training=training)
        p4 = self.layer3(p3, training=training)
        p5 = self.layer4(p4, training=training)

        d5 = self.top_layer(p5)
        d4 = self._upsample_add(d5, self.lateral_layer1(p4))
        d3 = self._upsample_add(d4, self.lateral_layer2(p3))
        d2 = self._upsample_add(d3, self.lateral_layer3(p2))

        d4 = self.smooth1(d4)
        d3 = self.smooth2(d3)
        d2 = self.smooth3(d2)

        return d2, d3, d4, d5


def ResNet18_fpn():
    return FPN(BasicBlock, [2, 2, 2, 2])


data = tf.ones(shape=[1, 416, 416, 3])
model = ResNet18_fpn()
fms = model(data)
for fm in fms:
    assert fm.shape[0] == 1, fm.shape
    assert fm.shape[3] == 256, fm.shape
