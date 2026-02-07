import tensorflow as tf


def check_positional(x):
    pass


def test_positional():
    input1 = tf.keras.Input(shape=(2,), dtype=tf.float32)
    output1 = tf.keras.layers.Dense(2)(input1)
    model1 = tf.keras.Model(input1, output1)

    out = model1(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 2]
    assert out.dtype == tf.float32
    check_positional(out)


def check_keyword(x):
    pass


def test_keyword():
    input2 = tf.keras.Input(shape=(2,), dtype=tf.float32)
    output2 = tf.keras.layers.Dense(2)(input2)
    model2 = tf.keras.Model(outputs=output2, inputs=input2)

    out = model2(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 2]
    assert out.dtype == tf.float32
    check_keyword(out)


def check_mixed(x):
    pass


def test_mixed():
    input3 = tf.keras.Input(shape=(2,), dtype=tf.float32)
    output3 = tf.keras.layers.Dense(2)(input3)
    model3 = tf.keras.Model(input3, outputs=output3)

    out = model3(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 2]
    assert out.dtype == tf.float32
    check_mixed(out)


class SubclassModel(tf.keras.Model):
    def __init__(self):
        super(SubclassModel, self).__init__()
        self.dense = tf.keras.layers.Dense(2)

    def call(self, inputs):
        return self.dense(inputs)


def check_subclass(x):
    pass


def test_subclass():
    model = SubclassModel()
    out = model(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 2]
    assert out.dtype == tf.float32
    check_subclass(out)


if __name__ == "__main__":
    test_positional()
    test_keyword()
    test_mixed()
    test_subclass()
