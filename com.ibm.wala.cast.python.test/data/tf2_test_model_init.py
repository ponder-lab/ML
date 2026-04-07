import tensorflow as tf


def check_positional(x):
    pass


def test_positional():
    input1 = tf.keras.Input(shape=(2,), dtype=tf.float32)
    assert input1.shape == (None, 2)
    assert input1.dtype == tf.float32

    output1 = tf.keras.layers.Dense(2)(input1)
    assert output1.shape == (None, 2)
    assert output1.dtype == tf.float32

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


def check_multiple(x, y):
    pass


def test_multiple():
    input3 = tf.keras.Input(shape=(2,), dtype=tf.float32)
    output3 = tf.keras.layers.Dense(2)(input3)
    model3 = tf.keras.Model(input3, outputs=output3)

    out = model3(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 2]
    assert out.dtype == tf.float32

    input4 = tf.keras.Input(shape=(2,), dtype=tf.int32)
    output4 = tf.keras.layers.Dense(2)(input4)
    model4 = tf.keras.Model(input4, outputs=output4)

    out2 = model4(tf.ones((2, 2)))
    assert out2.shape.as_list() == [2, 2]
    assert out2.dtype == tf.float32

    check_multiple(out, out2)


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
    test_multiple()
    test_subclass()
