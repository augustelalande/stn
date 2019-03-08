import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.nn import relu

from spatial_transformer import SpatialTransformer


class SpatialTransformerNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.transform = SpatialTransformer()
        self.classify = ClassNet()

    def call(self, x):
        x_t = self.transform(x)
        o = self.classify(x_t)
        return x_t, o


class ClassNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2D(16, 3, activation=relu)
        self.c2 = Conv2D(16, 3, activation=relu)
        self.d1 = Dense(514, activation=relu)
        self.d2 = Dense(10)

    def call(self, x):
        b, w, h, c = x.shape

        x = self.c1(x)
        x = self.c2(x)

        x = tf.reshape(x, [b, -1])

        x = self.d1(x)
        x = self.d2(x)
        return x
