import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.nn import relu


class LocNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.c1 = Conv2D(16, 3, activation=relu)
        self.c2 = Conv2D(16, 3, activation=relu)
        self.d1 = Dense(514, activation=relu)

        init = tf.constant_initializer([5, 0, 0, 0, 5, 0])
        self.d2 = Dense(6, activation=tf.tanh, bias_initializer=init)

    def call(self, x):
        b, w, h, c = x.shape

        x = self.c1(x)
        x = self.c2(x)

        x = tf.reshape(x, [b, -1])

        x = self.d1(x)
        x = self.d2(x)

        return x
