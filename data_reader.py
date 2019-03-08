import tensorflow as tf
import numpy as np


class DataReader(object):
    def __init__(self, batch_size):
        mnist_cluttered = np.load(
            './data/mnist_sequence1_sample_5distortions5x5.npz')

        X_train = mnist_cluttered['X_train']
        y_train = mnist_cluttered['y_train']
        X_valid = mnist_cluttered['X_valid']
        y_valid = mnist_cluttered['y_valid']
        X_test = mnist_cluttered['X_test']
        y_test = mnist_cluttered['y_test']

        X_train = tf.reshape(X_train, [-1, 40, 40, 1])
        X_valid = tf.reshape(X_valid, [-1, 40, 40, 1])
        X_test = tf.reshape(X_test, [-1, 40, 40, 1])

        y_train = tf.squeeze(tf.one_hot(y_train, 10))
        y_valid = tf.squeeze(tf.one_hot(y_valid, 10))
        y_test = tf.squeeze(tf.one_hot(y_test, 10))

        self.data = {
            "train": [
                self._make_iterator(X_train, batch_size),
                self._make_iterator(y_train, batch_size)
            ],

            "valid": [
                self._make_iterator(X_valid, batch_size),
                self._make_iterator(y_valid, batch_size)
            ],

            "test": [
                self._make_iterator(X_test, batch_size),
                self._make_iterator(y_test, batch_size)
            ]
        }

    def _make_iterator(self, data, batch_size, buffer_size=10000):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, True)
        dataset = dataset.repeat()
        return dataset.make_one_shot_iterator()

    def read(self, mode="train"):
        x = self.data[mode][0].get_next()
        y = self.data[mode][1].get_next()
        return x, y


if __name__ == '__main__':
    tf.enable_eager_execution()

    data_reader = DataReader(32)
    x, y = data_reader.read()
    print(x, y)
