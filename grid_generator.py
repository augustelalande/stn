import tensorflow as tf


class GridGenerator(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, im, theta):
        b, h, w, c = im.shape

        theta = tf.reshape(theta, (-1, 2, 3))

        grid = self._meshgrid(h, w)
        grid = tf.expand_dims(grid, 0)
        grid = tf.tile(grid, [b, 1, 1])

        T = theta @ grid
        return tf.reshape(T, [-1, 2, h, w])

    def _meshgrid(self, height, width):
        x_t, y_t = tf.meshgrid(
            tf.linspace(-1.0, 1.0, width),
            tf.linspace(-1.0, 1.0, height)
        )
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        ones = tf.ones_like(x_t_flat)

        grid = tf.stack([x_t_flat, y_t_flat, ones])
        return grid


if __name__ == '__main__':
    tf.enable_eager_execution()

    im = tf.ones([10, 64, 64, 3], dtype=tf.float32)
    theta = tf.constant([[1, 0, 0, 0, 1, 0]], dtype=tf.float32)
    theta = tf.tile(theta, [10, 1])

    gg = GridGenerator()

    grid = gg(im, theta)
    print(grid.shape)
