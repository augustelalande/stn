import tensorflow as tf


class GridSampler(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, im, grid):
        # b, h, w, c = im.shape
        # h_f, w_f = tf.cast(h, tf.float32), tf.cast(w, tf.float32)
        #
        # grid_x = grid[:, 0]
        # grid_y = grid[:, 1]
        #
        # grid_x = (grid_x + 1) * w_f / 2
        # grid_y = (grid_y + 1) * h_f / 2
        # warp = tf.stack([grid_x, grid_y], -1)
        # return tf.contrib.resampler.resampler(im, warp)

        b, h, w, c = im.shape
        h_f, w_f = tf.cast(h, tf.float32), tf.cast(w, tf.float32)

        grid_x = grid[:, 0]
        grid_y = grid[:, 1]

        grid_x = (grid_x + 1) * w_f / 2
        grid_y = (grid_y + 1) * h_f / 2

        grid_x0 = tf.floor(grid_x)
        grid_x1 = grid_x0 + 1
        grid_y0 = tf.floor(grid_y)
        grid_y1 = grid_y0 + 1

        grid_x0 = tf.clip_by_value(grid_x0, 0, w_f - 2)
        grid_x1 = tf.clip_by_value(grid_x1, 1, w_f - 1)
        grid_y0 = tf.clip_by_value(grid_y0, 0, h_f - 2)
        grid_y1 = tf.clip_by_value(grid_y1, 1, h_f - 1)

        b_index = tf.reshape(tf.range(b), [b, 1, 1])
        b_index = tf.tile(b_index, [1, h, w])

        grid_x0_i = tf.cast(grid_x0, tf.int32)
        grid_x1_i = tf.cast(grid_x1, tf.int32)
        grid_y0_i = tf.cast(grid_y0, tf.int32)
        grid_y1_i = tf.cast(grid_y1, tf.int32)

        Q00 = tf.gather_nd(im, tf.stack([b_index, grid_y0_i, grid_x0_i], -1))
        Q01 = tf.gather_nd(im, tf.stack([b_index, grid_y1_i, grid_x0_i], -1))
        Q10 = tf.gather_nd(im, tf.stack([b_index, grid_y0_i, grid_x1_i], -1))
        Q11 = tf.gather_nd(im, tf.stack([b_index, grid_y1_i, grid_x1_i], -1))
        Q = tf.stack([Q00, Q01, Q10, Q11], -1)
        Q = tf.reshape(Q, [b, h, w, c, 2, 2])

        Wx = tf.stack([grid_x1 - grid_x, grid_x - grid_x0], -1)
        Wx = tf.reshape(Wx, [b, h, w, 1, 1, 2])
        Wx = tf.tile(Wx, [1, 1, 1, c, 1, 1])

        Wy = tf.stack([grid_y1 - grid_y, grid_y - grid_y0], -1)
        Wy = tf.reshape(Wy, [b, h, w, 1, 2, 1])
        Wy = tf.tile(Wy, [1, 1, 1, c, 1, 1])

        Wd = (grid_x1 - grid_x0) * (grid_y1 - grid_y0)
        Wd = tf.expand_dims(Wd, -1)

        im_t = Wx @ Q @ Wy
        im_t = tf.reshape(im_t, [b, h, w, c])
        im_t = im_t / Wd
        return im_t


if __name__ == '__main__':
    tf.enable_eager_execution()

    from grid_generator import GridGenerator

    im = tf.ones([10, 64, 64, 3], dtype=tf.float32)
    theta = tf.constant([[1, 0, 0, 0, 1, 0]], dtype=tf.float32)
    theta = tf.tile(theta, [10, 1])

    gg = GridGenerator()
    grid = gg(im, theta)

    gs = GridSampler()
    im_t = gs(im, grid)

    print(im_t.shape)
