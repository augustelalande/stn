import os

import tensorflow as tf
import tensorflow.contrib.summary as summary
from tensorboard.plugins.beholder import Beholder

from stn import SpatialTransformerNetwork
from data_reader import DataReader
from utils import *


S_max = int(1e5)
batch_size = 100
lr = 1e-5

logs_path = "/localdata/auguste/logs_stn"


if __name__ == '__main__':
    session_name = get_session_name()
    session_logs_path = os.path.join(logs_path, session_name)

    global_step = tf.train.get_or_create_global_step()

    data_reader = DataReader(batch_size)
    model = SpatialTransformerNetwork()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    beholder = Beholder(logs_path)
    writer = summary.create_file_writer(session_logs_path, max_queue=1)
    writer.set_as_default()

    with summary.record_summaries_every_n_global_steps(50):

        # Train

        x, y = data_reader.read()
        theta, x_t, o = model(x)

        loss = tf.losses.softmax_cross_entropy(y, o)
        optimize = optimizer.minimize(loss, global_step=global_step)

        acc, acc_op = tf.metrics.accuracy(tf.argmax(y, -1), tf.argmax(o, -1))

        summary.scalar("loss", loss, family="train")
        summary.scalar("accuracy", acc_op, family="train")

        summary.image("image input", cast_im(x), max_images=3)
        summary.image("image transformed", cast_im(x_t), max_images=3)

        # Valid

        x, y = data_reader.read("valid")
        theta, x_t, o = model(x)

        loss = tf.losses.softmax_cross_entropy(y, o)
        acc, acc_op = tf.metrics.accuracy(tf.argmax(y, -1), tf.argmax(o, -1))

        summary.scalar("loss", loss, family="valid")
        summary.scalar("accuracy", acc_op, family="valid")

        summary.image("image input valid", cast_im(x), max_images=3)
        summary.image("image transformed valid", cast_im(x_t), max_images=3)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        summary.initialize(graph=tf.get_default_graph())

        for s in range(S_max):
            l, acc, *_ = sess.run(
                [loss, acc_op, optimize, summary.all_summary_ops()])
            beholder.update(session=sess)

            if s % 50 == 0:
                print("Iteration: {}  Loss: {} Accuracy: {}".format(s, l, acc))
