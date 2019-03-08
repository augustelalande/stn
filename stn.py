import tensorflow as tf

from localisation_net import LocNet
from grid_generator import GridGenerator
from grid_sampler import GridSampler
from classification_net import ClassNet


class SpatialTransformerNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.loc = LocNet()
        self.grid_gen = GridGenerator()
        self.grid_sample = GridSampler()
        self.classify = ClassNet()

    def call(self, batch):
        theta = self.loc(batch)
        grid = self.grid_gen(batch, theta)
        batch_t = self.grid_sample(batch, grid)
        o = self.classify(batch_t)
        return theta, batch_t, o
