import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from distribution.base import distribution

# just used in sampler, so no need to implement log_prob


class mvn(distribution):
    """
    multivariate normal distribution
    """

    def __init__(self, transformation, sigma):
        self.sigmaChol = np.linalg.cholesky(sigma)
        self.transformation = transformation

    def sample(self, Input):
        mu = self.transformation.transform(Input)
        return mu + np.dot(self.sigmaChol, np.random.randn(len(mu)))

    def get_mvn(self, Input=None):
        with tf.name_scope(self.name):
            if Input is None:
                mu = self.output_0
            else:
                mu = self.transformation.transform(Input)
            sigma_tmp = tf.maximum(
                tf.nn.softplus(
                    self.sigma),
                sigma_min,
                name="sigma_tmp")
            mvn = tfd.MultivariateNormalDiag(mu, sigma_tmp,
                                             validate_args=True,
                                             allow_nan_stats=False)
            return mvn

    def sample_and_log_prob(self, Input, sample_shape=(), name=None):
        mvn = self.get_mvn(Input)
        with tf.name_scope(name or self.name):
            sample = mvn.sample(sample_shape)
            log_prob = mvn.log_prob(sample)
            return sample, log_prob

    def log_prob(self, Input, output, name=None):
        mvn = self.get_mvn(Input)
        with tf.name_scope(name or self.name):
            return mvn.log_prob(output)

    def mean(self, Input, name=None):
        mvn = self.get_mvn(Input)
        with tf.name_scope(name or self.name):
            return mvn.mean()
