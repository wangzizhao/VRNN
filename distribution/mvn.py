import numpy as np

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
