import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from distribution.base import distribution

#just used in sampler, so no need to implement log_prob
class poisson(distribution):
	"""
	multivariate poisson distribution
	"""
	def __init__(self, transformation):
		self.transformation = transformation

	def sample(self, Input):
		assert isinstance(Input, np.ndarray), "Input for poisson must be np.ndarray, but {} is given".format(type(Input))

		def safe_softplus(z, limit=30):
			z[z < limit] = np.log(1.0 + np.exp(z[z < limit]))
			return z

		lambdas = safe_softplus(self.transformation.transform(Input))
		return np.random.poisson(lambdas)