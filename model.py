"""
some of the code is written with reference to https://github.com/phreeza/tensorflow-vrnn
"""

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

import numpy as np 

class VRNNCell(tf.contrib.rnn.LSTMBlockCell):
	"""Variational RNN cell."""

	def __init__(self, Dx, Dh, Dz,
				 x_ft_Dhs = None, z_ft_Dhs = None,
				 prior_Dhs = None, enc_Dhs = None, dec_Dhs = None,
				 sigma_bias_init = 0.6):
		self.Dx = Dx
		self.Dh = Dh
		self.Dz = Dz
		self.phi_x_Dhs = x_ft_Dhs or []
		self.phi_z_Dhs = z_ft_Dhs or []
		self.prior_Dhs = prior_Dhs or [Dz]
		self.enc_Dhs = enc_Dhs or [Dz]
		self.dec_Dhs = dec_Dhs or [Dx]

		self.sigma_bias_init = sigma_bias_init

		self.lstm = tf.contrib.rnn.LSTMBlockCell(self.Dh)

	@property
	def c_size(self):
		return self.Dh

	@property
	def h_size(self):
		return self.Dh
	
	@property
	def state_size(self):
		return (self.Dh, self.Dh)

	@property
	def output_size(self):
		return self.Dh

	def __call__(self, x, state, scope=None):
		with tf.variable_scope(scope or type(self).__name__):
			h, c = state

			with tf.variable_scope("Prior"):
				prior_hidden = h
				for i, prior_Dh in enumerate(self.prior_Dhs):
					prior_hidden = fully_connected(prior_hidden, prior_Dh, 
												   weights_initializer=xavier_initializer(uniform=False), 
												   biases_initializer=tf.constant_initializer(0),
												   activation_fn = tf.nn.relu,
												   reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				prior_mu = fully_connected(prior_hidden, self.Dz, 
										   weights_initializer=xavier_initializer(uniform=False), 
										   biases_initializer=tf.constant_initializer(0),
										   activation_fn = None,
										   reuse = tf.AUTO_REUSE, scope = "mu")
				prior_sigma = fully_connected(prior_hidden, self.Dz, 
											  weights_initializer=xavier_initializer(uniform=False), 
											  biases_initializer=tf.constant_initializer(self.sigma_bias_init),
											  activation_fn = tf.nn.softplus,
											  reuse = tf.AUTO_REUSE, scope = "sigma")

			with tf.variable_scope("phi_x"):
				phi_x_hidden = x
				for i, phi_x_Dh in enumerate(self.phi_x_Dhs):
					phi_x_hidden = fully_connected(phi_x_hidden, phi_x_Dh, 
												   weights_initializer=xavier_initializer(uniform=False), 
												   biases_initializer=tf.constant_initializer(0),
												   activation_fn = tf.nn.relu,
												   reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				phi_x = fully_connected(phi_x_hidden, self.Dx,
										weights_initializer=xavier_initializer(uniform=False), 
										biases_initializer=tf.constant_initializer(0),
										activation_fn = tf.nn.relu,
										reuse = tf.AUTO_REUSE, scope = "phi_x")

			with tf.variable_scope("encoder"):
				enc_hidden = tf.concat(values=(phi_x, h), axis = -1)
				for i, enc_Dh in enumerate(self.enc_Dhs):
					enc_hidden = fully_connected(enc_hidden, enc_Dh, 
												 weights_initializer=xavier_initializer(uniform=False), 
												 biases_initializer=tf.constant_initializer(0),
												 activation_fn = tf.nn.relu,
												 reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				enc_mu = fully_connected(enc_hidden, self.Dz, 
										 weights_initializer=xavier_initializer(uniform=False), 
										 biases_initializer=tf.constant_initializer(0),
										 activation_fn = None,
										 reuse = tf.AUTO_REUSE, scope = "mu")
				enc_sigma = fully_connected(enc_hidden, self.Dz, 
											weights_initializer=xavier_initializer(uniform=False), 
											biases_initializer=tf.constant_initializer(self.sigma_bias_init),
											activation_fn = tf.nn.softplus,
											reuse = tf.AUTO_REUSE, scope = "sigma")

			eps = tf.random_normal((x.get_shape().as_list()[0], self.Dz), name = "eps")
			z = tf.add(enc_mu, tf.multiply(enc_sigma, eps), name = "z")

			with tf.variable_scope("phi_z"):
				phi_z_hidden = z
				for i, phi_z_Dh in enumerate(self.phi_z_Dhs):
					phi_z_hidden = fully_connected(phi_z_hidden, phi_z_Dh, 
												   weights_initializer=xavier_initializer(uniform=False), 
												   biases_initializer=tf.constant_initializer(0),
												   activation_fn = tf.nn.relu,
												   reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				phi_z = fully_connected(phi_z_hidden, self.Dz,
										weights_initializer=xavier_initializer(uniform=False), 
										biases_initializer=tf.constant_initializer(0),
										activation_fn = tf.nn.relu,
										reuse = tf.AUTO_REUSE, scope = "phi_z")
				# for prediction
				phi_z_prior_hidden = prior_mu
				for i, phi_z_Dh in enumerate(self.phi_z_Dhs):
					phi_z_prior_hidden = fully_connected(phi_z_prior_hidden, phi_z_Dh, 
														 weights_initializer=xavier_initializer(uniform=False), 
														 biases_initializer=tf.constant_initializer(0),
														 activation_fn = tf.nn.relu,
														 reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				phi_z_prior = fully_connected(phi_z_prior_hidden, self.Dz,
											  weights_initializer=xavier_initializer(uniform=False), 
											  biases_initializer=tf.constant_initializer(0),
											  activation_fn = tf.nn.relu,
											  reuse = tf.AUTO_REUSE, scope = "phi_z")

			with tf.variable_scope("dec"):
				dec_hidden = tf.concat(values=(phi_z, h), axis = -1)
				for i, dec_Dh in enumerate(self.dec_Dhs):
					dec_hidden = fully_connected(dec_hidden, dec_Dh, 
												 weights_initializer=xavier_initializer(uniform=False), 
												 biases_initializer=tf.constant_initializer(0),
												 activation_fn = tf.nn.relu,
												 reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				dec_mu = fully_connected(dec_hidden, self.Dx, 
										 weights_initializer=xavier_initializer(uniform=False), 
										 biases_initializer=tf.constant_initializer(0),
										 activation_fn = None,
										 reuse = tf.AUTO_REUSE, scope = "mu")
				dec_sigma = fully_connected(dec_hidden, self.Dx, 
											weights_initializer=xavier_initializer(uniform=False), 
											biases_initializer=tf.constant_initializer(self.sigma_bias_init),
											activation_fn = tf.nn.softplus,
											reuse = tf.AUTO_REUSE, scope = "sigma")

				# for prediction
				dec_predict_hidden = tf.concat(values=(phi_z_prior, h), axis = -1)
				for i, dec_Dh in enumerate(self.dec_Dhs):
					dec_predict_hidden = fully_connected(dec_predict_hidden, dec_Dh, 
														 weights_initializer=xavier_initializer(uniform=False), 
														 biases_initializer=tf.constant_initializer(0),
														 activation_fn = tf.nn.relu,
														 reuse = tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				x_prediction = fully_connected(dec_predict_hidden, self.Dx, 
											   weights_initializer=xavier_initializer(uniform=False), 
											   biases_initializer=tf.constant_initializer(0),
											   activation_fn = None,
											   reuse = tf.AUTO_REUSE, scope = "mu")

			output, next_state = self.lstm(tf.concat((phi_x, phi_z), axis = -1), state)

		return (prior_mu, prior_sigma, enc_mu, enc_sigma, dec_mu, dec_sigma, x_prediction), next_state

class VRNN_model():
	def __init__(self, VRNNCell, batch_size, 
				 initial_state_trainable = False,
				 sigma_min = 1e-9):
		self.VRNNCell = VRNNCell
		self.batch_size = batch_size
		self.initial_state_trainable = initial_state_trainable
		self.sigma_min = sigma_min

	def get_output(self, Input_BxTxDx):
		"""
		Input shape: [batch_size, time, x_dim]
		"""
		with tf.variable_scope("initial_state", reuse = tf.AUTO_REUSE):
			initial_c = tf.get_variable("initial_c", [self.batch_size, self.VRNNCell.c_size], 
										tf.float32, initializer=tf.zeros_initializer(), 
										trainable = self.initial_state_trainable)
			initial_h = tf.get_variable("initial_h", [self.batch_size, self.VRNNCell.h_size], 
										tf.float32, initializer=tf.zeros_initializer(), 
										trainable = self.initial_state_trainable)

			initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)

		Inputs = tf.unstack(Input_BxTxDx, axis = 1)
		outputs, last_state = tf.nn.static_rnn(self.VRNNCell, Inputs, initial_state = initial_state)

		names = ["prior_mu", "prior_sigma", "enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "x_prediction"]
		outputs_reshaped = []
		for i, name in enumerate(names):
			x_BxTxD = tf.stack([output[i] for output in outputs], axis = 1, name = name)
			outputs_reshaped.append(x_BxTxD)

		return outputs_reshaped, last_state

	def get_loss(self, Input, output):
		def gaussian_log_prob(x, mu, sigma):
			with tf.variable_scope("gaussian_log_prob"):
				scale_diag = tf.maximum(sigma, self.sigma_min, name = "scale_diag")
				mvn = tfd.MultivariateNormalDiag(loc = mu, scale_diag = scale_diag,
												 validate_args=True,
												 allow_nan_stats = False, 
												 name = "mvn")
				return mvn.log_prob(x)

		def KL_gauss_gauss(mu1, sigma1, mu2, sigma2):
			with tf.variable_scope("KL_gauss_gauss"):
				sigma1 = tf.maximum(self.sigma_min, sigma1)
				sigma2 = tf.maximum(self.sigma_min, sigma2)
				return tf.reduce_sum(0.5 * 
										(
											2 * tf.log(sigma2, name="log_sigma2") 
										  - 2 * tf.log(sigma1, name="log_sigma1")
										  + (sigma1**2 + (mu1 - mu2)**2) / sigma2**2
										  - 1
										), 
									axis = -1, name = "KL_gauss_gauss")

		prior_mu, prior_sigma, enc_mu, enc_sigma, dec_mu, dec_sigma, _ = output
		KL_loss = KL_gauss_gauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
		log_prob_loss = gaussian_log_prob(Input, dec_mu, dec_sigma)

		return tf.log(tf.reduce_mean(KL_loss - log_prob_loss), name = "loss")

	def get_prediction(self, output):
		return output[-1]

	def get_loss_val(self, sess, loss, obs, obs_set):
		total_loss = 0
		for i in range(0, len(obs_set), self.batch_size):
			total_loss += sess.run(loss, feed_dict = {obs:obs_set[i:i+self.batch_size]})
		return total_loss/(len(obs_set)/self.batch_size)