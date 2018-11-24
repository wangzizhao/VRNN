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
				 batch_size,
				 n_particles = 1,
				 x_ft_Dhs = None, z_ft_Dhs = None,
				 prior_Dhs = None, enc_Dhs = None, dec_Dhs = None,
				 sigma_bias_init = 0.6):
		self.Dx = Dx
		self.Dh = Dh
		self.Dz = Dz

		self.batch_size = batch_size
		self.n_particles = n_particles

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
												   activation_fn=tf.nn.relu,
												   reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				prior_mu = fully_connected(prior_hidden, self.Dz, 
										   weights_initializer=xavier_initializer(uniform=False), 
										   biases_initializer=tf.constant_initializer(0),
										   activation_fn=None,
										   reuse=tf.AUTO_REUSE, scope = "mu")
				prior_sigma = fully_connected(prior_hidden, self.Dz, 
											  weights_initializer=xavier_initializer(uniform=False), 
											  biases_initializer=tf.constant_initializer(self.sigma_bias_init),
											  activation_fn=tf.nn.softplus,
											  reuse=tf.AUTO_REUSE, scope = "sigma")

			with tf.variable_scope("phi_x"):
				phi_x_hidden = x
				for i, phi_x_Dh in enumerate(self.phi_x_Dhs):
					phi_x_hidden = fully_connected(phi_x_hidden, phi_x_Dh, 
												   weights_initializer=xavier_initializer(uniform=False), 
												   biases_initializer=tf.constant_initializer(0),
												   activation_fn=tf.nn.relu,
												   reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				phi_x = fully_connected(phi_x_hidden, self.Dx,
										weights_initializer=xavier_initializer(uniform=False), 
										biases_initializer=tf.constant_initializer(0),
										activation_fn=tf.nn.relu,
										reuse=tf.AUTO_REUSE, scope = "phi_x")
				phi_x_expanded = tf.expand_dims(phi_x, axis = 1, name = "phi_x_expanded")
				phi_x_tiled = tf.tile(phi_x_expanded, (1, self.n_particles, 1), name = "phi_x_tiled")

			with tf.variable_scope("encoder"):
				enc_hidden = tf.concat(values=(phi_x_tiled, h), axis = -1)
				for i, enc_Dh in enumerate(self.enc_Dhs):
					enc_hidden = fully_connected(enc_hidden, enc_Dh, 
												 weights_initializer=xavier_initializer(uniform=False), 
												 biases_initializer=tf.constant_initializer(0),
												 activation_fn=tf.nn.relu,
												 reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				enc_mu = fully_connected(enc_hidden, self.Dz, 
										 weights_initializer=xavier_initializer(uniform=False), 
										 biases_initializer=tf.constant_initializer(0),
										 activation_fn=None,
										 reuse=tf.AUTO_REUSE, scope = "mu")
				enc_sigma = fully_connected(enc_hidden, self.Dz, 
											weights_initializer=xavier_initializer(uniform=False), 
											biases_initializer=tf.constant_initializer(self.sigma_bias_init),
											activation_fn=tf.nn.softplus,
											reuse=tf.AUTO_REUSE, scope = "sigma")

			assert len(x.shape) == 2, "x doesn't keep the shape [batch_size, Dx]"
			eps = tf.random_normal([self.batch_size, self.n_particles, self.Dz], name = "eps")
			z = tf.add(enc_mu, tf.multiply(enc_sigma, eps), name = "z")

			with tf.variable_scope("phi_z"):
				phi_z_hidden = z
				for i, phi_z_Dh in enumerate(self.phi_z_Dhs):
					phi_z_hidden = fully_connected(phi_z_hidden, phi_z_Dh, 
												   weights_initializer=xavier_initializer(uniform=False), 
												   biases_initializer=tf.constant_initializer(0),
												   activation_fn=tf.nn.relu,
												   reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				phi_z = fully_connected(phi_z_hidden, self.Dz,
										weights_initializer=xavier_initializer(uniform=False), 
										biases_initializer=tf.constant_initializer(0),
										activation_fn=tf.nn.relu,
										reuse=tf.AUTO_REUSE, scope = "phi_z")
				# for prediction
				phi_z_prior_hidden = prior_mu
				for i, phi_z_Dh in enumerate(self.phi_z_Dhs):
					phi_z_prior_hidden = fully_connected(phi_z_prior_hidden, phi_z_Dh, 
														 weights_initializer=xavier_initializer(uniform=False), 
														 biases_initializer=tf.constant_initializer(0),
														 activation_fn=tf.nn.relu,
														 reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				phi_z_prior = fully_connected(phi_z_prior_hidden, self.Dz,
											  weights_initializer=xavier_initializer(uniform=False), 
											  biases_initializer=tf.constant_initializer(0),
											  activation_fn=tf.nn.relu,
											  reuse=tf.AUTO_REUSE, scope = "phi_z")

			with tf.variable_scope("dec"):
				dec_hidden = tf.concat(values=(phi_z, h), axis = -1)
				for i, dec_Dh in enumerate(self.dec_Dhs):
					dec_hidden = fully_connected(dec_hidden, dec_Dh, 
												 weights_initializer=xavier_initializer(uniform=False), 
												 biases_initializer=tf.constant_initializer(0),
												 activation_fn=tf.nn.relu,
												 reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				dec_mu = fully_connected(dec_hidden, self.Dx, 
										 weights_initializer=xavier_initializer(uniform=False), 
										 biases_initializer=tf.constant_initializer(0),
										 activation_fn=None,
										 reuse=tf.AUTO_REUSE, scope = "mu")
				dec_sigma = fully_connected(dec_hidden, self.Dx, 
											weights_initializer=xavier_initializer(uniform=False), 
											biases_initializer=tf.constant_initializer(self.sigma_bias_init),
											activation_fn=tf.nn.softplus,
											reuse=tf.AUTO_REUSE, scope = "sigma")

				# for prediction
				dec_predict_hidden = tf.concat(values=(phi_z_prior, h), axis = -1)
				for i, dec_Dh in enumerate(self.dec_Dhs):
					dec_predict_hidden = fully_connected(dec_predict_hidden, dec_Dh, 
														 weights_initializer=xavier_initializer(uniform=False), 
														 biases_initializer=tf.constant_initializer(0),
														 activation_fn=tf.nn.relu,
														 reuse=tf.AUTO_REUSE, scope = "Dh_{}".format(i))
				x_prediction = fully_connected(dec_predict_hidden, self.Dx, 
											   weights_initializer=xavier_initializer(uniform=False), 
											   biases_initializer=tf.constant_initializer(0),
											   activation_fn=None,
											   reuse=tf.AUTO_REUSE, scope = "mu")

			# flat all things to rank 2, because lstm only accept rank 2 input and state
			with tf.variable_scope("lstm_update"):
				phi_x_phi_z = tf.concat((phi_x_tiled, phi_z), axis = -1, name = "phi_x_phi_z")
				phi_x_phi_z_flat = tf.reshape(phi_x_phi_z, (-1, self.Dx + self.Dz), name = "phi_x_phi_z_flat")

				c_flat = tf.reshape(c, (-1, self.Dh), name = "c_flat")
				h_flat = tf.reshape(h, (-1, self.Dh), name = "h_flat")
				state_flat = tf.nn.rnn_cell.LSTMStateTuple(c_flat, h_flat)

				output, next_state_flat = self.lstm(phi_x_phi_z_flat, state_flat)

				next_c_flat, next_h_flat = next_state_flat
				next_c = tf.reshape(next_c_flat, (self.batch_size, self.n_particles, self.Dh), name = "next_c")
				next_h = tf.reshape(next_h_flat, (self.batch_size, self.n_particles, self.Dh), name = "next_h")
				next_state = tf.nn.rnn_cell.LSTMStateTuple(next_c, next_h)

		return (prior_mu, prior_sigma, enc_mu, enc_sigma, dec_mu, dec_sigma, z, x_prediction), next_state

class VRNN_model():
	def __init__(self,
				 VRNNCell,
				 batch_size,
				 n_particles,
				 initial_state_all_zero = True,
				 is_lstm_Dh = 50,
				 sigma_min = 1e-9):
		self.VRNNCell = VRNNCell
		self.batch_size = batch_size
		self.n_particles = n_particles
		self.initial_state_all_zero = initial_state_all_zero
		self.sigma_min = sigma_min
		self.is_lstm_Dh = is_lstm_Dh
		self.is_f_lstm = tf.contrib.rnn.LSTMBlockCell(is_lstm_Dh, name = "is_f_lstm")
		self.is_b_lstm = tf.contrib.rnn.LSTMBlockCell(is_lstm_Dh, name = "is_b_lstm")

	def get_initial_state(self, Inputs):
		with tf.variable_scope("initial_state"):
			if self.initial_state_all_zero:
				initial_c = tf.zeros([self.batch_size, self.n_particles, self.VRNNCell.c_size], name = "initial_c")
				initial_h = tf.zeros([self.batch_size, self.n_particles, self.VRNNCell.h_size], name = "initial_h")
				initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c, initial_h)
			else:
				is_lstm_initial_c = tf.zeros([self.batch_size, self.is_lstm_Dh], name = "is_lstm_initial_c")
				is_lstm_initial_h = tf.zeros([self.batch_size, self.is_lstm_Dh], name = "is_lstm_initial_h")
				is_lstm_initial_state = tf.nn.rnn_cell.LSTMStateTuple(is_lstm_initial_c, is_lstm_initial_h)

				f_Inputs = list(Inputs[:len(Inputs)//2])
				b_Inputs = list(reversed(Inputs[:len(Inputs)//2]))
				_, f_last_state = tf.nn.static_rnn(self.is_f_lstm, f_Inputs, initial_state = is_lstm_initial_state)
				_, b_last_state = tf.nn.static_rnn(self.is_b_lstm, b_Inputs, initial_state = is_lstm_initial_state)

				f_last_h, b_last_h = f_last_state[1], b_last_state[1]
				f_b_last_h = tf.concat([f_last_h, b_last_h], axis = -1, name = "f_b_last_state")

				initial_c_h_flat = fully_connected(f_b_last_h, 2*self.VRNNCell.c_size,
												   weights_initializer=xavier_initializer(uniform=False),
												   biases_initializer=tf.constant_initializer(0),
												   activation_fn=None,
												   reuse=tf.AUTO_REUSE, scope = "initial_state")

				initial_c_h = tf.reshape(initial_c_h_flat, [self.batch_size, 2, self.VRNNCell.c_size], name = "initial_c_h")
				initial_c, initial_h = tf.unstack(initial_c_h, axis=1, name="unstack_initial_c_h")

				initial_c_expanded = tf.expand_dims(initial_c, axis = 1, name = "initial_c_expanded")
				initial_c_tiled = tf.tile(initial_c_expanded, (1, self.n_particles, 1), name = "initial_c_tiled")
				initial_h_expanded = tf.expand_dims(initial_h, axis = 1, name = "initial_h_expanded")
				initial_h_tiled = tf.tile(initial_h_expanded, (1, self.n_particles, 1), name = "initial_h_tiled")

				initial_state = tf.nn.rnn_cell.LSTMStateTuple(initial_c_tiled, initial_h_tiled)
			return initial_state

	def get_output(self, Input_BxTxDx):
		"""
		Input shape: [batch_size, time, x_dim]
		"""

		Inputs = tf.unstack(Input_BxTxDx, axis = 1)
		initial_state = self.get_initial_state(Inputs)
		outputs, last_state = tf.nn.static_rnn(self.VRNNCell, Inputs, initial_state = initial_state)

		names = ["prior_mu", "prior_sigma", "enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "z_sample", "x_prediction"]
		outputs_reshaped = []
		for i, name in enumerate(names):
			x_BxTxNxD = tf.stack([output[i] for output in outputs], axis = 1, name = name)
			outputs_reshaped.append(x_BxTxNxD)

		return outputs_reshaped, last_state

	def get_loss(self, Input, output):
		def gaussian_log_prob(x, mu, sigma, name = "gaussian_log_prob"):
			with tf.variable_scope(name):
				sigma = tf.where(tf.is_nan(sigma), tf.zeros_like(sigma), sigma)
				scale_diag = tf.maximum(sigma, self.sigma_min, name = "scale_diag")
				mvn = tfd.MultivariateNormalDiag(loc = mu, scale_diag = scale_diag,
												 validate_args=True,
												 allow_nan_stats = False, 
												 name = "mvn")
				return mvn.log_prob(x)

		def KL_gauss_gauss(mu1, sigma1, mu2, sigma2):
			with tf.variable_scope("KL_gauss_gauss"):
				sigma1 = tf.where(tf.is_nan(sigma1), tf.zeros_like(sigma1), sigma1)
				sigma2 = tf.where(tf.is_nan(sigma2), tf.zeros_like(sigma2), sigma2)
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

		prior_mu, prior_sigma, enc_mu, enc_sigma, dec_mu, dec_sigma, z_sample, _ = output

		Input_expanded = tf.expand_dims(Input, axis = 2, name = "Input_expanded")
		Input_tiled = tf.tile(Input_expanded, (1, 1, self.n_particles, 1), name = "Input_tiled")

		dec_log_prob = gaussian_log_prob(Input_tiled, dec_mu, dec_sigma, name = "dec_log_prob")
		if self.n_particles > 1:
			prior_log_prob = gaussian_log_prob(z_sample, prior_mu, prior_sigma, name = "prior_log_prob")
			enc_log_prob = gaussian_log_prob(z_sample, enc_mu, enc_sigma, name = "enc_log_prob")
			log_w = dec_log_prob + prior_log_prob - enc_log_prob
			w = tf.exp(log_w, name = "w")
			w_mean = tf.reduce_mean(w, axis = -1, name = "w_mean")
			log_w_mean = tf.log(w_mean, name = "log_w_mean")
			loss = tf.reduce_mean(log_w_mean, name = "loss")

		else:
			KL_loss = KL_gauss_gauss(enc_mu, enc_sigma, prior_mu, prior_sigma)

			loss = tf.reduce_mean(dec_log_prob - KL_loss, name = "loss")

		log_loss = tf.log(loss, name = "log_loss")
		return loss
		return log_loss

	def get_prediction(self, output):
		return output[-1]

	def get_hidden(self, output):
		return output[2]

	def get_MSE(self, sess, prediction, obs, obs_set):
		total_MSE = 0
		for i in range(0, len(obs_set), self.batch_size):
			prediction_val = sess.run(prediction, feed_dict = {obs:obs_set[i:i+self.batch_size]})
			prediction_val = np.mean(prediction_val, axis = 2)
			total_MSE += np.mean((prediction_val - obs_set[i:i+self.batch_size])**2)
		return total_MSE/(len(obs_set)/self.batch_size)

	def get_loss_val(self, sess, loss, obs, obs_set):
		total_loss = 0
		for i in range(0, len(obs_set), self.batch_size):
			total_loss += sess.run(loss, feed_dict = {obs:obs_set[i:i+self.batch_size]})
		return total_loss/(len(obs_set)/self.batch_size)