import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf

# import from files
from model import VRNNCell, VRNN_model
from data import get_stock_data
from result_saving import *
from sampler import create_train_test_dataset
from transformation import *
from distribution import *

# for data saving stuff
import sys
import pickle
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":

	# ============================================ parameter part ============================================ #
	# training hyperparameters
	time = 30

	batch_size = 16
	lr = 1e-4
	epoch = 2

	n_train = 1 * batch_size
	n_test  = 1 * batch_size

	seed = 1
	tf.set_random_seed(seed)
	np.random.seed(seed)

	use_stock_data = True

	# model hyperparameters
	if use_stock_data:
		Dx = 5
		Dh = 100
		Dz = 100
		x_ft_Dhs 	= [100, 100]
		z_ft_Dhs 	= [100, 100]
		prior_Dhs 	= [100, 100, 100]
		enc_Dhs 	= [100, 100, 100]
		decoder_Dhs = [100, 100, 100]
	else:
		Dx = 10
		Dh = 50
		Dz = 2
		x_ft_Dhs = [100]
		z_ft_Dhs = [100]
		prior_Dhs = [100]
		enc_Dhs = [100]
		decoder_Dhs = [100]

	initial_state_all_zero = False
	is_lstm_Dh = 50
	sigma_min = 1e-6

	# printing and data saving params
	print_freq = 1

	store_res = True
	save_freq = 10
	saving_num = min(n_train, 1*batch_size)
	# rslt_dir_name = "VRNN"
	rslt_dir_name = "dow_jones"

	# ============================================= dataset part ============================================= #
	if use_stock_data:
		stock_idx_name = "dow_jones" # or "nasdaq" or "sp_500"
		obs_train, obs_test = get_stock_data(stock_idx_name, time, n_train, n_test, Dx)
		obs_train /= 1e4
		obs_test /= 1e4
		hidden_train = hidden_test = None
	else:

		#define transition f and emission g

		#f:fhn, g:linear
		#Dz:2, Dx:10
		fhn_params = (1.0, 0.95, 0.05, 1.0, 0.15)
		f_sample_tran = fhn.fhn_transformation(fhn_params)
		f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

		linear_params = np.array([[0.99,0.01], [0.91,0.09], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])
		g_sample_tran = linear.linear_transformation(linear_params)
		mvn_sigma = np.eye(10)
		g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)

		#f:lorez, g:linear
		#Dz:3, Dx:10
		# lorenz_params = (10.0, 28.0, 8.0/3.0, 0.01)
		# f_sample_tran = lorenz.lorenz_transformation(lorenz_params)
		# f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

		# linear_params = np.array([[0.95,0.05, 0.01], [0.8,0.15, 0.05], [0.7, 0.1, 0.2], [0.5, 0.4, 0.1], [0.6, 0.1, 0.3], [0.4, 0.2, 0.4], [0.4, 0.5, 0.1], [0.3, 0.2, 0.5], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
		# g_sample_tran = linear.linear_transformation(linear_params)
		# g_sample_dist = poisson.poisson(g_sample_tran)

		#f:linear, g:linear,poisson
		#Dz:2, Dx:2
		# f_linear_params = np.array([[0.95,0],[0, 0.83]])
		# f_sample_tran = linear.linear_transformation(f_linear_params)
		# f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

		# g_linear_params = np.array([[0.8,0.2],[0.3, 0.7]])
		# g_sample_tran = linear.linear_transformation(g_linear_params)
		# g_sample_dist = poisson.poisson(g_sample_tran)

		#f:linear, g:linear,mvn
		#Dz:2, Dx:2
		# f_linear_params = np.array([[0.95,0],[0, 0.83]])
		# f_sample_tran = linear.linear_transformation(f_linear_params)
		# f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

		# g_linear_params = np.array([[2,3],[5,10]])
		# g_sample_tran = linear.linear_transformation(g_linear_params)
		# mvn_sigma = np.array([[1,0],[0,1]])
		# g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)

		hidden_train, obs_train, hidden_test, obs_test = create_train_test_dataset(n_train, n_test, time, Dz, Dx, f_sample_dist, g_sample_dist, None, -3, 3)

	# ============================================== model part ============================================== #
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name = "obs")

	myVRNNCell = VRNNCell(Dx, Dh, Dz,
						  x_ft_Dhs, z_ft_Dhs,
						  prior_Dhs, enc_Dhs, decoder_Dhs)
	model = VRNN_model(myVRNNCell,
					   batch_size,
					   initial_state_all_zero = initial_state_all_zero,
					   is_lstm_Dh = is_lstm_Dh,
					   sigma_min = sigma_min)

	output, last_state = model.get_output(obs)
	loss = model.get_loss(obs, output)
	hidden = model.get_hidden(output)
	prediction = model.get_prediction(output)

	with tf.name_scope("train"):
		train_op = tf.train.AdamOptimizer(lr).minimize(-loss)
	init = tf.global_variables_initializer()

	# =========================================== data saving part =========================================== #
	if store_res == True:
		experiment_params = {"time":time, 
							 "batch_size":batch_size,
							 "lr":lr, 
							 "epoch":epoch, 
							 "seed":seed, 
							 "n_train":n_train,
							 "rslt_dir_name":rslt_dir_name}
		RLT_DIR = create_RLT_DIR(experiment_params)
		writer = tf.summary.FileWriter(RLT_DIR)
		saver = tf.train.Saver()

	loss_trains = []
	loss_tests = []

	# ============================================= training part ============================================ #
	with tf.Session() as sess:

		sess.run(init)

		if store_res == True:
			writer.add_graph(sess.graph)

		loss_train = model.get_loss_val(sess, loss, obs, obs_train)
		loss_test  = model.get_loss_val(sess, loss, obs, obs_test)
		print("iter {:>3}, train loss: {:>7.3f}, test loss: {:>7.3f}"\
			.format(0, loss_train, loss_test))

		loss_trains.append(loss_train)
		loss_tests.append(loss_test)

		for i in range(epoch):
			if hidden_train is not None:
				hidden_train, obs_train = shuffle(hidden_train, obs_train)
			else:
				obs_train = shuffle(obs_train)
			for j in range(0, len(obs_train), batch_size):
				sess.run(train_op, feed_dict={obs:obs_train[j:j+batch_size]})
				
			# print training and testing loss
			if (i+1)%print_freq == 0:
				loss_train = model.get_loss_val(sess, loss, obs, obs_train)
				loss_test  = model.get_loss_val(sess, loss, obs, obs_test)
				print("iter {:>3}, train loss: {:>7.3f}, test loss: {:>7.3f}"\
					.format(i+1, loss_train, loss_test))

				loss_trains.append(loss_train)
				loss_tests.append(loss_test)

			if store_res == True and (i+1)%save_freq == 0:
				if not os.path.exists(RLT_DIR+"model"): os.makedirs(RLT_DIR+"model")
				saver.save(sess, RLT_DIR+"model/model_epoch", global_step=i+1)

		print("finish training")


		#hidden_train(generated hidden variable) compare with output[2] (predicted hidden variable, Batch size * T * Dz)
		if store_res and hidden_train is not None:
			hidden_val = np.zeros((saving_num, time, Dz))
			for i in range(0, saving_num, batch_size):
				hidden_val[i:i+batch_size] = sess.run(hidden, feed_dict={obs:obs_train[i:i+batch_size]})
			plot_hidden(RLT_DIR, hidden_val, hidden_train[0:saving_num])

		if store_res:
			prediction_val = np.zeros((saving_num, time, Dx))
			for i in range(0, saving_num, batch_size):
				prediction_val[i:i+batch_size] = sess.run(prediction, feed_dict={obs:obs_train[i:i+batch_size]})
			plot_expression(RLT_DIR, prediction_val, obs_train[0:saving_num])

	# ======================================== anther data saving part ======================================== #


	if store_res == True:

		params_dict = {"time":time,
					   "batch_size":batch_size,
					   "lr":lr,
					   "epoch":epoch,
					   "n_train":n_train,
					   "seed":seed}
		loss_dict = {"loss_trains":loss_trains,
					 "loss_tests": loss_tests}
		data_dict = {"params_dict":params_dict,
					 "loss_dict":loss_dict}
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)

		learned_val_dict = {"prediction":prediction_val}
		data_dict["learned_val_dict"] = learned_val_dict
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)

		plot_loss(RLT_DIR, loss_trains, loss_tests)