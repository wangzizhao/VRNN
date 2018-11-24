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
	time = 100

	n_particles = 100
	batch_size = 16

	lr = 1e-4
	epoch = 100

	n_train = 50 * batch_size
	n_test  = 1  * batch_size

	seed = 0
	tf.set_random_seed(seed)
	np.random.seed(seed)

	use_stock_data = False

	# model hyperparameters
	if use_stock_data:
		Dx = 6
		Dh = 100
		Dz = 100
		x_ft_Dhs 	= [100, 100]
		z_ft_Dhs 	= [100, 100]
		prior_Dhs 	= [100, 100, 100]
		enc_Dhs 	= [100, 100, 100]
		decoder_Dhs = [100, 100, 100]
	else:
		Dx = 1
		Dh = 50
		Dz = 2
		x_ft_Dhs = [100]
		z_ft_Dhs = [100]
		prior_Dhs = [100]
		enc_Dhs = [100]
		decoder_Dhs = [100]

	initial_state_all_zero = True
	is_lstm_Dh = 50
	sigma_min = 1e-6

	# printing and data saving params
	print_freq = 1

	store_res = True
	save_freq = 10
	saving_num = min([n_train, n_test, 1*batch_size])
	# rslt_dir_name = "VRNN"
	#rslt_dir_name = "dow_jones"
	rslt_dir_name = "generated_data"

	# ============================================= dataset part ============================================= #
	if use_stock_data:
		stock_idx_name = "dow_jones" # or "nasdaq" or "sp_500"
		obs_train, obs_test = get_stock_data(stock_idx_name, time, n_train, n_test, Dx)
		hidden_train = hidden_test = None
	else:

		#define transition f and emission g

		#f:fhn, g:linear
		#Dz:2, Dx:10
		# fhn_params = (1.0, 0.95, 0.05, 1.0, 0.15)
		# f_sample_tran = fhn.fhn_transformation(fhn_params)
		# f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

		# #linear_params = np.array([[0.99,0.01], [0.91,0.09], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])
		# linear_params = np.array([[0,1]])
		# g_sample_tran = linear.linear_transformation(linear_params)
		# mvn_sigma = 0.2 * np.eye(1)
		# g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)

		#f:lorez, g:linear
		#Dz:3, Dx:10
		# lorenz_params = (10.0, 28.0, 8.0/3.0, 0.01)
		# f_sample_tran = lorenz.lorenz_transformation(lorenz_params)
		# f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

		# #linear_params = np.array([[0.95,0.05, 0.01], [0.8,0.15, 0.05], [0.7, 0.1, 0.2], [0.5, 0.4, 0.1], [0.6, 0.1, 0.3], [0.4, 0.2, 0.4], [0.4, 0.5, 0.1], [0.3, 0.2, 0.5], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
		# linear_params = np.random.randn(1,3)
		# g_sample_tran = linear.linear_transformation(linear_params)
		# mvn_sigma = 0.2 * np.eye(1)
		# g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)
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


		#1
		#f:fhn, g:linear，epoch = 100, time = 100, initial_state_all_zero = False, linear param [0,1]
		#Dz:2, Dx:1

		#2 linear param [1,0]

		#3 linear param [1,1]

		#4 initial_state_all_zero = True, n_particles = 1, linear param [0,1]

		#5 linear param [1,0]

		#6 linear param [1,1]

		#7 n_particles = 100, linear param [0,1]

		#8 linear param [1,0]

		#9 linear param [1,1]

		fhn_params = (1.0, 0.95, 0.05, 1.0, 0.15)
		f_sample_tran = fhn.fhn_transformation(fhn_params)
		f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)
		linear_params = np.array([[1,1]])
		g_sample_tran = linear.linear_transformation(linear_params)
		mvn_sigma = 0.2 * np.eye(1)
		g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)






		hidden_train, obs_train, hidden_test, obs_test = create_train_test_dataset(n_train, n_test, time, Dz, Dx, f_sample_dist, g_sample_dist, None, -3, 3)

	# ============================================== model part ============================================== #
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name = "obs")

	myVRNNCell = VRNNCell(Dx, Dh, Dz,
						  batch_size,
						  n_particles,
						  x_ft_Dhs, z_ft_Dhs,
						  prior_Dhs, enc_Dhs, decoder_Dhs)
	model = VRNN_model(myVRNNCell,
					   batch_size,
					   n_particles,
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
							 "n_particles":n_particles,
							 "batch_size":batch_size,
							 "lr":lr,
							 "epoch":epoch,
							 "seed":seed,
							 "n_train":n_train,
							 "rslt_dir_name":rslt_dir_name}

		print("experiment_params")
		for key, val in experiment_params.items():
			print("\t{}:{}".format(key, val))

		RLT_DIR = create_RLT_DIR(experiment_params)
		writer = tf.summary.FileWriter(RLT_DIR)
		saver = tf.train.Saver()

	loss_trains = []
	loss_tests = []
	MSE_trains = []
	MSE_tests = []

	# ============================================= training part ============================================ #
	with tf.Session() as sess:

		sess.run(init)

		if store_res == True:
			writer.add_graph(sess.graph)

		loss_train = model.get_loss_val(sess, loss, obs, obs_train)
		loss_test  = model.get_loss_val(sess, loss, obs, obs_test)
		MSE_train = model.get_MSE(sess, prediction, obs, obs_train)
		MSE_test  = model.get_MSE(sess, prediction, obs, obs_test)
		print("iter {:>3}, train loss: {:>7.3f}, test loss: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}"\
			.format(0, loss_train, loss_test, MSE_train, MSE_test))

		loss_trains.append(loss_train)
		loss_tests.append(loss_test)
		MSE_trains.append(MSE_train)
		MSE_tests.append(MSE_test)

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
				MSE_train = model.get_MSE(sess, prediction, obs, obs_train)
				MSE_test  = model.get_MSE(sess, prediction, obs, obs_test)
				print("iter {:>3}, train loss: {:>7.3f}, test loss: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}"\
					.format(i+1, loss_train, loss_test, MSE_train, MSE_test))

				loss_trains.append(loss_train)
				loss_tests.append(loss_test)
				MSE_trains.append(MSE_train)
				MSE_tests.append(MSE_test)

			if store_res == True and (i+1)%save_freq == 0:
				if not os.path.exists(RLT_DIR+"model"): os.makedirs(RLT_DIR+"model")
				saver.save(sess, RLT_DIR+"model/model_epoch", global_step=i+1)

		print("finish training")


		#hidden_train(generated hidden variable) compare with output[2] (predicted hidden variable, Batch size * T * Dz)

		if store_res and not use_stock_data:
			hidden_val = np.zeros((saving_num, time, n_particles, Dz))
			for i in range(0, saving_num, batch_size):
				hidden_val[i:i+batch_size] = sess.run(hidden, feed_dict={obs:obs_train[i:i+batch_size]})
			plot_hidden(RLT_DIR, np.mean(hidden_val, axis = 2), hidden_train[0:saving_num], is_test = False)
			plot_hidden_2d(RLT_DIR, np.mean(hidden_val, axis = 2), is_test = False)
			#plot_hidden_3d(RLT_DIR, np.mean(hidden_val, axis = 2), is_test = False)

			for i in range(0, saving_num, batch_size):
				hidden_val[i:i+batch_size] = sess.run(hidden, feed_dict={obs:obs_test[i:i+batch_size]})
			plot_hidden(RLT_DIR, np.mean(hidden_val, axis = 2), hidden_test[0:saving_num], is_test = True)
			plot_hidden_2d(RLT_DIR, np.mean(hidden_val, axis = 2), is_test = True)
			#plot_hidden_3d(RLT_DIR, np.mean(hidden_val, axis = 2), is_test = True)

		if store_res:
			prediction_val = np.zeros((saving_num, time, n_particles, Dx))
			for i in range(0, saving_num, batch_size):
				prediction_val[i:i+batch_size] = sess.run(prediction, feed_dict={obs:obs_train[i:i+batch_size]})
			plot_expression(RLT_DIR, np.mean(prediction_val, axis = 2), obs_train[0:saving_num], is_test = False)

			for i in range(0, saving_num, batch_size):
				prediction_val[i:i+batch_size] = sess.run(prediction, feed_dict={obs:obs_test[i:i+batch_size]})
			plot_expression(RLT_DIR, np.mean(prediction_val, axis = 2), obs_test[0:saving_num], is_test = True)

		if store_res and not use_stock_data:
			plot_training(RLT_DIR, hidden_train[0:saving_num], obs_train[0:saving_num], is_test = False)
			plot_training(RLT_DIR, hidden_test[0:saving_num], obs_test[0:saving_num], is_test = True)
			# plot_training_3d(RLT_DIR, hidden_train[0:saving_num], obs_train[0:saving_num], is_test = False)
			# plot_training_3d(RLT_DIR, hidden_test[0:saving_num], obs_test[0:saving_num], is_test = True)

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
		MSE_dict = {"MSE_trains":MSE_trains,
					"MSE_tests":MSE_tests}
		data_dict = {"params_dict":params_dict,
					 "loss_dict":loss_dict,
					 "MSE_dict":MSE_dict}
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)

		if use_stock_data:
			learned_val_dict = {"prediction":prediction_val}
		else:
			learned_val_dict = {"hidden_val":hidden_val,
								"prediction":prediction_val}
		data_dict["learned_val_dict"] = learned_val_dict
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)

		plot_loss(RLT_DIR, loss_trains, loss_tests)
		plot_MSE(RLT_DIR, MSE_trains, MSE_tests)