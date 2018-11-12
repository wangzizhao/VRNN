import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf

# import from files
from model import VRNNCell, VRNN_model
from data import get_stock_data
from result_saving import *

# for data saving stuff
import sys
import pickle
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":

	# ============================================ parameter part ============================================ #
	# training hyperparameters
	time = 10

	batch_size = 16
	lr = 1e-4
	epoch = 100

	n_train = 5	* batch_size
	n_test  = 1 * batch_size

	seed = 0
	tf.set_random_seed(seed)
	np.random.seed(seed)

	# model hyperparameters
	Dx = 6
	Dh = 50
	Dz = 100
	x_ft_Dhs = [100]
	z_ft_Dhs = [100]
	prior_Dhs = [100]
	enc_Dhs = [100]
	decoder_Dhs = [100]

	initial_state_trainable = False,
	sigma_min = 1e-9

	# printing and data saving params
	print_freq = 1

	store_res = True
	save_freq = 10
	saving_num = min(n_train, 1*batch_size)
	rslt_dir_name = "VRNN"

	# ============================================= dataset part ============================================= #
	stock_idx_name = "dow_jones" # or "nasdaq" or "sp_500"
	obs_train, obs_test = get_stock_data(stock_idx_name, time, n_train, n_test)

	# ============================================== model part ============================================== #
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name = "obs")

	myVRNNCell = VRNNCell(Dx, Dh, Dz,
						  x_ft_Dhs, z_ft_Dhs,
						  prior_Dhs, enc_Dhs, decoder_Dhs)
	model = VRNN_model(myVRNNCell, batch_size,
					   initial_state_trainable = initial_state_trainable,
					   sigma_min = sigma_min)

	output, last_state = model.get_output(obs)
	loss = model.get_loss(obs, output)
	prediction = model.get_prediction(output)

	with tf.name_scope("train"):
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
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

		if store_res:
			prediction_val = np.zeros((saving_num, time, Dx))
			for i in range(0, saving_num, batch_size):
				prediction_val[i:i+batch_size] = sess.run(prediction, feed_dict={obs:obs_train[i:i+batch_size]})

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