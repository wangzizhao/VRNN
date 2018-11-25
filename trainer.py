import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf

# for data saving stuff
import os

# define trainer to train the model given as parameter


class Trainer:

    def __init__(self, model, obs):
        self.model = model

        self.batch_size = model.batch_size
        self.n_particles = model.n_particles

        self.Dx = model.VRNNCell.Dx
        self.Dz = model.VRNNCell.Dz

        self.obs = obs

        self.store_res = False

    def set_result_saving(self, RLT_DIR, save_freq, saving_num):
        self.store_res = True
        self.RLT_DIR = RLT_DIR
        self.save_freq = save_freq
        self.saving_num = saving_num
        self.writer = tf.summary.FileWriter(RLT_DIR)

    def set_data_set(self, hidden_train, obs_train, hidden_test, obs_test):
        self.hidden_train = hidden_train
        self.hidden_test = hidden_test
        self.obs_train = obs_train
        self.obs_test = obs_test
        self.time = hidden_train.shape[1]

    def train(self, lr, epoch, print_freq):
        self.lr = lr
        self.epoch = epoch
        self.print_freq = print_freq

        output, last_state = self.model.get_output(self.obs)
        loss = self.model.get_loss(self.obs, output)
        hidden = self.model.get_hidden(output)
        prediction = self.model.get_prediction(output)

        loss_trains = []
        loss_tests = []
        MSE_trains = []
        MSE_tests = []

        with tf.name_scope("train"):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(-loss)
        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(init)

            if self.store_res:
                self.writer.add_graph(sess.graph)

            loss_train = self.model.get_loss_val(
                sess, loss, self.obs, self.obs_train)
            loss_test = self.model.get_loss_val(
                sess, loss, self.obs, self.obs_test)
            MSE_train = self.model.get_MSE(
                sess, prediction, self.obs, self.obs_train)
            MSE_test = self.model.get_MSE(
                sess, prediction, self.obs, self.obs_test)
            print(
                "iter {:>3}, train loss: {:>7.3f}, test loss: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}" .format(
                    0,
                    loss_train,
                    loss_test,
                    MSE_train,
                    MSE_test))

            loss_trains.append(loss_train)
            loss_tests.append(loss_test)
            MSE_trains.append(MSE_train)
            MSE_tests.append(MSE_test)

            for i in range(self.epoch):
                if self.hidden_train is not None:
                    self.hidden_train, self.obs_train = shuffle(
                        self.hidden_train, self.obs_train)
                else:
                    self.obs_train = shuffle(self.obs_train)
                for j in range(0, len(self.obs_train), self.batch_size):
                    sess.run(train_op, feed_dict={
                             self.obs: self.obs_train[j:j + self.batch_size]})

                # print training and testing loss
                if (i + 1) % self.print_freq == 0:
                    loss_train = self.model.get_loss_val(
                        sess, loss, self.obs, self.obs_train)
                    loss_test = self.model.get_loss_val(
                        sess, loss, self.obs, self.obs_test)
                    MSE_train = self.model.get_MSE(
                        sess, prediction, self.obs, self.obs_train)
                    MSE_test = self.model.get_MSE(
                        sess, prediction, self.obs, self.obs_test)
                    print(
                        "iter {:>3}, train loss: {:>7.3f}, test loss: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}" .format(
                            i +
                            1,
                            loss_train,
                            loss_test,
                            MSE_train,
                            MSE_test))

                    loss_trains.append(loss_train)
                    loss_tests.append(loss_test)
                    MSE_trains.append(MSE_train)
                    MSE_tests.append(MSE_test)

                if self.store_res and (i + 1) % self.save_freq == 0:
                    if not os.path.exists(self.RLT_DIR + "model"):
                        os.makedirs(self.RLT_DIR + "model")
                    saver.save(
                        sess,
                        self.RLT_DIR +
                        "model/model_epoch",
                        global_step=i +
                        1)

            print("finish training")

            hidden_val_train = np.zeros(
                (self.saving_num, self.time, self.n_particles, self.Dz))
            for i in range(0, self.saving_num, self.batch_size):
                hidden_val_train[i:i + self.batch_size] = sess.run(
                    hidden, feed_dict={self.obs: self.obs_train[i:i + self.batch_size]})

            hidden_val_test = np.zeros(
                (self.saving_num, self.time, self.n_particles, self.Dz))
            for i in range(0, self.saving_num, self.batch_size):
                hidden_val_test[i:i + self.batch_size] = sess.run(
                    hidden, feed_dict={self.obs: self.obs_test[i:i + self.batch_size]})

            prediction_val_train = np.zeros(
                (self.saving_num, self.time, self.n_particles, self.Dx))
            for i in range(0, self.saving_num, self.batch_size):
                prediction_val_train[i:i + self.batch_size] = sess.run(
                    prediction, feed_dict={self.obs: self.obs_train[i:i + self.batch_size]})

            prediction_val_test = np.zeros(
                (self.saving_num, self.time, self.n_particles, self.Dx))
            for i in range(0, self.saving_num, self.batch_size):
                prediction_val_test[i:i + self.batch_size] = sess.run(
                    prediction, feed_dict={self.obs: self.obs_test[i:i + self.batch_size]})

            metrics = [loss_trains, loss_tests, MSE_trains, MSE_tests]
            hidden_val = [hidden_val_train, hidden_val_test]
            prediction_val = [prediction_val_train, prediction_val_test]
            return metrics, hidden_val, prediction_val
