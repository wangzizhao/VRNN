import numpy as np

import tensorflow as tf

# import from files
from model import VRNNCell, VRNN_model
from data import get_stock_data
from result_saving import NumpyEncoder, create_RLT_DIR
from result_saving import plot_loss, plot_MSE, plot_loss_MSE
from result_saving import plot_hidden, plot_expression
from result_saving import plot_hidden_2d, plot_hidden_3d
from result_saving import plot_training_2d, plot_training_3d
from result_saving import plot_training
from sampler import create_train_test_dataset
from transformation import fhn, linear, lorenz
from distribution import dirac_delta, mvn
from trainer import Trainer

# for data saving stuff
import pickle
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

if __name__ == "__main__":

    # =========================== parameter part =========================== #
    # training hyperparameters
    time = 100

    n_particles = 100
    batch_size = 16

    lr = 1e-4
    epoch = 100

    n_train = 50 * batch_size
    n_test = 1 * batch_size

    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)

    use_stock_data = False

    # model hyperparameters
    if use_stock_data:
        Dx = 6
        Dh = 100
        Dz = 100
        x_ft_Dhs = [100, 100]
        z_ft_Dhs = [100, 100]
        prior_Dhs = [100, 100, 100]
        enc_Dhs = [100, 100, 100]
        decoder_Dhs = [100, 100, 100]
    else:
        Dx = 2
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

    print_freq = 1

    store_res = True
    save_freq = 10
    saving_num = min([n_train, n_test, 1 * batch_size])

    rslt_dir_name = "linear_linear"

    # =========================== dataset part =========================== #
    if use_stock_data:
        stock_idx_name = "dow_jones"  # or "nasdaq" or "sp_500"
        obs_train, obs_test = get_stock_data(
            stock_idx_name, time, n_train, n_test, Dx)
        hidden_train = hidden_test = None
    else:

        # define transition f and emission g

        # f:fhn, g:linear
        # Dz:2, Dx:10
        fhn_params = (1.0, 0.95, 0.05, 1.0, 0.15)
        f_sample_tran = fhn.fhn_transformation(fhn_params)
        f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

        linear_params = np.array([[0, 1]])
        g_sample_tran = linear.linear_transformation(linear_params)
        mvn_sigma = 0.2 * np.eye(1)
        g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)

        # f:lorez, g:linear
        # Dz:3, Dx:10
        lorenz_params = (10.0, 28.0, 8.0 / 3.0, 0.01)
        f_sample_tran = lorenz.lorenz_transformation(lorenz_params)
        f_sample_dist = dirac_delta.dirac_delta(f_sample_tran)

        linear_params = np.random.randn(1, 3)
        g_sample_tran = linear.linear_transformation(linear_params)
        mvn_sigma = 0.2 * np.eye(1)
        g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)
        g_sample_tran = linear.linear_transformation(linear_params)

        # f:linear, mvn; g:linear, mvn
        f_linear_params = np.array([[0.99, 0.01], [0.2, 0.8]])
        f_sample_tran = linear.linear_transformation(f_linear_params)
        mvn_sigma = np.eye(2)
        f_sample_dist = mvn.mvn(f_sample_tran, mvn_sigma)

        g_linear_params = np.array([[2, 1], [1, 2]])
        g_sample_tran = linear.linear_transformation(g_linear_params)
        mvn_sigma = np.eye(2)
        g_sample_dist = mvn.mvn(g_sample_tran, mvn_sigma)

        dataset = create_train_test_dataset(
            n_train,
            n_test,
            time,
            Dz,
            Dx,
            f_sample_dist,
            g_sample_dist,
            None,
            -3,
            3)
        hidden_train, obs_train, hidden_test, obs_test = dataset

    # =========================== model part =========================== #
    obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name="obs")

    myVRNNCell = VRNNCell(Dx, Dh, Dz,
                          batch_size,
                          n_particles,
                          x_ft_Dhs, z_ft_Dhs,
                          prior_Dhs, enc_Dhs, decoder_Dhs)
    model = VRNN_model(myVRNNCell,
                       batch_size,
                       n_particles,
                       initial_state_all_zero=initial_state_all_zero,
                       is_lstm_Dh=is_lstm_Dh,
                       sigma_min=sigma_min)

    # ====================== data saving part ====================== #
    if store_res:
        experiment_params = {"time": time,
                             "n_particles": n_particles,
                             "batch_size": batch_size,
                             "lr": lr,
                             "epoch": epoch,
                             "seed": seed,
                             "n_train": n_train,
                             "initial_state_all_zero": initial_state_all_zero,
                             "rslt_dir_name": rslt_dir_name}

        print("experiment_params")
        for key, val in experiment_params.items():
            print("\t{}:{}".format(key, val))

        RLT_DIR = create_RLT_DIR(experiment_params)

    # =========================== training part =========================== #
    trainer = Trainer(model, obs)
    trainer.set_result_saving(RLT_DIR, save_freq, saving_num)
    trainer.set_data_set(hidden_train, obs_train, hidden_test, obs_test)
    metrics, hidden_val, prediction_val = trainer.train()

    loss_trains, loss_tests, MSE_trains, MSE_tests = metrics
    hidden_val_train, hidden_val_test = hidden_val
    prediction_val_train, prediction_val_test = prediction_val

    # ==================== anther data saving part ==================== #
    if store_res and not use_stock_data:
        plot_hidden(RLT_DIR, np.mean(hidden_val_train, axis=2),
                    hidden_train[0:saving_num], is_test=False)
        plot_hidden(RLT_DIR, np.mean(hidden_val_test, axis=2),
                    hidden_test[0:saving_num], is_test=True)
        if isinstance(f_sample_tran, fhn.fhn_transformation):
            plot_hidden_2d(
                RLT_DIR,
                np.mean(
                    hidden_val_train,
                    axis=2),
                is_test=False)
            plot_hidden_2d(
                RLT_DIR,
                np.mean(
                    hidden_val_test,
                    axis=2),
                is_test=True)
        if isinstance(f_sample_tran, lorenz.lorenz_transformation):
            plot_hidden_3d(
                RLT_DIR,
                np.mean(
                    hidden_val_train,
                    axis=2),
                is_test=False)
            plot_hidden_3d(
                RLT_DIR,
                np.mean(
                    hidden_val_test,
                    axis=2),
                is_test=True)

    if store_res:
        plot_expression(RLT_DIR,
                        np.mean(
                            prediction_val_train,
                            axis=2),
                        obs_train[0:saving_num],
                        is_test=False)
        plot_expression(RLT_DIR,
                        np.mean(
                            prediction_val_test,
                            axis=2),
                        obs_test[0:saving_num],
                        is_test=True)

    if store_res and not use_stock_data:
        plot_training(RLT_DIR,
                      hidden_train[0:saving_num],
                      obs_train[0:saving_num],
                      is_test=False)
        plot_training(RLT_DIR,
                      hidden_test[0:saving_num],
                      obs_test[0:saving_num],
                      is_test=True)
        if isinstance(f_sample_tran, fhn.fhn_transformation):
            plot_training_2d(
                RLT_DIR,
                hidden_train[0:saving_num],
                is_test=False)
            plot_training_2d(
                RLT_DIR,
                hidden_test[0:saving_num],
                is_test=True)
        if isinstance(f_sample_tran, lorenz.lorenz_transformation):
            plot_training_3d(
                RLT_DIR,
                hidden_train[0:saving_num],
                is_test=False)
            plot_training_3d(
                RLT_DIR,
                hidden_test[0:saving_num],
                is_test=True)

    if store_res:
        params_dict = {"time": time,
                       "batch_size": batch_size,
                       "lr": lr,
                       "epoch": epoch,
                       "n_train": n_train,
                       "seed": seed}
        loss_dict = {"loss_trains": loss_trains,
                     "loss_tests": loss_tests}
        MSE_dict = {"MSE_trains": MSE_trains,
                    "MSE_tests": MSE_tests}
        data_dict = {"params_dict": params_dict,
                     "loss_dict": loss_dict,
                     "MSE_dict": MSE_dict}
        with open(RLT_DIR + 'data.json', 'w') as f:
            json.dump(data_dict, f, indent=4, cls=NumpyEncoder)

        if use_stock_data:
            learned_val_dict = {"prediction_train": prediction_val_train,
                                "prediction_test": prediction_val_test}
            true_val_dict = {"obs_train": obs_train[0:saving_num],
                             "obs_test": obs_test[0:saving_num]}
        else:
            learned_val_dict = {"hidden_train": hidden_val_train,
                                "hidden_test": hidden_val_test,
                                "prediction_train": prediction_val_train,
                                "prediction_test": prediction_val_test}
            true_val_dict = {"obs_train": obs_train[0:saving_num],
                             "obs_test": obs_test[0:saving_num],
                             "hidden_train": hidden_train[0:saving_num],
                             "hidden_test": hidden_test[0:saving_num]}

        data_dict["learned_val_dict"] = learned_val_dict
        data_dict["true_val_dict"] = true_val_dict

        with open(RLT_DIR + 'data.p', 'wb') as f:
            pickle.dump(data_dict, f)

        plot_loss(RLT_DIR, loss_trains, loss_tests)
        plot_MSE(RLT_DIR, MSE_trains, MSE_tests)
        plot_loss_MSE(RLT_DIR, loss_trains, loss_tests, MSE_trains, MSE_tests)