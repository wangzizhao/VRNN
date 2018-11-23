import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import datetime

import json
import os

def addDateTime():
    """
    Collection of little pythonic tools. Might need to organize this better in the future.

    @author: danielhernandez
    """
    date = str(datetime.datetime.now())
    date = date[2:4] + date[5:7] + date[8:10] + "_" + date[11:13] + date[14:16]
    return "D" + date

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_RLT_DIR(Experiment_params):
    # create the dir to save data
    # Experiment_params is a dict containing param_name&param pair
    # Experiment_params must contain "rslt_dir_name":rslt_dir_name
    cur_date = addDateTime()

    local_rlt_root = "results/" + Experiment_params["rslt_dir_name"] + "/"

    params_str = ""
    for param_name, param in Experiment_params.items():
        if param_name == "rslt_dir_name":
            continue
        params_str += "_" + param_name + "_" + str(param)

    RLT_DIR = os.getcwd().replace("\\", "/") + "/" + (local_rlt_root + cur_date + params_str) + "/"

    if not os.path.exists(RLT_DIR): 
        os.makedirs(RLT_DIR)

    return RLT_DIR

def plot_loss(RLT_DIR, loss_trains, loss_tests):
    plt.figure()
    plt.plot(loss_trains)
    plt.plot(loss_tests)
    plt.legend(["loss_trains", "loss_tests"])
    sns.despine()
    plt.savefig(RLT_DIR + "loss")
    plt.show()


def plot_MSE(RLT_DIR, MSE_trains, MSE_tests):
    plt.figure()
    plt.plot(MSE_trains)
    plt.plot(MSE_tests)
    plt.legend(["MSE_trains", "MSE_tests"])
    sns.despine()
    plt.savefig(RLT_DIR + "MSE")
    plt.show()


def plot_hidden(RLT_DIR, predicted_hidden, true_hidden, is_test):
    PLT_DIR = "hidden_compare_" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(true_hidden.shape[0]):
        for j in range(true_hidden.shape[-1]):
            plt.figure()
            plt.plot(true_hidden[i, :, j])
            plt.plot(predicted_hidden[i, :, j])
            plt.legend(["true_hidden", "predicted_hidden"])
            sns.despine()
            plt.savefig(RLT_DIR + PLT_DIR + "hidden_dim_{}_idx_{}".format(j, i))
            plt.close()
            #plt.show()

def plot_expression(RLT_DIR, predicted_obs, true_obs, is_test):
    PLT_DIR = "obs_compare_" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(true_obs.shape[0]):
        for j in range(true_obs.shape[-1]):
            plt.figure()
            plt.plot(true_obs[i, :, j])
            plt.plot(predicted_obs[i, :, j])
            plt.legend(["true_obs", "predicted_obs"])
            sns.despine()
            plt.savefig(RLT_DIR + PLT_DIR + "obs_dim_{}_idx_{}".format(j, i))
            plt.close()
            #plt.show()

def plot_hidden_2d(RLT_DIR, predicted_hidden, is_test):
    PLT_DIR = "hidden_result_2d" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(predicted_hidden.shape[0]):
        plt.figure()
        plt.plot(predicted_hidden[i,:,0], predicted_hidden[i,:,1])
        plt.legend(["predicted_hidden"])
        sns.despine()
        plt.savefig(RLT_DIR + PLT_DIR + "hidden_idx_{}".format(i))
        plt.close()

def plot_hidden_3d(RLT_DIR, predicted_hidden, is_test):
    PLT_DIR = "hidden_result_3d" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(predicted_hidden.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(predicted_hidden[i,:,0], predicted_hidden[i,:,1], predicted_hidden[i,:,2])
        ax.legend(["predicted_hidden"])
        sns.despine()
        fig.savefig(RLT_DIR + PLT_DIR + "hidden_idx_{}".format(i))
        plt.close()

def plot_training_2d(RLT_DIR, true_hidden, true_obs, is_test):
    PLT_DIR = "training_2d_" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(true_hidden.shape[0]):
        plt.figure()
        plt.plot(true_hidden[i,:,0])
        plt.plot(true_hidden[i,:,1])
        plt.plot(true_obs[i,:,0])
        plt.legend(["true_hidden_dim_1", "true_hidden_dim_2", "true_obs"])
        sns.despine()
        plt.savefig(RLT_DIR + PLT_DIR + "training_idx_{}".format(i))
        plt.close()

def plot_training_3d(RLT_DIR, true_hidden, true_obs, is_test):
    PLT_DIR = "training_3d_" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(true_hidden.shape[0]):
        plt.figure()
        plt.plot(true_hidden[i,:,0])
        plt.plot(true_hidden[i,:,1])
        plt.plot(true_hidden[i,:,2])
        plt.plot(true_obs[i,:,0])
        plt.legend(["true_hidden_dim_1", "true_hidden_dim_2", "true_hidden_dim_3", "true_obs"])
        sns.despine()
        plt.savefig(RLT_DIR + PLT_DIR + "training_idx_{}".format(i))
        plt.close()

def plot_training(RLT_DIR, true_hidden, true_obs, is_test):
    PLT_DIR = "training_" + ("test" if is_test else "train") + "/"
    if not os.path.exists(RLT_DIR + PLT_DIR): os.makedirs(RLT_DIR + PLT_DIR)
    for i in range(true_hidden.shape[0]):
        plt.figure()
        legend = []
        for j in range(true_hidden.shape[-1]):
            plt.plot(true_hidden[i,:,j])
            legend.append("true_hidden_dim_" + str(j))
        for j in range(true_obs.shape[-1]):
            plt.plot(true_obs[i,:,j])
            legend.append("true_obs_dim_" + str(j))
        plt.legend(legend)
        sns.despine()
        plt.savefig(RLT_DIR + PLT_DIR + "training_idx_{}".format(i))
        plt.close()