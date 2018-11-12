import numpy as np
import matplotlib.pyplot as plt
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

    if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)

    return RLT_DIR

def plot_loss(RLT_DIR, loss_trains, loss_tests):
    plt.figure()
    plt.plot(loss_trains)
    plt.plot(loss_tests)
    plt.legend(['loss_trains', 'loss_tests'])
    sns.despine()
    plt.savefig(RLT_DIR + "loss")
    plt.show()


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