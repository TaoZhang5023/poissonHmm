import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import utils

N_ITER = 500
N_SPLIT = 5
N_FEATURE = 1
LABEL = "interval"
OBS_FILE = []
LENGTH_FILE = []
MODEL_NAME = "../"+LABEL+"_model/model_"
N_COMPONENTS = 5
# TYPE = ["mbs", "pbs"]
# SUGGEST =    [30,90]
# THREADHOLD = [90,120]
TYPE = ["pbs"]
SUGGEST =    [30]
THREADHOLD = [90]

def init_lamdas(train_data, suggest):
    lamdas_ = np.zeros((1,N_COMPONENTS))
    lamdas_[0,0] = np.amin(train_data)
    lamdas_[0,1] = np.percentile(train_data, [25])
    lamdas_[0,2] = np.mean(train_data)
    lamdas_[0,3] = np.percentile(train_data, [75])
    lamdas_[0,4] = np.amax(train_data)
    print("min: ",lamdas_[0,0], " mean: ", np.mean(train_data), " max: ", lamdas_[0,2])
    return lamdas_

def train_model(train_data, train_length, lamdas_):
    model = utils.train_model(train_data, train_length, N_COMPONENTS, N_ITER, lamdas_)
    return model

def validate_model(model, test_data, test_length):
    pred_states = model.predict(test_data, test_length)
    # utils.draw_result(pred_states[0:500]*30, test_data[0:500])
    model._print_info()


def save_model(model, filename):
    joblib.dump(model, filename)

def train_and_test_one_model(obs, lengths, col_index):
    obs = utils.preprocess_data(obs, THREADHOLD[col_index])
    lamdas_ = init_lamdas(obs, SUGGEST[col_index])
    for i in range(0, len(lengths)):
        start_index = 0
        print("the ", i , "th round.")
        # Calculate the mask
        for j in range(0, i):
            start_index += lengths[j].sum()
        end_index = start_index + lengths[i].sum()
        test_data, train_data = utils.split_data(obs, start_index, end_index)
        test_length, train_length = utils.split_length(lengths, i)
        print(lamdas_)
        model = train_model(train_data, train_length, lamdas_)
        model_name = MODEL_NAME+TYPE[col_index]+".pkl"
        save_model(model, model_name)
        validate_model(model, test_data, test_length)
        break
    print("end")

def train_and_test_models(obs_file, length_file, n_split=N_SPLIT):
    # Read data
    obses, lengths = utils.read_files(obs_file, length_file)
    print(obs_file, length_file)
    assert len(obses) == len(lengths)
    for i in range(len(obses)):
        obs = obses[i]
        length = lengths[i][:,1:].flatten()
        print("There are ", obs.shape[0], " observations and ", length.shape, " patients.")
        length = np.array_split(length, n_split)
        train_and_test_one_model(obs, length, i)

def test_model(obs_file=OBS_FILE, length_file=LENGTH_FILE):
    obs, length = utils.read_files(obs_file, length_file)
    obs = utils.preprocess_data(obs)
    # model = joblib.load(MODEL_NAME+str(0)+".pkl")
    model = joblib.load("../interval_model/model_mbs.pkl")
    model._print_info()
    validate_model(model, obs[:,0].reshape(-1,1), length)

def run():
    OBS_FILE, LENGTH_FILE = utils.set_filename(LABEL, TYPE)
    print("run")
    print(OBS_FILE)
    print(LENGTH_FILE)
    train_and_test_models(OBS_FILE, LENGTH_FILE)

if __name__ =="__main__":
    run()
# test_model()
# set_filename()
