import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib

N_COMPONENTS = 5
N_ITER = 2000
OBS_FILE = "data/obsv1.csv"
LENGTH_FILE = "data/lengthv1.csv"

def split_data(obs, start_index, end_index):
    print("test data: ", start_index, end_index)
    mask = np.zeros(obs.shape[0], dtype=bool)
    mask[start_index:end_index] = True
    test_data = obs[mask,:]
    train_data = obs[~mask,:]
    return test_data, train_data

def validation(train_data, test_data, lengths, i):
    b=train_data > 28
    train_data[b] = 28
    b = test_data > 28
    test_data[b] = 28
    model = poissonHMM.poissonHMM(n_components=N_COMPONENTS, n_iter=N_ITER)
    model.fit(train_data, lengths)
    model._print_info()
    filename = "model/model"+str(i)+".pkl"
    save_model(model, filename)

def save_model(model, filename):
    joblib.dump(model, filename)

def read_files(obs_file, length_file):
    length = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,[3,4,5]]
    return obs,length

def test_model(obs_file=OBS_FILE, length_file=LENGTH_FILE, n_split=5):
    obs, length = read_files(obs_file, length_file)
    print("There are ", obs.shape[0], " observations and ", len(length), " patients.")
    lengths = np.array_split(length, n_split)
    for i in range(0, len(lengths)):
        start_index = 0
        print("the ", i , "th round.")
        # Calculate the mask
        for j in range(0, i):
            start_index += lengths[j].sum()
        end_index = start_index + lengths[i].sum()
        test_data, train_data = split_data(obs, start_index, end_index)
        validation(train_data, test_data, lengths[i], i)
    print("end")

test_model()
