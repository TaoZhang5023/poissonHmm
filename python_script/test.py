import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

N_COMPONENTS = 5
N_ITER = 500
N_SPLIT = 10
N_FEATURE = 2
OBS_FILE = "../data/obs_p.csv"
LENGTH_FILE = "../data/length_p.csv"


def read_files(obs_file, length_file):
    length = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,[3,4]]
    return obs,length

def load_model(filename):
    return joblib.load(filename)


def test(model_name, result, index):
    model = load_model(model_name)
    model._print_info()
    obs, lengths = read_files(OBS_FILE, LENGTH_FILE)
    prob = []
    start_index = 0
    total_length = lengths.sum()
    predict_event = []
    obs_event = []
    for i in range(0, len(lengths)):
        for j in range(1,lengths[i]):
            prob = model.predict_hospital(obs[start_index:start_index+j, :])
            if(start_index+j < total_length):
                obs_event.append(obs[start_index+j,1])
                predict_event.append(prob)
        start_index += lengths[i]
    #
    # predict_event = predict_event[17500:18500]
    # obs_event = obs_event[17500:18500]
    correct = 0;
    for i in range(0,len(obs_event)):
        if ((obs_event[i] > 0) & (predict_event[i] > 0.3)) or ((obs_event[i] == 0) & (predict_event[i] < 0.1)):
            correct += 1
    result[index] = correct/len(obs_event)

# plt.plot(obs_event)
# plt.plot(predict_event)
# plt.show()
result = np.zeros(10)
for i in range(0,10):
    model_name = "../model/model_p"+str(i)+".pkl"
    test(model_name, result, i)
for i in range(0,10):
    print("testing model ", i)
    print("correctness: ", result[i])
