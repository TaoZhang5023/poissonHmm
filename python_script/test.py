import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

N_COMPONENTS = 5
N_ITER = 500
N_SPLIT = 10
N_FEATURE = 3
OBS_FILE = "../data/obs_halfyear.csv"
LENGTH_FILE = "../data/length_halfyear.csv"


def read_files(obs_file, length_file):
    length = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,[3,4,5]]
    return obs,length

def load_model(filename):
    return joblib.load(filename)

model = load_model("../model/model_halfyear1.pkl")
model._print_info()
obs, lengths = read_files(OBS_FILE, LENGTH_FILE)
prob = []
start_index = 0
total_length = lengths.sum()
predict_event = []
obs_event = []
for i in range(0, 400):
    print("new Patient: ")
    for j in range(1,lengths[i]):
        prob = model.predict_hospital(obs[start_index:start_index+j, :])
        if(start_index+j+1 < total_length):
            obs_event.append(obs[start_index+j+1,2])
            predict_event.append(prob)
    start_index += lengths[i]
print("end")

predict_event = predict_event[450:650]
obs_event = obs_event[450:650]
plt.plot(obs_event)
plt.plot(predict_event)
plt.show()
