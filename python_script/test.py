import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

N_ITER = 500
N_SPLIT = 10
N_FEATURE = 8
LABEL = "6mh_cate2"
OBS_FILE = "../data/obs_" + LABEL + ".csv"
LENGTH_FILE = "../data/length_" + LABEL + ".csv"
MODEL_NAME = "../model/model_" + LABEL


def read_files(obs_file, length_file):
    length = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,3:]
    return obs,length

def load_model(filename):
    return joblib.load(filename)

def test_n(model_name, result, index):
    model = load_model(model_name)
    model._print_info()
    obs, lengths = read_files(OBS_FILE, LENGTH_FILE)
    start_index = 0
    total_length = lengths.sum()
    predict_event = []
    obs_event = []
    for i in range(0, len(lengths)):
        for j in range(1,lengths[i]):
            prob = model.predict_n_hospital(obs[start_index:start_index+j, :])
            if(start_index+j < total_length):
                obs_event.append(obs[start_index+j,1])
                predict_event.append(prob[1, np.argmax(prob,axis=1)[0]])
        start_index += lengths[i]
    plt.plot(obs_event)
    plt.plot(predict_event)
    plt.show()
    # correct = 0;
    # n_event = 0;
    # for i in range(0,len(obs_event)):
    #     if (obs_event[i] > 0):
    #         n_event += 1
    #         if (predict_event[i] > 0.3):
    #             correct += 1
    # result[index] = correct/n_event
    # print(result[index])


def test(model_name, result, index):
    model = load_model(model_name)
    # model._print_info()
    obs, lengths = read_files(OBS_FILE, LENGTH_FILE)
    prob = []
    start_index = 0
    total_length = lengths.sum()
    predict_event = []
    obs_event = []
    for i in range(0, len(lengths)):
        for j in range(1,lengths[i]):
            prob = model.predict_hospital(obs[start_index:start_index+j, :], 15)
            if(start_index+j < total_length):
                obs_event.append(obs[start_index+j,15])
                predict_event.append(prob)
        start_index += lengths[i]
    HH = 0;
    HN = 0;
    NN = 0;
    NH = 0;
    H = 0;
    N = 0;
    n_event = 0;
    threshold = 0.2
    for i in range(0,len(obs_event)):
        if (obs_event[i] > 0):
            H += 1
            if (predict_event[i] > threshold):
                HH += 1
            else:
                HN += 1
        else:
            N += 1
            if (predict_event[i] > threshold):
                NH += 1
            else:
                NN += 1
    print("there is ", len(obs_event), " records")
    print(HH/H, " percent of hospital is detected as hospital")
    print(HN/H, " percent of hospital is detected as non-hospital")
    print(NH/N, " percent of non-hospital is detected as hospital")
    print(NN/N, " percent of non-hospital is detected as non-hospital")
    plt.plot(obs_event)
    plt.plot(predict_event)
    plt.show()

def show_state(model_name, result):
    model = load_model(model_name)
    obs, lengths = read_files(OBS_FILE, LENGTH_FILE)
    predict = model.predict(obs, lengths)
    plt.plot(predict[0:2000],'b.')
    plt.show()


# result = np.zeros(10)
# for i in range(0,10):
#     model_name = MODEL_NAME+str(i)+".pkl"
#     test(model_name, result, i)
# for i in range(0,10):
#     print("testing model ", i)
#     print("correctness: ", result[i])
result = np.zeros(10)
i = 99
model_name = "../model/model_6mh_cate25_1.pkl"
# test(model_name, result, i)
model = load_model(model_name)
model._print_info()
# show_state(model_name, result)
