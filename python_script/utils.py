import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def preprocess_data(obs, thred):
    # Update the number of events only
    obs[:,0][obs[:,0] > thred] = thred
    return obs

def set_filename(label, cate):
    obs_files = []
    length_files = []
    for t in cate:
        obs = "../data_"+label+"/"+t+"_predict.csv"
        length = "../data_"+label+"/"+t+"_length_predict.csv"
        obs_files.append(obs)
        length_files.append(length)
    return obs_files, length_files

def read_files(obs_files, length_files):
    assert len(obs_files) == len(length_files)
    obses = []
    lengths = []
    for i in range(len(obs_files)):
        obs = pd.read_csv(obs_files[i], sep=',', header=0).values[:,2:]
        length = pd.read_csv(length_files[i], sep=',', header=0).values[:,1:]
        obses.append(obs)
        lengths.append(length)
    return obses,lengths

def split_data(obs, start_index, end_index):
    print("test data: ", start_index, end_index)
    mask = np.zeros(obs.shape[0], dtype=bool)
    mask[start_index:end_index] = True
    test_data = obs[mask,:]
    train_data = obs[~mask,:]
    return test_data, train_data

def split_length(lengths, i):
    test_length = lengths[i]
    new = np.delete(lengths, i, 0)
    train_length = [item for sublist in new for item in sublist]
    return test_length, np.asarray(train_length)

def train_model(train_data, train_length, n_components, n_iter, lamdas_):
    model = poissonHMM.poissonHMM(n_components=n_components, n_iter=n_iter, lamdas_=lamdas_)
    model.fit(train_data, train_length)
    return model

def draw_result(predict_state, test_data):
    plt.rcParams['figure.figsize'] = (15, 5)
    plt.title("prediction")
    plt.bar(np.arange(len(test_data))+0.1,test_data.flatten(), width=0.3,label="Test Data");
    plt.bar(np.arange(len(predict_state))-0.1,predict_state.flatten(), width=0.3, label="Predict State");
    plt.tight_layout()
    plt.legend()
    plt.show()
