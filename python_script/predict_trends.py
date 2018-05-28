import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from scipy.stats import levene

import train_model
import utils
import patient as Patient

N_COMPONENTS = train_model.N_COMPONENTS
TYPE = train_model.TYPE
LABEL = train_model.LABEL
THREADHOLD = train_model.THREADHOLD
MODEL_NAME = train_model.MODEL_NAME
TREND_TYPE = ["depress"]
USE_PROPORTION = False

def get_trends(patients, trend_types = TREND_TYPE):
    for trend_type in trend_types:
        filename = "../data_"+LABEL+"/" + trend_type + "_predict.csv"
        trend_records = pd.read_csv(filename, sep=',', header=0).values[:,1:]
        for trend_record in trend_records:
            patient_id = trend_record[0]
            if patient_id in patients:
                patients[patient_id].add_trend(trend_record[1], trend_type)
            else:
                patients[patient_id] = Patient.Patient(patient_id)
                patients[patient_id].add_trend(trend_record[1], trend_type)

def load_model(filename):
    return joblib.load(filename)

def get_markov_chain_for_each(model, obs, lengths, patients, col_index):
    print("obs.shape: ", obs.shape)
    obs = utils.preprocess_data(obs, THREADHOLD[col_index])
    start_index = 0
    for i in range(0, len(lengths)):
        patient_id = lengths[i,0]
        end_index = start_index+lengths[i,1]
        states = model.predict(obs[start_index:end_index, :])
        # print("states: ", states)
        state_frequency = np.bincount(states)
        state_frequency = np.pad(state_frequency,(0,N_COMPONENTS-state_frequency.shape[0]),'constant',constant_values=0)
        if USE_PROPORTION:
            state_proportion = state_frequency/np.sum(state_frequency)
        else:
            state_proportion = state_frequency
        # print("state_frequency: ", state_frequency)
        if patient_id in patients:
            patients[patient_id].add_state_proportion(state_proportion, col_index)
        else:
            patients[patient_id] = Patient.Patient(patient_id)
            patients[patient_id].add_state_proportion(state_proportion, col_index)
        start_index = end_index


def get_markov_chains(obses, lengths, patients):
    for i in range(len(obses)):
        model_name = MODEL_NAME+TYPE[i]+'.pkl'
        model = load_model(model_name)
        get_markov_chain_for_each(model, obses[i], lengths[i], patients, i)

def get_train_and_test(chains, items, sample_rate=0.8):
    n_train = int(chains.shape[0] * sample_rate)
    indices = np.random.permutation(chains.shape[0])
    training_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train = chains[training_idx]
    X_test = chains[test_idx]
    y_train = items[training_idx]
    y_test = items[test_idx]
    return X_train, X_test, y_train, y_test

def predict_trends_rf(max_depth, X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(max_depth = max_depth, min_samples_leaf=5)
    y_train = y_train.flatten()
    clf.fit(X_train[:,3:], y_train)
    pred = clf.predict(X_test[:,3:])
    correct = (pred == y_test).sum()
    print(correct/y_test.shape[0])
    utils.draw_result(pred[0:100], y_test[0:100])

    return correct/y_test.shape[0]

def pair_X_and_y_for_patients(obses, lengths, patients):
    get_markov_chains(obses, lengths, patients)
    get_trends(patients)
    for key in list(patients.keys()):
        if patients[key].has_nan_var():
            del patients[key]
        else:
            patients[key].concate_proportions()
    return patients

def get_patients_data():
    OBS_FILE, LENGTH_FILE = utils.set_filename(LABEL, TYPE)
    obses, lengths = utils.read_files(OBS_FILE, LENGTH_FILE)
    assert len(obses) == len(lengths)

    patients = dict()
    Patient.Patient.vol = len(TYPE)
    print("vol: ", Patient.Patient.vol)
    Patient.Patient.trend_types = TREND_TYPE
    return pair_X_and_y_for_patients(obses, lengths, patients)

patients = get_patients_data()
X = []
y = []
index = []
for key in patients:
    X.append(patients[key].get_proportions())
    y.append(patients[key].get_trends_by_key(TREND_TYPE[0]))
    index.append(patients[key].patient_id)
X = np.asarray(X)

pca = PCA(n_components=4)
X = X / X.max(axis=0)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
X = pca.fit_transform(X)

y = np.asarray(y).astype(int)
index = np.asarray(index)
print(X.shape)
print(y.shape)
print("the trend type for first 20 patients: ",y[:20])
freq = np.bincount(y)
print("frequency for each trend type: ", freq)
X_train, X_test, y_train, y_test = get_train_and_test(X, y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
predict_trends_rf(10, X_train, y_train, X_test, y_test)
# df = pd.DataFrame({"id": index, "mbs_0" : X[:,0], "mbs_1" : X[:,1], "mbs_2" : X[:,2],
#     "pbs_0" : X[:,3], "pbs_1" : X[:,4], "pbs_2" : X[:,5], "trend type" : y})
# df.to_csv("patients.csv", index=False)
