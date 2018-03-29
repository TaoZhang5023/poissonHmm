import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib

N_COMPONENTS = 5
N_ITER = 100
N_SPLIT = 10
N_FEATURE = 3
OBS_FILE = "../data/obs_halfyear.csv"
LENGTH_FILE = "../data/length_halfyear.csv"


def read_files(obs_file, length_file):
    length = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,[3,4,5]]
    return obs,length

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

def validation(train_data, test_data, train_length, test_length, i, lamdas_):
    b=train_data > 80
    train_data[b] = 80
    b = test_data > 80
    test_data[b] = 80
    model = poissonHMM.poissonHMM(n_components=N_COMPONENTS, n_iter=N_ITER, lamdas_=lamdas_)
    model.fit(train_data, train_length)
    filename = "../model/model_halfyear"+str(i)+".pkl"
    save_model(model, filename)
    # Train cross validated likelihood
    likelihood = 0-model.score(train_data, train_length)
    n_events = train_length.sum()*3
    print("train_cross_validated_likelihood: ", likelihood/n_events)
    train_cross_validated_likelihood[N_COMPONENTS-1] += likelihood/n_events
    # Test cross validated likelihood
    likelihood = 0-model.score(test_data, test_length)
    n_events = test_length.sum()*3
    print("test_cross_validated_likelihood: ", likelihood/n_events)
    test_cross_validated_likelihood[N_COMPONENTS-1] += likelihood/n_events
    # AIC
    P = N_COMPONENTS * N_COMPONENTS + N_COMPONENTS * N_FEATURE
    AIC[N_COMPONENTS-1] += likelihood + 2*P
    print("AIC: ", likelihood + 2*P)

    # BIC
    BIC[N_COMPONENTS-1] += likelihood + P*np.log(n_events)
    print("BIC: ", likelihood + P*np.log(n_events))


def save_model(model, filename):
    joblib.dump(model, filename)

def test_model(obs_file=OBS_FILE, length_file=LENGTH_FILE, n_split=N_SPLIT):
    obs, length = read_files(obs_file, length_file)
    print("There are ", obs.shape[0], " observations and ", len(length), " patients.")
    lengths = np.array_split(length, n_split)
    lamdas_ = np.random.rand(N_FEATURE, N_COMPONENTS)*10
    for i in range(0, len(lengths)):
        start_index = 0
        print("the ", i , "th round.")
        # Calculate the mask
        for j in range(0, i):
            start_index += lengths[j].sum()
        end_index = start_index + lengths[i].sum()
        test_data, train_data = split_data(obs, start_index, end_index)
        test_length, train_length = split_length(lengths, i)
        validation(train_data, test_data, train_length, test_length, i, lamdas_)
    print("end")

#main
N_STATE = 5
test_cross_validated_likelihood = np.zeros(N_STATE)
train_cross_validated_likelihood = np.zeros(N_STATE)
AIC = np.zeros(N_STATE)
BIC = np.zeros(N_STATE)
for i in range(0,N_STATE):
    N_COMPONENTS = i+1
    test_model()
test_cross_validated_likelihood = test_cross_validated_likelihood/N_SPLIT
train_cross_validated_likelihood = train_cross_validated_likelihood/N_SPLIT
AIC = AIC/N_SPLIT
BIC = BIC/N_SPLIT
print('\t\t\t', "test_cvl\t", "train_cvl\t", "AIC\t", "BIC\t")
for i in range(0, N_STATE):
    print(i+1,"States:\t", test_cross_validated_likelihood[i], '\t', train_cross_validated_likelihood[i],
    '\t', AIC[i], '\t', BIC[i])
