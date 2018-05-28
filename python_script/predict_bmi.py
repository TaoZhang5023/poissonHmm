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


N_STATE = 5
OBS_FILE = "../data/obs_predict.csv"
LENGTH_FILE = "../data/length_predict.csv"
BMI_FILE = "../data/bmi_predict.csv"
TCHOL_FILE = "../data/tchol_predict.csv"
GLUCOSE_FILE = "../data/glucose_predict.csv"
SYSBP_FILE = "../data/sysbp_predict.csv"
DIASBP_FILE = "../data/diasbp_predict.csv"

# BMI_FILE = "../data/bmi_change.csv"
# TCHOL_FILE = "../data/tchol_change.csv"
# GLUCOSE_FILE = "../data/glucose_change.csv"
# SYSBP_FILE = "../data/sysbp_change.csv"
# DIASBP_FILE = "../data/diasbp_change.csv"
MODEL_NAME = "../model/model_6m_cate25_99.pkl"

def read_files(obs_file = OBS_FILE, length_file = LENGTH_FILE,
    bmi_file = BMI_FILE, tchol_file = TCHOL_FILE, glucose_file = GLUCOSE_FILE,
    sysbp_file = SYSBP_FILE, diasbp_file = DIASBP_FILE):
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,[2,8]]
    lengths = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    bmi = pd.read_csv(bmi_file, sep=',', header=0).values[:,2]
    tchol = pd.read_csv(tchol_file, sep=',', header=0).values[:,2]
    glucose = pd.read_csv(glucose_file, sep=',', header=0).values[:,2]
    sysbp = pd.read_csv(sysbp_file, sep=',', header=0).values[:,2]
    diasbp = pd.read_csv(diasbp_file, sep=',', header=0).values[:,2]
    return obs,lengths,bmi,tchol,glucose,sysbp,diasbp

def load_model(filename):
    return joblib.load(filename)

def get_markov_chain(obs, lengths, model):
    # frequency = np.zeros((len(lengths),N_STATE))
    size = np.amax(lengths)
    chains = np.zeros((len(lengths), size))
    start_index = 0
    for i in range(0, len(lengths)):
        for j in range(1,lengths[i]):
            states = model.predict(obs[start_index:start_index+j, :])
            states = np.pad(states,(0,size-states.shape[0]),'constant',constant_values=0)
            chains[i] = states
            # state_frequency = np.bincount(states)
            # state_frequency = np.pad(state_frequency,(0,5-state_frequency.shape[0]),'constant',constant_values=0)
            # frequency[i] = state_frequency
        start_index += lengths[i]
    return chains

def predict_trends_rf(max_depth, X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(max_depth = max_depth)
    y_train = y_train.flatten()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    correct = 0
    # print("actual\tpred")
    for i in range(y_test.shape[0]):
        # print(y_test[i], '\t', pred[i])
        if (pred[i] == y_test[i]):
            correct += 1
    return correct/y_test.shape[0]

def predict_trends_gbr(max_depth, X_train, y_train, X_test, y_test):
    clf = GradientBoostingRegressor(loss='lad', n_estimators=1000, max_depth = 50, learning_rate=0.01)
    y_train = y_train.flatten()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print("test set")
    print(np.var(pred), "\t", np.var(y_test))
    print(np.mean(pred), "\t", np.mean(y_test))
    print(levene(pred, y_test))
    plt.plot(pred, 'r.')
    plt.plot(y_test, 'b.', alpha=0.3)
    plt.show()
    print("---")
    print("train set")
    pred = clf.predict(X_train)
    print(np.var(pred), "\t", np.var(y_train))
    print(np.mean(pred), "\t", np.mean(y_train))
    print(levene(pred, y_train))
    print("==============")
    # correct = 0

    # print("feature_importances_: ", clf.feature_importances_)
    # print("actual\tpred")
    # for i in range(y_test.shape[0]):
    #     # print(y_test[i], '\t', pred[i])
    #     if (pred[i] == y_test[i]):
    #         correct += 1
    # return correct/y_test.shape[0]

def get_train_and_test(X,y, sample_rate=0.8):
    n_train = int(X.shape[0] * sample_rate)
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train = X[training_idx]
    X_test = X[test_idx]
    y_train = y[training_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test

def test():
    obs,lengths,bmi,tchol,glucose,sysbp,diasbp = read_files()
    print(lengths.shape)

    y = [bmi,glucose,tchol,sysbp,diasbp]
    y_name = ["bmi","glucose","tchol","sysbp","diasbp"]
    model = load_model(MODEL_NAME)
    # print(model._print_info())
    X = get_markov_chain(obs, lengths, model)
    # X = X[:,[1,2,4]]
    X = X.astype(int)
    X = X/np.sum(X,axis=1)[:,None]
    X = np.nan_to_num(X)
    np.savetxt("state_frequency.csv", X, delimiter=",")
    print(X.shape)

    for i in range(len(y)):
        y[i] = np.nan_to_num(y[i])
        # y[i] -= np.mean(y[i])
        # y[i] /= np.amax(y[i])
        X_train, X_test, y_train, y_test = get_train_and_test(X, y[i])
        print(y_name[i])
        print(predict_trends_rf(7, X_train, y_train, X_test, y_test))


def feature_selection():
    obs,lengths,bmi,tchol,glucose = read_files()
    model = load_model(MODEL_NAME)
    # print(model._print_info())
    X = get_markov_chain(obs, lengths, model)
    X = X.astype(int)
    X = X/np.sum(X,axis=1)[:,None]
    X = np.nan_to_num(X)
    Y = np.nan_to_num(bmi).reshape(bmi.shape[0],1)

    #US
    test = SelectKBest(score_func = f_regression, k=4)
    fit = test.fit(X,Y)
    # summarize scores US
    np.set_printoptions(precision=3)
    print(fit.scores_)
    print(fit.pvalues_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:5,:])

    # RFE feature extraction
    estimator = SVR(kernel="linear")
    rfe = RFE(estimator, 5)
    fit = rfe.fit(X, Y)
    print(fit.n_features_)
    print(fit.support_)
    print(fit.ranking_)

    # PCA feature extraction
    pca = PCA(n_components=3)
    fit = pca.fit(X)
    # summarize components
    print(fit.explained_variance_ratio_)
    print(fit.components_)

def try_model():
    x = np.random.rand(1000,)
    y = np.random.rand(1000).reshape(1000,1)

    X_train, X_test, glucose_train, glucose_test = get_train_and_test(x, y)
    plt.plot(X_train, glucose_train,'.')
    plt.show()
    predict_trends_gbr(7, X_train, glucose_train, X_test, glucose_test)

test()
# try_model()
