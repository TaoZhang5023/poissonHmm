import poissonHMM as poissonHMM
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from matplotlib import gridspec

def draw_compare(X1,Z1,figure_name1,X2,Z2,figure_name2):
    n_samples, n_components = X1.shape
    fig, (ax1,ax2) = plt.subplots(1,2)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax1.set_title(figure_name1)
    ax1.set_xlabel('time')
    ax1.set_ylabel('observation')
    barlist = ax1.bar(np.arange(0,n_samples), X1[:,0])
    ax1.tick_params(axis='y')
    legends = []
    labels = []
    for i in range(0, n_samples):
        color = "C%s" %(Z1[i])
        barlist[i].set_color(color)
    ax1.legend(legends, labels)

    n_samples, n_components = X2.shape
    ax2 = plt.subplot(gs[1])
    ax2.set_title(figure_name2)
    ax2.set_xlabel('time')
    ax2.set_ylabel('observation')
    barlist = ax2.bar(np.arange(0,n_samples), X2[:,0])
    ax2.tick_params(axis='y')
    legends = []
    labels = []
    for i in range(0, n_samples):
        color = "C%s" %(Z2[i])
        barlist[i].set_color(color)
    ax2.legend(legends, labels)

    plt.show()

def draw(X,Z,figure_name):
    n_samples, n_components = X.shape
    fig, ax1 = plt.subplots()
    ax1.set_title(figure_name)
    ax1.set_xlabel('time')
    ax1.set_ylabel('observation')
    barlist = plt.bar(np.arange(0,n_samples), X[:,0])
    plt.tick_params(axis='y')
    legends = []
    labels = []
    for i in range(0, n_samples):
        color = "C%s" %(Z[i])
        barlist[i].set_color(color)
    plt.show()

def load_model(filename):
    return joblib.load(filename)

def test_model(obs_file='data/obsv1.csv', length_file='data/lengthv1.csv', n_split=500, n_components=5, n_iter=2000):
    model = poissonHMM.poissonHMM(n_components=n_components, n_iter=n_iter)
    length = pd.read_csv(length_file, sep=',', header=0).values[:,2]
    obs = pd.read_csv(obs_file, sep=',', header=0).values[:,[3,4,5]]
    print("There are ", obs.shape[0], " observations and ", len(length), " patients.")
    lengths = np.array_split(length, n_split)
    for i in range(0, len(lengths)):
        start_index = 0
        end_index = start_index + lengths[i].sum()
        test_data = obs[start_index:end_index, :]
        b = test_data > 28
        test_data[b] = 28
        model = load_model("model2.pkl")
        print(model)
        Z = model.predict(test_data, lengths[i])
        draw(test_data,Z,"result"+str(i))
    print("end")


test_model()
