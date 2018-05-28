import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

obs = pd.read_csv("patients.csv", sep=',', header=0).values[:,1:-1]
print(obs.shape)
pca = PCA(n_components=4)
obs = obs / obs.max(axis=0)
pca.fit(obs)
# plt.scatter(obs[:, 3], obs[:, 4],marker='o')
# plt.show()
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
obs = pca.fit_transform(obs)
np.savetxt("pca.csv", obs, delimiter=",")
