import numpy as np
import pandas as pd
import math
import operator
import matplotlib.pyplot as plt
from matplotlib import gridspec

from random import randint
from hmmlearn import hmm
from hmmlearn import base
from hmmlearn.utils import normalize, log_normalize
from scipy.misc import logsumexp
from sklearn.externals import joblib



class poissonHMM(base._BaseHMM):
	"""Parameters
	-------------
	lamdas 1-D array, shape(n_components)

	"""
	def __init__(self, n_components=1,
				 startprob_prior=1.0, transmat_prior=1.0,
				 algorithm="viterbi", random_state=None,
				 n_iter=100, tol=1e-2, verbose=False,
				 params="sto", init_params="sto"):
		base._BaseHMM.__init__(self, n_components=n_components,
						  startprob_prior=startprob_prior,
						  transmat_prior=transmat_prior, algorithm=algorithm,
						  random_state=random_state, n_iter=n_iter,
						  tol=tol, params=params, verbose=verbose,
						  init_params=init_params)

	def _init(self, X, lengths=None):
		super()._init(X, lengths)
		self.n_features = X.shape[1]
		self.lamdas_ = np.random.rand(self.n_features, self.n_components)*10

		print("self.lamdas_init: ", self.lamdas_)
		# for i in range(n_features):
		# 	for j in range(n_components):
		# 		self.lamdas_[i,j] = i+j
		# self.lamdas_ = np.full((self.n_features, self.n_components), 1/n_components)
		# print("self.n_components: ", self.n_components)

	def _check(self):
		super()._check()

	# convert likelihood to x_i * log(lamda) -
	# TODO: try to vectorize this code script
	# TODO: When number of observation is large, we may able to use Normal Approximation to
	def _compute_log_likelihood(self, X):
		# print("self.lamdas_.shape: ", self.lamdas_.shape)
		# print("self.lamdas_: ", self.lamdas_)
		prob = np.full((len(X), self.n_components),1.)
		for t in range(0, len(X)):
			for i in range(0, self.n_components):
				for j in range (0, self.n_features):
					num = math.exp(0-self.lamdas_[j,i])
					numerator = self.lamdas_[j,i] ** X[t,j]
					denominator = math.factorial(X[t,j])
					# sum = X[t,j]*math.log(self.lamdas_[j,i])
					# - math.log(math.factorial(X[t,j]))
					# - self.lamdas_[j,i]
					prob[t,i] *= num*numerator/denominator
				# prob[t,i] += sum
		# print("prob: ", np.log(prob))
		return np.log(prob)

	def _generate_sample_from_state(self, state, random_state=None):
		# index = np.random.choice(list(range(0,self.n_components)),p=self.transmat_[state,:])
		# index, value = max(enumerate(self.transmat_[state,:]), key=operator.itemgetter(1))
		print("will go to state: ", state)
		new_state = np.zeros(self.n_features)
		for i in range(self.n_features):
			new_state[i] = np.random.poisson(lam=(self.lamdas_[i,state]), size=1)

		print("with observation: ", new_state)
		return new_state

	def _initialize_sufficient_statistics(self):
		stats = super()._initialize_sufficient_statistics()
		stats['post'] = np.zeros(self.n_components)
		stats['obs']  = np.zeros((self.n_components, self.n_features))
		return stats

	#call super
	def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
										  posteriors, fwdlattice, bwdlattice):

		# print("fwdlattice: ", fwdlattice)
		# print("bwdlattice: ", bwdlattice)
		# print("posteriors: ", posteriors)
		log_gamma = fwdlattice + bwdlattice
		# print("log_gamma: ", log_gamma)
		log_normalize(log_gamma, axis=1)
		# print("equal: ", np.exp(log_gamma))

		super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
		if 'o' in self.params:
			stats['post'] += posteriors.sum(axis=0)
			stats['obs'] += np.dot(posteriors.T, X)
			# print("accumulate post", stats['post'])
			# print("accumulate obs", stats['obs'])
		return 0

	def _do_mstep(self, stats):
		# print("self.startprob_: ", self.startprob_)
		# print("self.transmat_: ", self.transmat_)
		# print("stats.start: ", stats['start'])
		# print("stats.trans: ", stats['trans'])
		super()._do_mstep(stats)
		if 'o' in self.params:
			# print("obs: ", stats['obs'])
			# print("post: ", stats['post'])
			self.lamdas_ = np.divide(stats['obs'].T,stats['post'])

	def _print_info(self):
		print("self.lamdas_: ", self.lamdas_)
		# print("self.startprob_: ", self.startprob_)
		# print("self.transmat_: ", self.transmat_)


	def _get_transmat(self):
		return self.transmat_

	def _get_lamdas(self):
		return self.lamdas_

def draw(X1,Z1,figure_name1,X2,Z2,figure_name2):
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



n_components = 5
n_features = 3
model = poissonHMM(n_components=n_components, n_iter=2000)

# Training
n_samples = 30
X1 = np.random.poisson(lam=(0.), size=(n_samples, n_features))
X2 = np.random.poisson(lam=(5.), size=(n_samples, n_features))
X3 = np.random.poisson(lam=(20.), size=(n_samples, n_features))
X4 = np.random.poisson(lam=(55.), size=(n_samples, n_features))
X5 = np.random.poisson(lam=(60.), size=(n_samples, n_features))
X_train = np.concatenate([X1, X2, X3, X4, X5])
np.random.shuffle(X_train)
lengths = [len(X1), len(X2), len(X3), len(X4), len(X5)]
model.fit(X_train,lengths)
model._print_info()
Z_train = model.predict(X_train)

# Predict
n_samples = 30
X1 = np.random.poisson(lam=(0.), size=(n_samples, n_features))
X2 = np.random.poisson(lam=(30.), size=(n_samples, n_features))
X3 = np.random.poisson(lam=(60.), size=(n_samples, n_features))
X = np.zeros((n_samples, n_features))
for i in range(0, n_samples):
	num = randint(0,2)
	if num == 0:
		X[i,:] = X1[i,:]
	elif num == 1:
		X[i,:] = X2[i,:]
	else:
		X[i,:] = X3[i,:]
Z = model.predict(X)
draw(X_train,Z_train,"training", X,Z,"predicting")

# Sample and Test
# n_samples = 9999
# sample = model.sample(n_samples)
# states = sample[1]
# matrix = np.zeros((n_components, n_components))
# for i in range(0, len(states)-1):
# 	matrix[states[i],states[i+1]] += 1
# for i in range(0, n_components):
# 	matrix[i] = matrix[i]/(matrix.sum(axis=1)[i])
# print("=======Sample Transmat=======")
# print(matrix)
# print("=======Model Transmat=======")
# print(model._get_transmat())
sample = model.sample(30)
Z_sample = model.predict(sample[0])
# print(sample[0])
print(sample[1])
draw(sample[0], sample[1], "sampling", sample[0], Z_sample, "predicting")



# print(Z2)
