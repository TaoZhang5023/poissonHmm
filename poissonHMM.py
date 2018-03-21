import numpy as np
import pandas as pd
import math
from hmmlearn import hmm
from hmmlearn import base
from hmmlearn.utils import normalize
from scipy.misc import logsumexp
from sklearn.externals import joblib



class poissonHMM(base._BaseHMM):
	"""Parameters
	-------------
	lamdas 1-D array, shape(n_components)

	"""
	def __init__(self, n_components=1,
				 startprob_prior=1.0, transmat_prior=1.0, lamdas_prior=None,
				 algorithm="viterbi", random_state=None,
				 n_iter=10, tol=1e-2, verbose=False,
				 params="sto", init_params="sto"):
		base._BaseHMM.__init__(self, n_components=n_components,
						  startprob_prior=startprob_prior,
						  transmat_prior=transmat_prior, algorithm=algorithm,
						  random_state=random_state, n_iter=n_iter,
						  tol=tol, params=params, verbose=verbose,
						  init_params=init_params)
		self.lamdas_prior = np.full(n_components, 1/n_components)
		# self.lamdas = lamdas
		# self.transmat_matrix = transmat_matrix
		# self.stationary_distribution = stationary_distribution

	def _init(self, X, lengths=None):
		super()._init(X, lengths)
		self.lamdas_ = np.full(n_components, 1/n_components)
		self.n_features = X.shape[1]

	def _check(self):
		super()._check()

	# convert likelihood to x_i * log(lamda) - 
	def _compute_log_likelihood(self, X):
		prob = np.zeros((len(X), n_components))
		for i in range(0, len(X)):
			for j in range(0, n_components):
				num = math.exp(0-self.lamdas_[j])
				numerator = self.lamdas_[j] ** X[i]
				denominator = math.factorial(X[i])
				prob[i,j] = num*numerator/denominator
		return np.log(prob)

	def _generate_sample_from_state(self, state, random_state=None):
		return 0

	def _initialize_sufficient_statistics(self):
		stats = super()._initialize_sufficient_statistics()
		stats['lamb'] = np.zeros(self.n_components) 
		stats['post'] = np.zeros(self.n_components)
		stats['obs']  = np.zeros((self.n_components, self.n_features))
		# TODO: This need to be checked
		return stats

	def compute_obs(self, n_samples, n_components, fwdlattice, bwdlattice, obs):
		for t in range(n_samples):
			for i in range(n_components):
				obs[t,i] = fwdlattice[t,i] + bwdlattice[t,i]


	#call super 
	def _accumulate_sufficient_statistics(self, stats, X, framelogprob, 
										  posteriors, fwdlattice, bwdlattice):
		super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
		if 'o' in self.params:
			stats['post'] += posteriors.sum(axis=0)
			stats['obs'] += np.dot(posteriors.T, X)
		return 0

	# def _do_forward_pass(self, framelogprob):
	# 	n_samples, n_components = framelogprob.shape
	# 	fwdlattice = np.zeros((n_samples, n_components))
	# 	fwdlattice[0,:] = self.stationary_distribution * framelogprob[0, :]
	# 	for i in range(1, n_samples):
	# 		for j in range(0, n_components):
	# 			sum = fwdlattice[i-1,:] * self.transmat_matrix[:,j]
	# 			fwdlattice[i,j] = np.sum(sum)*framelogprob[i,j]
	# 	return logsumexp(fwdlattice[-1]), fwdlattice

	# def _do_backward_pass(self, framelogprob):
	# 	n_samples, n_components = framelogprob.shape
	# 	bwdlattice = np.zeros((n_samples, n_components))
	# 	bwdlattice[-1,:] = 1
	# 	for i in range(n_samples-2, -1, -1):
	# 		for j in range(0, n_components):
	# 			sum = framelogprob[i+1,:] * bwdlattice[i+1,:] * self.transmat_matrix[j,:]
	# 			bwdlattice[i,j] = np.sum(sum)
	# 	return bwdlattice

	#
	def _do_mstep(self, stats):
		super()._do_mstep(stats)
		if 'o' in self.params:
			self.lamdas_ = np.divide(stats['obs'],stats['post'].T).flatten()

n_components = 4
n_samples = 50
model = poissonHMM(n_components=n_components, n_iter=100)
# X = np.random.randint(10, size=n_samples)
X = np.random.rand(n_samples,1)*100
print("Observe: " , X.shape)
model.fit(X)
Z2 = model.predict(X)
print(Z2)

