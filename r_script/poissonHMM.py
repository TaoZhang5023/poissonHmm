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
	def __init__(self, n_components=1, lamdas_=None,
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
		self.lamdas_ = lamdas_

	def _init(self, X, lengths=None):
		super()._init(X, lengths)
		self.n_features = X.shape[1]
		# self.lamdas_ = np.random.rand(self.n_features, self.n_components)*10
		print("self.lamdas_init: ", self.lamdas_)

	def _check(self):
		super()._check()

	def _compute_log_likelihood(self, X):
		prob = np.full((len(X), self.n_components),1.)
		for t in range(0, len(X)):
			for i in range(0, self.n_components):
				for j in range (0, self.n_features):
					try:
						num = math.exp(0-self.lamdas_[j,i])
						numerator = self.lamdas_[j,i] ** X[t,j]
						denominator = math.factorial(X[t,j])
						prob[t,i] *= num*numerator/denominator
						# print("calculating num: ", num, "numerator: ", numerator, "denominator", denominator)
					except:
						print("num: ", num, "numerator: ", numerator, "denominator", denominator)
		return np.log(prob)

	def _generate_sample_from_state(self, state, random_state=None):
		new_state = np.zeros(self.n_features)
		for i in range(self.n_features):
			new_state[i] = np.random.poisson(lam=(self.lamdas_[i,state]), size=1)
		return new_state

	def _initialize_sufficient_statistics(self):
		stats = super()._initialize_sufficient_statistics()
		stats['post'] = np.zeros(self.n_components)
		stats['obs']  = np.zeros((self.n_components, self.n_features))
		return stats

	#call super
	def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
										  posteriors, fwdlattice, bwdlattice):
		log_gamma = fwdlattice + bwdlattice
		log_normalize(log_gamma, axis=1)
		super()._accumulate_sufficient_statistics(
            stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
		if 'o' in self.params:
			stats['post'] += posteriors.sum(axis=0)
			stats['obs'] += np.dot(posteriors.T, X)
		return 0

	def _do_mstep(self, stats):
		super()._do_mstep(stats)
		if 'o' in self.params:
			self.lamdas_ = np.divide(stats['obs'].T,stats['post'])

	def _print_info(self):
		print("self.lamdas_: ", self.n_components)
		print("self.lamdas_: ", self.lamdas_)
		print("self.startprob_: ", self.startprob_)
		print("self.transmat_: ", self.transmat_)

	def predict_hospital(self, X, index=2):
		state = self.predict(X)[-1]
		prob = np.zeros(self.n_components)
		lamdas = self.lamdas_[index, :]
		prob = np.exp(0-lamdas)
		prob = prob * self.transmat_[state, :]
		return 1-prob.sum()

	def _get_transmat(self):
		return self.transmat_

	def _get_lamdas(self):
		return self.lamdas_
