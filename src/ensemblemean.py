import numpy as np

class EnsembleMean():
	"""Wrapper class for Ensemble Mean"""
	def predict(self, x):
		return np.nanmean(x, axis=1).reshape(-1,1)
