import numpy as np


class SVD():
	"""Wrapper class for Singular Value Decomposition MLR methodology"""
	#method from https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/
	def fit(self, x, y):
		self.b = np.linalg.pinv(x).dot(y)
		return self

	def predict(self, x):
		return x.dot(self.b)
