import numpy as np

class Scaler:
	"""Handles all data standardization / scaling
	------------------------------------------------------------------------
		Methodologies:
			Minmax: scales to a predefined range by range_minimum + predefined_range * (x-data_minimum) / (data_range)

			Standard Anomaly / Gauss / Normalize: Subtract mean, divide by standard deviation
	--------------------------------------------------------------------------
		Class Methods:
			constructor(
				**initialize scaler object**
				minmax_range: list - length 2, first member is minimum scaling range, second is maximum
			)

			fit(
				**save metadata about data**
				data: np.array - shape (n_samples x n_features)
			)

			transform(
				**applies scaling to data provided**
				data: np.array - shape (n_samples x n_features)
				method: string - indicates methodology ['minmax', 'std_anomaly', None]
			)

			fit_transform(
				**combine fit and transform methods for convenience**
				data: np.array - shape (n_samples x n_features)
				method: string - indicates methodology ['minmax', 'std_anomaly', None]
			)

			recover(
				**reverse the scaling, transform data back to original distribution**
				data: np.array - shape (n_samples x n_features)
				method: string - indicates methodology ['minmax', 'std_anomaly', None]
			)
	----------------------------------------------------------------------------"""


	def __init__(self, minmax_range=[-1,1]):
		""" initialize scaler object """
		self.mu = 0
		self.std = 1
		self.mx, self.mn = 0, 0
		self.range = 0
		self.minmax_range_mn = minmax_range[0]
		self.minmax_range_size = minmax_range[1] - minmax_range[0]
		self.method = None

	def fit(self, data):
		"""save metadata for use in scaling """
		self.mu = np.nanmean(data, axis=0)
		self.std = np.nanstd(data, axis=0)
		self.mx, self.mn = np.nanmax(data, axis=0), np.nanmin(data, axis=0)
		self.range = self.mx - self.mn



	def transform(self, data, method='siriehtesolc'):
		"""apply scaling to data """
		if method == 'siriehtesolc':
			method = self.method
		self.method=method
		if method == 'minmax':
			scaled = self.minmax_range_mn +  self.minmax_range_size * ((data - self.mn) / self.range)
			return  scaled
		elif method == 'std_anomaly':
			return (data - self.mu) / self.std
		else:
			return data #

	def fit_transform(self, data, method):
		"""combine fit and transform methods"""
		self.fit(data)
		return self.transform(data, method=method)

	def recover(self, output, method='siriehtesolc'):
		"""transform data back to original distribution"""
		if method == 'siriehtesolc':
			method = self.method
		if method=='minmax':
			scaled = (((output - self.minmax_range_mn) / self.minmax_range_size) * self.range[1] + self.mn[1]).reshape(-1,1)
			return scaled
		elif method == 'std_anomaly':
			return (output * self.std[1]) + self.mu[1]
		else:
			return output
