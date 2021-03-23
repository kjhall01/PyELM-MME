from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
import hpelm
from .ensemblemean import EnsembleMean
from .svd import SVD


class SPM:
	"""	Single Point Modeler class to standardize interfaces of MME
		implementation classes
		------------------------------------------------------------------------
		MME Methodologies Standardized:
			- Multiple Linear regression (MLR): sklearn.linear_model.LinearRegression
			- Extreme Learning Machine (ELM): hpelm.ELM
			- Ensemble Mean (EM): numpy.nanmean
			- Bias-Corrected Ensemble Mean (BCEM): numpy.nanmean
		------------------------------------------------------------------------
		Methods:

			train(
				**fits the model to data**
				x: np.array - shape (n_samples x n_input_features)
				y: np.array - shape (n_samples x n_targets) (no guarantees if n_targets > 1)
			)

			predict(
				**makes predictions using trained model**
				x: np.array - shape (n_samples x n_input_features)
			)
		"""

	def __init__(self, model, xtrain_shape=7, ytrain_shape=1, hidden_layer_neurons=5, activation='sigm', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', W=None, pca=None):
		self.model_type = model

		if model in ['ELM']:
			self.model = hpelm.ELM(xtrain_shape, ytrain_shape)
			self.model.add_neurons(hidden_layer_neurons, activation)
		elif model in ['MLR']:
			self.model = LinearRegression( fit_intercept=fit_intercept)
		elif model in ['EM']:
			self.model = EnsembleMean()
		else:
			print('invalid model type {}'.format(model))

	def train(self, x, y):
		if self.model_type in [ 'MLR']:
			self.model.fit(x,y)
		elif self.model_type in ['ELM']:
			score = self.model.train(x, y, 'r')
		elif self.model_type in [ 'EM']:
			pass
		else:
			print('invalid model type')

	def predict(self, x):
		return self.model.predict(x)
