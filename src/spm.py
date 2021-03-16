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
			constructor(
				**Returns Single Point Modeler Object - wrapper for another regressor class***
				model: string - an MME Methodogoly [MLR, PCR, EM, BCEM, Ridge, MLR-SVD, SVD, ELM, PCA-ELM, ELM-PCA, SLFN)
				xtrain_shape: int - number of input features for ELM, ELM-PCA, PCA-ELM
				ytrain_shape: int - number of targets for ELM, PCA-ELM, ELM-PCA to predict
				xval_window: int - number of samples to leave out during each cross validation round
				hidden_layer_neurons: int - number of neurons in hidden layer of ELM, PCA-ELM, ELM-PCA
				activation: string - activation function for ELM, PCA-ELM, ELM-PCA ['sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf']
				standardization: string - data scaling methodology ['minmax', 'std_anomaly', None]
				max_iter: int - maximum number of training iterations over dataset for SLFN
				normalize: Boolean - whether to normalize data by default for Ridge Regression
				fit_intercept: Boolean - whether to use intercept in calculation of MLR / Ridge Regressions
				alpha: float - alpha for SLFN
				solver: string - training algorithm for SLFN ['auto', 'adam', 'lbfgs', 'sgd'] (check docs)
				W: np.array - matrix of n_features x n_components  for initialization of ELM weights. only used in PCA-ELM
				pca: sklearn.decomposition.PCA object - used to initialize W, B, and # neurons in ELM model based on PCA transformation of data
			)

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
		if model == 'SVD':
			self.model = SVD()
		elif model == 'MLR-SVD':
			self.model =  Ridge( normalize=normalize, fit_intercept=fit_intercept, alpha=0.0, solver='svd')
		elif model in ['ELM', 'EWP']:
			self.model = hpelm.ELM(xtrain_shape, ytrain_shape)
			self.model.add_neurons(hidden_layer_neurons, activation)
		elif model == 'PCA-ELM':
			self.model = hpelm.ELM(xtrain_shape, ytrain_shape)
			self.model.add_neurons(pca.components_.shape[0], activation, W=pca.components_.T[:xtrain_shape,:], B=np.arange(pca.components_.shape[0]))
		elif model in ['MLR', 'PCR']:
			self.model = LinearRegression( fit_intercept=fit_intercept)
		elif model == 'Ridge':
			self.model =  Ridge( normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver)
		elif model == 'SLFN':
			self.model =  MLPRegressor(max_iter=max_iter, solver=solver, hidden_layer_sizes=(hidden_layer_neurons), activation=activation)
		elif model in ['EM', 'BCEM']:
			self.model = EnsembleMean()
		else:
			print('invalid model type {}'.format(model))

	def train(self, x, y):
		if self.model_type in ['SVD', 'MLR-SVD', 'MLR', 'Ridge', 'PCR']:
			self.model.fit(x,y)
		elif self.model_type in ['ELM', 'PCA-ELM', 'EWP']:
			score = self.model.train(x, y, 'r')
		elif self.model_type == 'SLFN':
			self.model.fit(x, y.reshape(-1,1).ravel())
		elif self.model_type in [ 'EM', 'BCEM' ]:
			pass
		else:
			print('invalid model type')

	def predict(self, x):
		return self.model.predict(x)
