import numpy as np
import xarray as xr
import os
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle, copy
import warnings
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA

import cartopy.crs as ccrs
from cartopy import feature
import cartopy.mpl.ticker as cticker
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
warnings.filterwarnings("ignore")

class MME:
	"""Multi-Model Ensemble class - stores & handles data
	---------------------------------------------------------------------------
		Class Attributes:
			years: (np.array) - shape (n_years, 1)
			model_data: (np.array) - shape (n_models, n_latitude, n_longitude, n_years) for Multi-Point, (n_years, n_models) for Single-Point
			observations: (np.array) - shape (1, n_latitude, n_longitude, n_years) for Multi-Point, (n_years, 1) for Single-Point
			type: (string) - either 'Single-Point' or 'Multi-Point' depending on which data reading function is used
			hindcasts: (dict) - each key represents an MME methodology, or Observations.
							- hindcasts[key] is np.array of shape (n_lats, n_lons, n_years) for multi-point,
							- or (n_years, 1) for single-point, holding cross-validated hindcasts produced by each method
			hindcast_models: (dict) - each key represents an mme methodology
								  - hindcast_models[key] will be a list of SPM objects for each year,
								  - or a list of lists of lists of shape (n_lats, n_longs, n_years)
								  - holding SPM objects for each point for each year
			xval_hindcast_metrics: (dict) - each key1 represents an skill metric
			 							  - xval_hindcast_metrics[key1] will be another (dict)
										  - each key2 represents an MME methodology
										  - xval_hindcast_metrics[key1][key2] will be an (np.array) of shape (n_years, 1) for single Point
										  - or (n_lats, n_lons, n_years) for multi-point holding skill metric scores
			training_forecast_metrics: (dict) - same as xval_hindcast_metrics but for non-cross-validated hindcasts for training data
			training_forecasts (dict) - same as hindcasts except for training data, non-xvalidated
			forecast_models (dict) - holds models trained on all training data, non xvalidated, for making realtime forecasts
			real_time_forecasts (dict) - results of applying models trained on all training data (non xvalidated) to separate rtf data provided
			forecast_scalers (dict) - scaler objects uses to scale training data, saved so we can scale new rtf data to the same distribution
			ncdf_forecast_data (dict) - same as model_data except for real time forecast data
			forecast_pca(s) (dict) - stores pca objects trained on all training data for use in transforming new rtf data the same way
	---------------------------------------------------------------------------"""
	def __init__(self, reader_ret):
		self.type = reader_ret[0]
		if self.type == 'Single-Point':
			data_type, years, observations, model_data, lats, lons, hindcasts, training_forecasts = reader_ret
			if len(observations) == 0: #this is a forecast
				assert False, 'Must provide Y input data (historical observations) to initialize MME'
			else:
				self.type, self.years = data_type, years
				self.model_data, self.observations = model_data, observations
				self.hindcasts = hindcasts
				self.training_forecasts = training_forecasts
				self.lats, self.lons = lats, lons
		else:
			(data_type, model_data, lats, lons, years, nanmask, goodmask, observations, hindcasts, training_forecasts) = reader_ret
			if len(observations) == 0: #this is a forecast
				assert False, 'Must provide Y input data (historical observations) to initialize MME'
			else:
				self.years =  years
				self.model_data, self.observations = model_data, observations
				self.hindcasts = hindcasts #this is included becase were preemptively setting the 'obs' key for comparison
				self.training_forecasts = training_forecasts #this is included becase were preemptively setting the 'obs' key for comparison
				self.nanmask, self.goodmask = nanmask, goodmask
				self.lats, self.lons = lats, lons

		#filled in later
		self.hindcast_models = {}
		self.xval_hindcast_metrics = {}
		self.training_forecast_metrics = {}
		self.forecast_models = {}
		self.real_time_forecasts = {}
		self.forecast_scalers = {}
		self.ncdf_forecast_data = []
		self.forecast_pca = {}
		self.forecast_pcas = {}


	def add_forecast(self, reader_ret):
		assert reader_ret[0] == self.type, 'must have 2d data for a 2d mme, and 1d for 1d lol'
		if self.type == 'Single-Point':
			data_type, years, observations, model_data, lats, lons, hindcasts, training_forecasts = reader_ret
			self.forecast_data = model_data
			self.fcst_years = years
			self.fcst_obs = observations
		else:
			(data_type, model_data, lats, lons, years, nanmask, goodmask, observations, hindcasts, training_forecasts) = reader_ret
			self.forecast_data = model_data
			self.fcst_years = years
			self.fcst_obs = observations
		return True



	def compute_mmes(self, mme_methodologies, args):
		print('\nComputing MMEs')
		for method in mme_methodologies:
			print(method)
			if method == 'MLR':
				self.construct_crossvalidated_mme_hindcasts(method, xval_window=args['mlr_xval_window'], standardization=args['mlr_standardization'], fit_intercept=args['mlr_fit_intercept'])
			elif method == 'ELM':
				self.construct_crossvalidated_mme_hindcasts(method, xval_window=args['elm_xval_window'], hidden_layer_neurons=args['elm_hidden_layer_neurons'], activation=args['elm_activation'], standardization=args['elm_standardization'], minmax_range=args['elm_minmax_range'])
			elif method == 'EM':
				self.construct_crossvalidated_mme_hindcasts(method, xval_window=args['em_xval_window'], standardization=None)
			else:
				print('Invalid MME {}'.format(method))

	def compute_skill(self, metrics):
		print('\nCalculating Skill')
		spearman_flag, pearson_flag = 0, 0
		for metric in metrics:
			print(metric)
			if metric in ['SpearmanCoef', 'SpearmanP'] and spearman_flag == 0:
				spearman_flag = 1
				self.Spearman()
			elif metric in ['PearsonCoef', 'PearsonP'] and pearson_flag == 0:
				pearson_flag = 1
				self.Pearson()
			elif metric == 'MAE':
				self.MAE()
			elif metric == 'MSE':
				self.MSE()
			elif metric == 'RMSE':
				self.MSE(squared=False)
			elif metric == 'IOA':
				self.IOA()
			else:
				print('Invalid Metric {}'.format(metric))

	def train_rtf_models(self, forecast_methodologies, args):
		print('\nComputing MMEs')
		for method in forecast_methodologies:
			print(method)
			if method == 'MLR':
				self.train_forecast_model(method,  standardization=args['mlr_standardization'], fit_intercept=args['mlr_fit_intercept'])
			elif method == 'ELM':
				self.train_forecast_model(method,  hidden_layer_neurons=args['elm_hidden_layer_neurons'], activation=args['elm_activation'], standardization=args['elm_standardization'], minmax_range=args['elm_minmax_range'])
			elif method == 'EM':
				self.train_forecast_model(method, standardization=None)
			else:
				print('Invalid MME {}'.format(method))


	def compute_rtf(self, inputs, forecast_methodologies, args):
		self.rtf_output = {}
		for method in forecast_methodologies:
			print(method)
			if method == 'MLR':
				self.rtf_output['MLR'] = self.forecast(inputs, method,  standardization=args['mlr_standardization'], fit_intercept=args['mlr_fit_intercept'])
			elif method == 'ELM':
				self.rtf_output['ELM'] = self.forecast(inputs, method,  hidden_layer_neurons=args['elm_hidden_layer_neurons'], activation=args['elm_activation'], standardization=args['elm_standardization'], minmax_range=args['elm_minmax_range'])
			elif method == 'EM':
				self.rtf_output['EM'] = self.forecast(inputs, method, standardization=None)
			else:
				print('Invalid MME {}'.format(method))


	def __preprocess_data(self, data, crossval_start, crossval_end, standardization, pca_variability=-1, minmax_range=[-1,1], train_scaler=None, test_scaler=None, pca=None):
		"""Cut out the cross validation window, eliminate NaN values (they will be masked out in the graphics anyway),
		 	transform to orthogonal space using PCA if necessary, scale using scaling method if necessary, return data and objects"""

		if crossval_end == crossval_start and crossval_end == 0 and train_scaler is None:
			train_data = np.asarray(data[crossval_end+1:, :])
			test_data = np.asarray(data[crossval_end, :]).reshape(1,-1)
		elif crossval_end == crossval_start and crossval_end == data.shape[0]-1 and train_scaler is None:
			train_data = np.asarray(data[:crossval_end,:])
			test_data = np.asarray(data[crossval_end, :]).reshape(1,-1)
		elif crossval_start==0 and crossval_end == data.shape[0]-1:
			train_data, test_data = copy.deepcopy(data), copy.deepcopy(data)
		else:
			train_data = np.vstack((np.asarray(data[:crossval_start, :]), np.asarray(data[crossval_end+1:, :])))
			test_data = data[crossval_start:crossval_end+1, :]

		for i in range(train_data.shape[0]):
			for j in range(train_data.shape[1]):
				if np.isnan(train_data[i,j] ):
					train_data[i,j] = np.random.randint(10)
		for i in range(test_data.shape[0]):
			for j in range(test_data.shape[1]):
				if np.isnan(test_data[i,j] ):
					test_data[i,j] = np.random.randint(10)

		if pca_variability >= 0:
			if pca is None:
				pca = PCA(pca_variability)
				train_pca_space = pca.fit_transform(train_data[:,2:])
			else:
				train_pca_space = pca.transform(train_data[:,2:])
			test_pca_space = pca.transform(test_data[:,2:])
			train_data, test_data = np.hstack((train_data[:, :2], train_pca_space)), np.hstack((test_data[:,:2], test_pca_space))

		else:
			pca = None
		if train_scaler is None:
			train_scaler, test_scaler = Scaler(minmax_range=minmax_range), Scaler(minmax_range=minmax_range)
			train_data = train_scaler.fit_transform(train_data, standardization)
		else:
			train_data = train_scaler.transform(train_data, method=standardization)
		test_data = train_scaler.transform(test_data, method=standardization)
		x_train, y_train = train_data[:, 2:], train_data[:, 1].reshape(-1,1)
		if x_train.shape[0] == 1:
			x_train = x_train.reshape(1,-1)
		x_test, y_test = test_data[:, 2:], test_data[:, 1].reshape(-1,1)
		return x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca

	def __single_point_crossvalidated_hindcasts(self, data, model_type, xval_window=3, standardization='None', pca_variability=-1, hidden_layer_neurons=5, activation='sigm', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', W=None, minmax_range=[-1,1]):
		"""train and evaluate model of type model_type using "leave-N-out" cross-validation"""

		border = xval_window // 2
		models, hindcasts = [], []
		x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self.__preprocess_data(data, 0, xval_window-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
		model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
		model.train(x_train, y_train)
		models.append(models)
		xval_preds = model.predict(x_test)
		xval_preds = train_scaler.recover(xval_preds, method=standardization)

		for year in range(0, border+1): #save hindcasts for years in border, plus center year of first xval window
			hindcasts.append(xval_preds[year])

		#now get hindcasts for all xval windows until last
		for year in range(border+1, data.shape[0]-border-1):
			x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self.__preprocess_data(data, year-border, year+border, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
			model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
			model.train(x_train, y_train)
			models.append(models)
			xval_preds = model.predict(x_test)
			xval_preds = train_scaler.recover(xval_preds, method=standardization)
			hindcasts.append(xval_preds[border])

		x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self.__preprocess_data(data, data.shape[0]- 2*border-1, data.shape[0]-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
		model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
		model.train(x_train, y_train)
		models.append(models)
		xval_preds = model.predict(x_test)
		xval_preds = train_scaler.recover(xval_preds, method=standardization)

		for year in range(border, xval_window): #save hindcasts for years in border, plus center year of first xval window
			hindcasts.append(xval_preds[year])

		hindcasts = np.asarray(hindcasts).reshape(1,-1)[0]
		return hindcasts, models

	def construct_crossvalidated_mme_hindcasts(self, model_type, xval_window=3, hidden_layer_neurons=5, activation='sigm', standardization='minmax', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', pca_variability=-1, W=None, minmax_range=[-1,1] ):
		"""wrapper function for creating cross-validated hindcasts using leave-N-out xval method, either for one point or for multiple points """
		if self.type == 'Single-Point':
			data = np.hstack((self.years, self.observations, self.model_data))
			hindcasts, models = self.__single_point_crossvalidated_hindcasts(data, model_type, xval_window=xval_window, standardization=standardization, pca_variability=pca_variability, hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, minmax_range=minmax_range)
			self.hindcasts[model_type] = hindcasts
			self.hindcast_models[model_type] = models
		if self.type == 'Multi-Point':
			hindcasts, models = [], []
			for i in range(self.lats.shape[0]):
				hindcasts.append([])
				models.append([])
				for j in range(self.lons.shape[0]):
					#print(self.years.shape, self.observations[:,j,i,:].reshape(-1,1).shape, self.model_data[:,j,i,:].transpose().shape)
					data = np.hstack((self.years, self.observations[:,i,j,:].reshape(-1,1), self.model_data[:,i,j,:].transpose())).astype(float)
					output, model_output = self.__single_point_crossvalidated_hindcasts(data, model_type, xval_window=xval_window, standardization=standardization, pca_variability=pca_variability, hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, minmax_range=minmax_range)
					hindcasts[i].append(output)
					models[i].append(model_output)
				hindcasts[i] = np.asarray(hindcasts[i])
			hindcasts = np.asarray(hindcasts)
			hindcasts[self.nanmask,:] = np.nan
			self.hindcasts[model_type] = hindcasts
			self.hindcast_models[model_type] = models

	def __train_single_point_forecast_model(self, data, model_type, standardization='None', pca_variability=-1, hidden_layer_neurons=5, activation='sigm', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', W=None, minmax_range=[-1,1]):
		"""train model on all available data, rather than use leave-n-out xvalidation"""
		x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self.__preprocess_data(data, 0, data.shape[0]-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
		model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
		model.train(x_train, y_train)
		training_forecast = model.predict(x_train)
		training_forecast = train_scaler.recover(training_forecast, method=standardization)
		return training_forecast, model, train_scaler, pca

	def forecast(self, data, model_type, standardization='None', pca_variability=-1, hidden_layer_neurons=5, activation='sigm', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', W=None, minmax_range=[-1,1]):
		"""use models trained on all available training data to make real-time-forecasts based on new data """
		if self.type == 'Single-Point':
			x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self.__preprocess_data(data, 0, data.shape[0]-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range, train_scaler=self.forecast_scalers[model_type], pca=self.forecast_pca[model_type])
			out = self.forecast_models[model_type].predict(x_train)
			self.real_time_forecasts[model_type] = self.forecast_scalers[model_type].recover(out)
			return self.real_time_forecasts[model_type]
		else:
			outs = []
			for i in range(self.lats.shape[0]):
				outs.append([])
				for j in range(self.lons.shape[0]):
					#data = data[:,i,j,:].transpose().astype(float)
					data1 = np.hstack((np.arange(data[:,i,j,:].transpose().shape[0]*2).reshape(data[:,i,j,:].transpose().shape[0], 2), data[:,i,j,:].transpose()))
					x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self.__preprocess_data(data1, 0, data1.shape[0]-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range, train_scaler=self.forecast_scalers[model_type][i][j], pca=self.forecast_pcas[model_type][i][j])
					out = self.forecast_models[model_type][i][j].predict(x_train)
					out = self.forecast_scalers[model_type][i][j].recover(out)
					outs[i].append(out)
				#outs[i] = np.asarray(outs[i])
			outs = np.asarray(outs)
			shape = (outs.shape[0], outs.shape[1], np.max(outs.shape[2:]))
			outs = outs.reshape(shape)
			outs[self.nanmask,:] = np.nan
			self.real_time_forecasts[model_type] = outs
			return self.real_time_forecasts[model_type]


	def train_forecast_model(self, model_type, hidden_layer_neurons=5, activation='sigm', standardization='minmax', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', pca_variability=-1, W=None, minmax_range=[-1,1] ):
		"""wrapper class for training models on all available data for use in making real time forecasts """
		if self.type == 'Single-Point':
			data = np.hstack((self.years, self.observations, self.model_data))
			training_forecasts, model, train_scaler, pca = self.__train_single_point_forecast_model(data, model_type, standardization=standardization, pca_variability=pca_variability, hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, minmax_range=minmax_range)
			self.training_forecasts[model_type] = training_forecasts
			self.forecast_models[model_type] = model
			self.forecast_scalers[model_type] = train_scaler
			self.forecast_pca[model_type] = pca

		if self.type == 'Multi-Point':
			self.observations[:,self.nanmask,:] = np.random.randint(-1000, -900, self.observations.shape)[:,self.nanmask,:]
			training_forecasts, forecast_models, scalers, pcas = [], [], [], []
			for i in range(self.lats.shape[0]):
				training_forecasts.append([])
				forecast_models.append([])
				scalers.append([])
				pcas.append([])
				for j in range(self.lons.shape[0]):
					#print(self.years.shape, self.observations[:,j,i,:].reshape(-1,1).shape, self.model_data[:,j,i,:].transpose().shape)
					data = np.hstack((self.years, self.observations[:,i,j,:].reshape(-1,1), self.model_data[:,i,j,:].transpose())).astype(float)
					training_forecast, model, train_scaler, pca = self.__train_single_point_forecast_model(data, model_type,  standardization=standardization, pca_variability=pca_variability, hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, minmax_range=minmax_range)
					training_forecasts[i].append(training_forecast)
					forecast_models[i].append(model)
					scalers[i].append(train_scaler)
					pcas[i].append(pca)
				training_forecasts[i] = np.asarray(training_forecasts[i])
			training_forecasts = np.asarray(training_forecasts)
			training_forecasts[self.nanmask,:] = np.nan
			self.training_forecasts[model_type] = training_forecasts
			self.forecast_models[model_type] = forecast_models
			self.forecast_scalers[model_type] = scalers
			self.forecast_pcas[model_type] = pcas

	def plot(self, setting='Cross-Validated Hindcasts', point=None, years='NA', methods=None, hindcasts='NA',training_forecasts='NA', metrics='NA', rte_forecasts='NA', mme_members=[], fos=None):
		"""plot results of data, depending on setting.  Multi-point or single point, controlled by 'setting'
		---------------------------------------------------------------------------
			settings:
				'Cross-Validated Hindcasts': plot values of cross validated forecasts (single point, timeline, multi-point, many maps)
				'Real-Time Forecasts': plot values of real time forescasts (single-point, timeline, multipoint, maps)
				'Training Forecasts': plot values of hindcasts produced by models trained on all data (single-point, timeline, multipoint, maps)
				'xval_hindcast_skill': plot skill metrics of cross  validated hindcasts (SP: grid, MP: maps)
				'training_forecast_skill': plot skill metrics of hindcasts produced by models trained on all data (single-point, grid, multipoint, maps)
				'boxplot': plot boxplots of xval hindcast distributions  (SP only)
				'training_forecast_boxplot': plot boxplots of distributions of hindcasts produced by models trained on all data (SP only)
				"Real Time Deterministic Forecast": plot deterministic real time forecasts produced by models trained on all available data
		-------------------------------------------------------------------------"""

		if years == 'NA':
			years = self.years

		for key in self.hindcasts.keys():
			self.hindcasts[key][self.hindcasts[key] == -999] = np.nan

		if self.type=='Single-Point' or point is not None:
			if setting=='Cross-Validated Hindcasts':
				if methods is not None:
					keys = [key for key in methods]
				else:
					keys = [ key for key in self.hindcasts.keys()]
				plt.figure(figsize=(20,10))
				linestyles = ['-.', ':', '-.', ':', '-.', '-.', '--', '--', '-.', ':', '-.', ':', '-.', '-.', '--', '--']
				markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
				colors = ['r', 'g', 'm', 'b', 'k', 'g', 'r', 'g', 'r', 'g', 'm', 'b', 'k', 'g', 'r', 'g']
				ndx = 0
				for key in keys:
					if key == 'Obs':
						obs = self.hindcasts[key] if point is None else self.observations[0][point[0]][point[1]]
						plt.plot(obs, color='b', linestyle='-', label='Obs',marker='o',ms=3,  linewidth=1)
					else:
						data = self.hindcasts[key] if point is None else self.hindcasts[key][point[0]][point[1]]
						plt.plot(data, linestyle=linestyles[ndx], color=colors[ndx], label=key, marker='o',ms=3, linewidth=1)
						ndx += 1

				plt.title('{}{}'.format(setting, '' if point is None else ' - ({},{})'.format(point[0], point[1])))
				plt.xlabel('Year')
				if type(self.years[0,0]) != np.str_:
					plt.xticks(labels=["{0:.0f}".format(i[0]) for i in self.years], ticks=[i for i in range(len(self.years))], rotation=90)
				else:
					plt.xticks(labels=[i[0] for i in self.years], ticks=[i for i in range(len(self.years))], rotation=90)

				plt.ylabel('Precipitation')
				plt.legend()
				plt.show()

			elif setting=='Real Time Forecasts':
				if methods is not None:
					keys = [key for key in methods]
				else:
					keys = [ key for key in self.real_time_forecasts.keys()]
				plt.figure(figsize=(20,10))
				#linestyles = ['-.', ':', '-.', ':', '-.', '-.', '--', '--', '-.', ':', '-.', ':', '-.', '-.', '--', '--']
				#markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
				colors = ['r', 'g', 'm', 'b', 'k', 'g', 'r', 'g', 'r', 'g', 'm', 'b', 'k', 'g', 'r', 'g']
				width = 0.35
				hfl = width / (len(keys) + 1)
				ndx = 1
				if fos is not None:
					obs = np.genfromtxt(fos, dtype=float)
					#obs = self.real_time_forecasts[key] if point is None else self.observations[0][point[0]][point[1]]
					plt.bar(np.arange(len(obs)) - (width /2) + hfl * (ndx-1) , np.squeeze(obs), hfl, label='Obs')

				for key in keys:
					data = self.real_time_forecasts[key] if point is None else self.real_time_forecasts[key][point[0]][point[1]]
					plt.bar(np.arange(len(data)) - (width /2) + hfl * ndx , np.squeeze(data) , hfl,  label=key)
					ndx += 1

				plt.title('{}{}'.format(setting, '' if point is None else ' - ({},{})'.format(point[0], point[1])))
				plt.xlabel('Year')

				plt.ylabel('Precipitation')
				plt.legend()
				plt.show()
			elif setting=='Training Forecasts':
				if methods is not None:
					keys = [key for key in methods]
				else:
					keys = [ key for key in self.training_forecasts.keys()]
				plt.figure(figsize=(20,10))
				linestyles = ['-.', ':', '-.', ':', '-.', '-.', '--', '--', '-.', ':', '-.', ':', '-.', '-.', '--', '--']
				markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
				colors = ['r', 'g', 'm', 'b', 'k', 'g', 'r', 'g', 'r', 'g', 'm', 'b', 'k', 'g', 'r', 'g']
				ndx = 0
				for key in keys:
					if key == 'Obs':
						obs = self.training_forecasts[key] if point is None else self.observations[0][point[0]][point[1]]
						plt.plot(obs, color='b', linestyle='-', label='Obs',marker='o',ms=3,  linewidth=1)
					else:
						data = self.training_forecasts[key] if point is None else self.training_forecasts[key][point[0]][point[1]]
						plt.plot(data, linestyle=linestyles[ndx], color=colors[ndx], label=key, marker='o',ms=3, linewidth=1)
						ndx += 1

				plt.title('{}{}'.format(setting, '' if point is None else ' - ({},{})'.format(point[0], point[1])))
				plt.xlabel('Year')
				if type(self.years[0,0]) != np.str_:
					plt.xticks(labels=["{0:.0f}".format(i[0]) for i in self.years], ticks=[i for i in range(len(self.years))], rotation=90)
				else:
					plt.xticks(labels=[i[0] for i in self.years], ticks=[i for i in range(len(self.years))], rotation=90)

				plt.ylabel('Precipitation')
				plt.legend()
				plt.show()

			elif setting == 'xval_hindcast_skill':
				if metrics == 'NA':
					metrics = [key for key in self.xval_hindcast_metrics.keys()]
				if hindcasts == 'NA':
					hindcasts = [key for key in self.hindcasts.keys()]
				columns = [metric for metric in metrics]
				columns.insert(0, 'Std Dev')
				columns.insert(0, 'Mean')

				rows = copy.deepcopy(mme_members)
				for hindcast in hindcasts:
					if hindcast not in rows and hindcast != 'Obs':
						rows.append(hindcast)
				rows.insert(0,'Observations')
				rows1 = mme_members
				table = []
				for i in range(self.model_data.shape[1]):
					if "Model {}".format(i+1) in rows1 :
						table.append(["{:.2f}".format(np.nanmean(self.model_data[:,i])), "{:.2f}".format(np.nanstd(self.model_data[:,i]))])
				for hindcast in hindcasts:

					if hindcast not in rows1 and hindcast != 'Obs':
						table.append(["{:.2f}".format(np.nanmean(self.hindcasts[hindcast])), "{:.2f}".format(np.nanstd(self.hindcasts[hindcast]))])
				table.insert(0,["{:.2f}".format(np.nanmean(self.observations)), "{:.2f}".format(np.nanstd(self.observations))])

				for model in range(1, len(table)):
					for metric in metrics:
						table[model].append("{:.2f}".format(self.xval_hindcast_metrics[metric][rows[model]]))
				for metric in metrics:
					table[0].append('--')
				hcell, wcell = 0.3, 1.
				hpad, wpad = 0, 0
				fig=plt.figure(figsize=(2*len(rows)*wcell+wpad, 3*len(rows)*hcell+hpad))
				ax = fig.add_subplot(111)
				ax.axis('off')
				#do the table
				the_table = ax.table(cellText=table, rowLabels=rows, colLabels=columns,loc='center')
				the_table.set_fontsize(20)
				the_table.scale(1,4)
				plt.show()

			elif setting == 'training_forecast_skill':
				if metrics == 'NA':
					metrics = [key for key in self.training_forecast_metrics.keys()]
				if training_forecasts == 'NA':
					training_forecasts = [key for key in self.training_forecasts.keys()]
				columns = [metric for metric in metrics]
				columns.insert(0, 'Std Dev')
				columns.insert(0, 'Mean')

				rows = copy.deepcopy(mme_members)
				for training_forecast in training_forecasts:
					if training_forecast not in rows and training_forecast != 'Obs':
						rows.append(training_forecast)
				rows.insert(0,'Observations')
				rows1 = mme_members
				table = []
				for i in range(self.model_data.shape[1]):
					if "Model {}".format(i+1) in rows1 :
						table.append(["{:.2f}".format(np.nanmean(self.model_data[:,i])), "{:.2f}".format(np.nanstd(self.model_data[:,i]))])
				for training_forecast in training_forecasts:

					if training_forecast not in rows1 and training_forecast != 'Obs':
						table.append(["{:.2f}".format(np.nanmean(self.training_forecasts[training_forecast])), "{:.2f}".format(np.nanstd(self.training_forecasts[training_forecast]))])
				table.insert(0,["{:.2f}".format(np.nanmean(self.observations)), "{:.2f}".format(np.nanstd(self.observations))])

				for model in range(1, len(table)):
					for metric in metrics:
						table[model].append("{:.2f}".format(self.training_forecast_metrics[metric][rows[model]]))
				for metric in metrics:
					table[0].append('--')
				hcell, wcell = 0.3, 1.
				hpad, wpad = 0, 0
				fig=plt.figure(figsize=(2*len(rows)*wcell+wpad, 3*len(rows)*hcell+hpad))
				ax = fig.add_subplot(111)
				ax.axis('off')
				#do the table
				the_table = ax.table(cellText=table, rowLabels=rows, colLabels=columns,loc='center')
				the_table.set_fontsize(20)
				the_table.scale(1,4)
				plt.show()


			elif setting=='boxplot':
				if hindcasts == 'NA':
					hindcasts = [key for key in self.hindcasts.keys()]

				data = None
				for key in hindcasts:
					if data is None:
						data = self.hindcasts[key].ravel().reshape(-1,1) if point is None else self.hindcasts[key][point[0]][point[1]]
					else:
						data = np.hstack((data, self.hindcasts[key].ravel().reshape(-1,1) if point is None else self.hindcasts[key][point[0]][point[1]]))
				plt.boxplot(data, whis=255)
				plt.xticks(labels=hindcasts, ticks=[i+1 for i in range(len(hindcasts))] )
				plt.show()
			elif setting == 'training_forecast_boxplot':
				if training_forecasts == 'NA':
					training_forecasts = [key for key in self.training_forecasts.keys()]

				data = None
				for key in training_forecasts:
					if data is None:
						data = self.training_forecasts[key].ravel().reshape(-1,1) if point is None else self.training_forecasts[key][point[0]][point[1]]
					else:
						data = np.hstack((data, self.training_forecasts[key].ravel().reshape(-1,1) if point is None else self.training_forecasts[key][point[0]][point[1]]))
				plt.boxplot(data, whis=255)
				plt.xticks(labels=training_forecasts, ticks=[i+1 for i in range(len(training_forecasts))] )
				plt.show()




		else:
			if setting == 'Cross-Validated Hindcasts':
				if methods is None:
					keys = [key for key in self.hindcasts.keys()]
				else:
					keys = [key for key in methods]
				fig, ax = plt.subplots(nrows=len(years), ncols=len(keys), figsize=(2*len(keys), 4*len(years) ),sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()}) #creates pyplot plotgrid with maps

				states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries',scale='10m',facecolor='none')#setting more variables
				for i in range(len(years)): # for each model, but this is always one because were only doing one model
					for j in range(len(keys)): #for each season
						ax[i][j].set_extent([np.min(self.lons),np.max(self.lons), np.min(self.lats), np.max(self.lats)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
						ax[i][j].add_feature(feature.LAND) #adds predefined cartopy land feature - gets overwritten
						pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
						pl.xlabels_top, pl.ylabels_left, pl.ylabels_right, pl.xlabels_bottom  = False, True, False, True #adds labels to dashed gridlines on left and bottom
						pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff

						ax[i][j].add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

						if j == 0: # if this is the leftmost plot
							ax[i][j].text(-0.25, 0.5, 'Year {}'.format(str(years[i])),rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes) #print title vertially on the left side
						if i == 0: # if this is the top plot
							ax[i][j].set_title("{} Cross-Validated hindcast".format(keys[j])) #print season on top of each plot

						CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.hindcasts[keys[j]][:,:,i], cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else

						axins = inset_axes(ax[i][j], width="5%", height="100%",  loc='center right', bbox_to_anchor=(0., 0., 1.15, 1), bbox_transform=ax[i][j].transAxes, borderpad=0.1,) #describes where colorbar should go
						cbar_bdet = fig.colorbar(CS1, ax=ax[i][j],  cax=axins, orientation='vertical', pad = 0.02) #add colorbar based on hindcast data
						cbar_bdet.set_label('Rainfall (mm)')# add colorbar label
				plt.show()
			elif setting == "xval_hindcast_skill":
				if metrics == 'NA':
					metrics = [key for key in self.xval_hindcast_metrics.keys()]
				if hindcasts == 'NA':
					hindcasts = [key for key in self.hindcasts.keys()]
				keys = [key for key in metrics ]
				forkeys = [key for key in hindcasts]
				fig, ax = plt.subplots(nrows=len(forkeys), ncols=len(keys), figsize=(4*len(forkeys), 4*len(keys)),sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()}) #creates pyplot plotgrid with maps

				states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries',scale='10m',facecolor='none')#setting more variables
				for i in range(len(forkeys)): # for each model, but this is always one because were only doing one model
					for j in range(len(keys)): #for each season
						ax[i][j].set_extent([np.min(self.lons),np.max(self.lons), np.min(self.lats), np.max(self.lats)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
						ax[i][j].add_feature(feature.LAND) #adds predefined cartopy land feature - gets overwritten
						pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
						pl.xlabels_top, pl.ylabels_left, pl.ylabels_right, pl.xlabels_bottom  = False, False, False, False #adds labels to dashed gridlines on left and bottom
						pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff

						ax[i][j].add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

						if j == 0: # if this is the leftmost plot
							ax[i][j].text(-0.25, 0.5, 'hindcast: {}'.format(str(forkeys[i])),rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes) #print title vertially on the left side
						if i == 0: # if this is the top plot
							ax[i][j].set_title("Metric {}".format(keys[j])) #print season on top of each plot

						if keys[j] in ['SpearmanCoef', 'PearsonCoef']:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.xval_hindcast_metrics[keys[j]][forkeys[i]], vmin=-1, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
						elif keys[j] in ['IOA']:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.xval_hindcast_metrics[keys[j]][forkeys[i]], vmin=0, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
						elif keys[j] in ['RMSE']:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.xval_hindcast_metrics[keys[j]][forkeys[i]], cmap='Reds') #adds probability of below normal where below normal is most likely  and nan everywhere else
						else:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.xval_hindcast_metrics[keys[j]][forkeys[i]], cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else

						axins = inset_axes(ax[i][j], width="100%", height="5%",  loc='lower center', bbox_to_anchor=(0., -0.15, 1, 1), bbox_transform=ax[i][j].transAxes, borderpad=0.1,) #describes where colorbar should go
						cbar_bdet = fig.colorbar(CS1, ax=ax[i][j],  cax=axins, orientation='horizontal', pad = 0.02) #add colorbar based on hindcast data
				plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
				plt.show()
			elif setting == "training_forecast_skill":
				if metrics == 'NA':
					metrics = [key for key in self.training_forecast_metrics.keys()]
				if training_forecasts == 'NA':
					training_forecasts = [key for key in self.training_forecasts.keys()]
				keys = [key for key in metrics ]
				forkeys = [key for key in training_forecasts]
				fig, ax = plt.subplots(nrows=len(forkeys), ncols=len(keys), figsize=(4*len(forkeys), 4*len(keys)),sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()}) #creates pyplot plotgrid with maps

				states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries',scale='10m',facecolor='none')#setting more variables
				for i in range(len(forkeys)): # for each model, but this is always one because were only doing one model
					for j in range(len(keys)): #for each season
						ax[i][j].set_extent([np.min(self.lons),np.max(self.lons), np.min(self.lats), np.max(self.lats)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
						ax[i][j].add_feature(feature.LAND) #adds predefined cartopy land feature - gets overwritten
						pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
						pl.xlabels_top, pl.ylabels_left, pl.ylabels_right, pl.xlabels_bottom  = False, False, False, False #adds labels to dashed gridlines on left and bottom
						pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff

						ax[i][j].add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

						if j == 0: # if this is the leftmost plot
							ax[i][j].text(-0.25, 0.5, 'hindcast: {}'.format(str(forkeys[i])),rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes) #print title vertially on the left side
						if i == 0: # if this is the top plot
							ax[i][j].set_title("Metric {}".format(keys[j])) #print season on top of each plot

						if keys[j] in ['SpearmanCoef', 'PearsonCoef']:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.training_forecast_metrics[keys[j]][forkeys[i]], vmin=-1, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
						elif keys[j] in ['IOA']:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.training_forecast_metrics[keys[j]][forkeys[i]], vmin=0, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
						elif keys[j] in ['RMSE']:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.training_forecast_metrics[keys[j]][forkeys[i]], cmap='Reds') #adds probability of below normal where below normal is most likely  and nan everywhere else
						else:
							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.training_forecast_metrics[keys[j]][forkeys[i]], cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else

						axins = inset_axes(ax[i][j], width="100%", height="5%",  loc='lower center', bbox_to_anchor=(0., -0.15, 1, 1), bbox_transform=ax[i][j].transAxes, borderpad=0.1,) #describes where colorbar should go
						cbar_bdet = fig.colorbar(CS1, ax=ax[i][j],  cax=axins, orientation='horizontal', pad = 0.02) #add colorbar based on hindcast data
				plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
				plt.show()
			elif setting == "Real Time Deterministic Forecast":
				if rte_forecasts == 'NA':
					rte_forecasts = [key for key in self.real_time_forecasts.keys()]

				years = [i for i in range( self.real_time_forecasts[rte_forecasts[0]].shape[2])]
				fig, ax = plt.subplots(nrows=len(years), ncols=len(rte_forecasts), figsize=(10*len(years), 40*len(rte_forecasts)),sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()}) #creates pyplot plotgrid with maps
				if len(years) == 1:
					ax = [ax]
				if len(rte_forecasts) == 1:
					ax = [ax]
				states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries',scale='10m',facecolor='none')#setting more variables
				for i in range(len(years)): # for each model, but this is always one because were only doing one model
					for j in range(len(rte_forecasts)): #for each season
						ax[i][j].set_extent([np.min(self.lons),np.max(self.lons), np.min(self.lats), np.max(self.lats)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
						ax[i][j].add_feature(feature.LAND) #adds predefined cartopy land feature - gets overwritten
						pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
						pl.xlabels_top, pl.ylabels_left, pl.ylabels_right, pl.xlabels_bottom  = False, False, False, False #adds labels to dashed gridlines on left and bottom
						pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff

						ax[i][j].add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

						if j == 0: # if this is the leftmost plot
							ax[i][j].text(-0.25, 0.5, 'Year: {}'.format(str(years[i])),rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes) #print title vertially on the left side
						if i == 0: # if this is the top plot
							ax[i][j].set_title("{}".format(rte_forecasts[j])) #print season on top of each plot

							CS1 = ax[i][j].pcolormesh(np.linspace(self.lons[0], self.lons[-1],num=self.lons.shape[0]), np.linspace(self.lats[0], self.lats[-1], num=self.lats.shape[0]), self.real_time_forecasts[rte_forecasts[j]][:,:,years[i]], cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else

						axins = inset_axes(ax[i][j], width="100%", height="5%",  loc='lower center', bbox_to_anchor=(0., -0.15, 1, 1), bbox_transform=ax[i][j].transAxes, borderpad=0.1,) #describes where colorbar should go
						cbar_bdet = fig.colorbar(CS1, ax=ax[i][j],  cax=axins, orientation='horizontal', pad = 0.02) #add colorbar based on hindcast data
				plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
				plt.show()


	def __index_of_agreement(self, y, x):
		"""implements index of agreement metric"""
		return 1 - np.sum((y-x)**2) / np.sum(  (np.abs(y - np.nanmean(x)) + np.abs(x - np.nanmean(x)))**2)

	def IOA(self, setting='xval_hindcast_metrics'):
		"""computes IOA for each point"""
		if setting == 'xval_hindcast_metrics':
			if 'IOA' not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics['IOA'] = {}
			if self.type=='Single-Point':
				for key in self.hindcasts.keys():
					self.xval_hindcast_metrics['IOA'][key] = self.__index_of_agreement(self.observations, self.hindcasts[key])
			else:
				for key in self.hindcasts.keys():
					self.hindcasts[key][self.nanmask,:] = -999
					met = []
					for i in range(self.hindcasts[key].shape[0]):
						met.append([])
						for j in range(self.hindcasts[key].shape[1]):
							met[i].append(self.__index_of_agreement(self.observations[:,i,j,:].reshape(-1,1), self.hindcasts[key][i,j,:]))

					self.xval_hindcast_metrics['IOA'][key] = np.asarray(met)
					self.xval_hindcast_metrics['IOA'][key][self.nanmask] = np.nan
		elif setting == 'training_forecast_metrics':
			if 'IOA' not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics['IOA'] = {}
			if self.type=='Single-Point':
				for key in self.training_forecasts.keys():
					self.training_forecast_metrics['IOA'][key] = self.__index_of_agreement(self.observations, self.training_forecasts[key])
			else:
				for key in self.hindcasts.keys():
					self.training_forecasts[key][self.nanmask,:] = -999
					met = []
					for i in range(self.training_forecasts[key].shape[0]):
						met.append([])
						for j in range(self.training_forecasts[key].shape[1]):
							met[i].append(self.__index_of_agreement(self.observations[:,i,j,:].reshape(-1,1), self.training_forecasts[key][i,j,:]))

					self.training_forecast_metrics['IOA'][key] = np.asarray(met)
					self.training_forecast_metrics['IOA'][key][self.nanmask] = np.nan


	def MAE(self, setting='xval_hindcast_metrics'):
		"""compute mean absolute error for each point"""
		if setting == 'xval_hindcast_metrics':
			if 'MAE' not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics['MAE'] = {}
			if self.type=='Single-Point':
				for key in self.hindcasts.keys():
					self.xval_hindcast_metrics['MAE'][key] = mean_absolute_error(self.observations, self.hindcasts[key])
			else:
				for key in self.hindcasts.keys():
					self.hindcasts[key][self.nanmask,:] = -999
					met = []
					for i in range(self.hindcasts[key].shape[0]):
						met.append([])
						for j in range(self.hindcasts[key].shape[1]):
							met[i].append(mean_absolute_error(self.observations[:,i,j,:].reshape(-1,1), self.hindcasts[key][i,j,:]))

					self.xval_hindcast_metrics['MAE'][key] = np.asarray(met)
					self.xval_hindcast_metrics['MAE'][key][self.nanmask] = np.nan
		elif setting == 'training_forecast_metrics':
			if 'MAE' not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics['MAE'] = {}
			if self.type=='Single-Point':
				for key in self.training_forecasts.keys():
					self.training_forecast_metrics['MAE'][key] = mean_absolute_error(self.observations, self.training_forecasts[key])
			else:
				for key in self.training_forecasts.keys():
					self.training_forecasts[key][self.nanmask,:] = -999
					met = []
					for i in range(self.training_forecasts[key].shape[0]):
						met.append([])
						for j in range(self.training_forecasts[key].shape[1]):
							met[i].append(mean_absolute_error(self.observations[:,i,j,:].reshape(-1,1), self.training_forecasts[key][i,j,:]))

					self.training_forecast_metrics['MAE'][key] = np.asarray(met)
					self.training_forecast_metrics['MAE'][key][self.nanmask] = np.nan

	def MSE(self, squared=True, setting='xval_hindcast_metrics'):
		"""compute mean squared error (or Root Mean Squared Error) for each point """
		if setting == 'xval_hindcast_metrics':
			key_name = 'MSE' if squared else 'RMSE'
			if key_name not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics[key_name] = {}
			if self.type=='Single-Point':
				for key in self.hindcasts.keys():
					self.xval_hindcast_metrics[key_name][key] = mean_squared_error(self.observations, self.hindcasts[key], squared=squared)
			else:
				for key in self.hindcasts.keys():
					self.hindcasts[key][self.nanmask,:] = -999
					met = []
					for i in range(self.hindcasts[key].shape[0]):
						met.append([])
						for j in range(self.hindcasts[key].shape[1]):
							met[i].append(mean_squared_error(self.observations[:,i,j,:].reshape(-1,1), self.hindcasts[key][i,j,:], squared=squared))

					self.xval_hindcast_metrics[key_name][key] = np.asarray(met)
					self.xval_hindcast_metrics[key_name][key][self.nanmask] = np.nan
		elif setting == 'training_forecast_metrics':
			key_name = 'MSE' if squared else 'RMSE'
			if key_name not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics[key_name] = {}
			if self.type=='Single-Point':
				for key in self.training_forecasts.keys():
					self.training_forecast_metrics[key_name][key] = mean_squared_error(self.observations, self.training_forecasts[key], squared=squared)
			else:
				for key in self.training_forecasts.keys():
					self.training_forecasts[key][self.nanmask,:] = -999
					met = []
					for i in range(self.training_forecasts[key].shape[0]):
						met.append([])
						for j in range(self.training_forecasts[key].shape[1]):
							met[i].append(mean_squared_error(self.observations[:,i,j,:].reshape(-1,1), self.training_forecasts[key][i,j,:], squared=squared))

					self.training_forecast_metrics[key_name][key] = np.asarray(met)
					self.training_forecast_metrics[key_name][key][self.nanmask] = np.nan

	def Pearson(self, setting='xval_hindcast_metrics'):
		"""computes pearson coefficient and pearson p-score for each point """
		if setting == 'xval_hindcast_metrics':
			if 'PearsonCoef' not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics['PearsonCoef'] = {}
			if 'PearsonP' not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics['PearsonP']= {}
			if self.type=='Single-Point':
				for key in self.hindcasts.keys():
					try:
						self.xval_hindcast_metrics['PearsonCoef'][key] = stats.pearsonr(np.squeeze(self.observations), np.squeeze(self.hindcasts[key]))[0]
						self.xval_hindcast_metrics['PearsonP'][key] = stats.pearsonr(np.squeeze(self.observations), np.squeeze(self.hindcasts[key]))[1]
					except:
						print(np.squeeze(self.observations))
						print(np.squeeze(self.hindcasts[key]))
			else:
				for key in self.hindcasts.keys():
					#self.hindcasts[key][self.nanmask,:] = -999
					met, r = [], []
					for i in range(self.hindcasts[key].shape[0]):
						met.append([])
						r.append([])
						for j in range(self.hindcasts[key].shape[1]):
							try:
								met[i].append(stats.pearsonr(np.squeeze(self.observations[:,i,j,:].astype(float)), np.squeeze(self.hindcasts[key][i,j,:].astype(float)))[0])
								r[i].append(stats.pearsonr(np.squeeze(self.observations[:,i,j,:].astype(float)), np.squeeze(self.hindcasts[key][i,j,:].astype(float)))[1])
							except:
								print(np.squeeze(self.observations[:,i,j,:].astype(float)))
								print(np.squeeze(self.hindcasts[key][i,j,:].astype(float)))
					self.xval_hindcast_metrics['PearsonCoef'][key] = np.asarray(met)
					self.xval_hindcast_metrics['PearsonCoef'][key][self.nanmask] = np.nan
					self.xval_hindcast_metrics['PearsonP'][key] = np.asarray(r)
					self.xval_hindcast_metrics['PearsonP'][key][self.nanmask] = np.nan
		elif setting == 'training_forecast_metrics':
			if 'PearsonCoef' not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics['PearsonCoef'] = {}
			if 'PearsonP' not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics['PearsonP']= {}
			if self.type=='Single-Point':
				for key in self.training_forecasts.keys():
					self.training_forecast_metrics['PearsonCoef'][key] = stats.pearsonr(np.squeeze(self.observations), np.squeeze(self.training_forecasts[key]))[0]
					self.training_forecast_metrics['PearsonP'][key] = stats.pearsonr(np.squeeze(self.observations), np.squeeze(self.training_forecasts[key]))[1]
			else:
				for key in self.training_forecasts.keys():
					#self.training_forecasts[key][self.nanmask,:] = -999
					met, r = [], []
					for i in range(self.training_forecasts[key].shape[0]):
						met.append([])
						r.append([])
						for j in range(self.training_forecasts[key].shape[1]):
							met[i].append(stats.pearsonr(np.squeeze(self.observations[:,i,j,:].astype(float)), np.squeeze(self.training_forecasts[key][i,j,:].astype(float)))[0])
							r[i].append(stats.pearsonr(np.squeeze(self.observations[:,i,j,:].astype(float)), np.squeeze(self.training_forecasts[key][i,j,:].astype(float)))[1])
					self.training_forecast_metrics['PearsonCoef'][key] = np.asarray(met)
					self.training_forecast_metrics['PearsonCoef'][key][self.nanmask] = np.nan
					self.training_forecast_metrics['PearsonP'][key] = np.asarray(r)
					self.training_forecast_metrics['PearsonP'][key][self.nanmask] = np.nan

	def Spearman(self, setting='xval_hindcast_metrics'):
		"""computes spearman coefficient, and spearman p-value for each point """
		if setting == 'xval_hindcast_metrics':
			if 'SpearmanCoef' not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics['SpearmanCoef'] = {}
			if 'SpearmanP' not in self.xval_hindcast_metrics.keys():
				self.xval_hindcast_metrics['SpearmanP']= {}

			if self.type=='Single-Point':
				for key in self.hindcasts.keys():
					self.xval_hindcast_metrics['SpearmanCoef'][key] = stats.spearmanr(self.observations, self.hindcasts[key])[0]
					self.xval_hindcast_metrics['SpearmanP'][key] = stats.spearmanr(self.observations, self.hindcasts[key])[1]

			else:
				for key in self.hindcasts.keys():
					self.hindcasts[key][self.nanmask,:] = -999
					met, r = [], []
					for i in range(self.hindcasts[key].shape[0]):
						met.append([])
						r.append([])
						for j in range(self.hindcasts[key].shape[1]):
							met[i].append(stats.spearmanr(np.squeeze(self.observations[:,i,j,:].reshape(-1,1).astype(float)), np.squeeze(self.hindcasts[key][i,j,:].astype(float)))[0])
							r[i].append(stats.spearmanr(np.squeeze(self.observations[:,i,j,:].reshape(-1,1).astype(float)), np.squeeze(self.hindcasts[key][i,j,:].astype(float)))[1])


					self.xval_hindcast_metrics['SpearmanCoef'][key] = np.asarray(met)
					self.xval_hindcast_metrics['SpearmanCoef'][key][self.nanmask] = np.nan
					self.xval_hindcast_metrics['SpearmanP'][key] = np.asarray(r)
					self.xval_hindcast_metrics['SpearmanP'][key][self.nanmask] = np.nan
		elif setting == 'training_forecast_metrics':
			if 'SpearmanCoef' not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics['SpearmanCoef'] = {}
			if 'SpearmanP' not in self.training_forecast_metrics.keys():
				self.training_forecast_metrics['SpearmanP']= {}

			if self.type=='Single-Point':
				for key in self.training_forecasts.keys():
					self.training_forecast_metrics['SpearmanCoef'][key] = stats.spearmanr(self.observations, self.training_forecasts[key])[0]
					self.training_forecast_metrics['SpearmanP'][key] = stats.spearmanr(self.observations, self.training_forecasts[key])[1]

			else:
				for key in self.training_forecasts.keys():
					self.training_forecasts[key][self.nanmask,:] = -999
					met, r = [], []
					for i in range(self.training_forecasts[key].shape[0]):
						met.append([])
						r.append([])
						for j in range(self.training_forecasts[key].shape[1]):
							met[i].append(stats.spearmanr(np.squeeze(self.observations[:,i,j,:].reshape(-1,1).astype(float)), np.squeeze(self.training_forecasts[key][i,j,:].astype(float)))[0])
							r[i].append(stats.spearmanr(np.squeeze(self.observations[:,i,j,:].reshape(-1,1).astype(float)), np.squeeze(self.training_forecasts[key][i,j,:].astype(float)))[1])


					self.training_forecast_metrics['SpearmanCoef'][key] = np.asarray(met)
					self.training_forecast_metrics['SpearmanCoef'][key][self.nanmask] = np.nan
					self.training_forecast_metrics['SpearmanP'][key] = np.asarray(r)
					self.training_forecast_metrics['SpearmanP'][key][self.nanmask] = np.nan

	def export(self, fname):
		"""saves 1D data to csv/tsv format - 2D not implemented """
		if self.type == 'Single-Point':
			data = np.hstack((self.years.reshape(-1,1), self.observations.reshape(-1,1)))
			keys = [key  for key in self.hindcasts.keys() if key != 'Obs']
			header = 'Year,Observations,' + ','.join(keys)
			for key in self.hindcasts.keys():
				if key != 'Obs':
					data = np.hstack((data, self.hindcasts[key].reshape(-1,1)))
			np.savetxt(fname, data, delimiter=',', header=header, comments='')
		else:
			print('exporting ncdf not yet available')


	def save(self, fname='test.mme'):
		"""saves data associated with MME model, but not the models themselves"""
		tosave = {}
		for key in vars(self).keys():
			if key != 'models':
				tosave[key] = vars(self)[key]
		pickle.dump(tosave, open(fname, 'wb'))

		return 1

	@classmethod
	def load(self, fname='test.mme'):
		"""loads previosly saved MME from file- must re-run training functions in order to access models"""
		try:
			print('Dill Pickle undetected: Just saving data, forgetting models. Try "conda install dill"')
			toload = pickle.load(open(fname, 'rb'))
			new = MME()
			for key in toload.keys():
				exec("new.{} = toload['{}']".format(key, key))
		except:
			print('Dill Pickle undetected: Just saving data, forgetting models. Try "conda install dill"')
			new = pickle.load(open(fname, 'rb'))
		return new
