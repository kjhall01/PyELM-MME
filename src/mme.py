from .scaler import *
from .spm import *
from .svd import *
from .ensemblemean import *
from .casts import Cast

import numpy as np
import xarray as xr
import os
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pickle, copy
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA


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
	#------------------ Stuff for Manipulating MME Data ------------------------#
	def __init__(self, reader_ret, verbose=False):
		self.type, self.hindcasts = reader_ret
		self.verbose = verbose
		if 'Obs' not in self.hindcasts.available_data():
			assert False, 'Must provide Y input data (historical observations) to initialize MME'
		self.hindcasts.get_nanmask() #1D data should never have nans
		self.hindcasts.replace_nans()

	def add_forecast(self, reader_ret):
		newfcst_type, self.forecasts = reader_ret
		self.training_forecasts = copy.deepcopy(self.hindcasts)
		self.forecasts.lats, self.forecasts.lons = self.training_forecasts.lats, self.training_forecasts.lons
		assert newfcst_type == self.type, 'must have 2d data for a 2d mme, and 1d for 1d lol'
		return True

	def _preprocess(self, data, crossval_start, crossval_end, standardization=None, pca_variability=-1, minmax_range=[-1,1], train_scaler=None, test_scaler=None, pca=None):
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
			train_data = train_scaler.transform(train_data)
		test_data = train_scaler.transform(test_data)
		x_train, y_train = train_data[:, 2:], train_data[:, 1].reshape(-1,1)
		if x_train.shape[0] == 1:
			x_train = x_train.reshape(1,-1)
		x_test, y_test = test_data[:, 2:], test_data[:, 1].reshape(-1,1)
		return x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca

	def export_csv(self, fname, fcst='hindcasts', obs=True):
		"""saves 1D data to csv/tsv format - 2D not implemented """
		casts = getattr(self, fcst)
		if obs:
			data = np.hstack((casts.years.reshape(-1,1), casts.data['Obs'].reshape(-1,1)))
		else:
			data = casts.years.reshape(-1,1)
		header = 'Year,Observations'
		for key in casts.available_mmes():
			data = np.hstack((data, casts.data[key].reshape(-1,1)))
			header = header + ',{}'.format(key)
		for key in casts.available_members():
			data = np.hstack((data, casts.data[key].reshape(-1,1)))
			header = header + ',{}'.format(key)
		np.savetxt(fname, data, delimiter=',', header=header, comments='', fmt='%10.5f')
		if self.verbose:
			print('Saved {} data to {}'.format(fcst, fname))

	def export_ncdf(self, fname, fcst='hindcasts'):
		casts = getattr(self, fcst)
		data_vars = {}
		for key in casts.available_data():
			data_vars[key] = (['latitude', 'longitude', 'time'], casts.data[key])
		coords = dict(
			latitude=(['latitude'], np.squeeze(casts.lats['Obs']) if 'Obs' in casts.lats.keys() else np.squeeze(casts.lats[[key for key in self.lats.keys()][0]])),
			longitude=(['longitude'], np.squeeze(casts.lons['Obs']) if 'Obs' in casts.lons.keys() else np.squeeze(casts.lons[[key for key in self.lons.keys()][0]])),
			time=(['time'], np.squeeze(casts.years))
		)
		DS = xr.Dataset(data_vars=data_vars, coords=coords)
		DS.to_netcdf(path=fname)
		if self.verbose:
			print('saved to {}'.format(fname))

	#------------------ Stuff for Making Xval Hindcast ------------------------#
	def train_mmes(self, mme_methodologies, args):
		if self.verbose:
			print('\nComputing MMEs')
		for method in mme_methodologies:
			if method == 'MLR':
				self.cross_validate(method, xval_window=args['mlr_xval_window'], standardization=args['mlr_standardization'], fit_intercept=args['mlr_fit_intercept'])
			elif method == 'ELM':
				self.cross_validate(method, xval_window=args['elm_xval_window'], hidden_layer_neurons=args['elm_hidden_layer_neurons'], activation=args['elm_activation'], standardization=args['elm_standardization'], minmax_range=args['elm_minmax_range'])
			elif method == 'EM':
				#self.cross_validate(method, xval_window=args['em_xval_window'], standardization=None)
				model_data = []
				for key in self.hindcasts.available_members():
					model_data.append(self.hindcasts.data[key])
				model_data = np.asarray(model_data)
				self.hindcasts.add_data('EM', np.nanmean(model_data, axis=0), mm=False)
				self.hindcasts.add_lats('EM', self.hindcasts.lats['Obs'])
				self.hindcasts.add_lons('EM', self.hindcasts.lons['Obs'])
				if self.verbose:
					print('EM [' + 25*'*' + ']' ) #calculating EM during input now
			else:
				assert False, 'invalid mme method {}'.format(method)

	def _xval1D(self, data, model_type, xval_window=3, standardization='None', pca_variability=-1, hidden_layer_neurons=5, activation='sigm', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', W=None, minmax_range=[-1,1]):
		"""train and evaluate model of type model_type using "leave-N-out" cross-validation"""
		border = xval_window // 2
		models, hindcasts = [], []
		x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self._preprocess(data, 0, xval_window-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
		model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
		model.train(x_train, y_train)
		models.append(models)
		xval_preds = model.predict(x_test)
		xval_preds = train_scaler.recover(xval_preds, method=standardization)

		for year in range(0, border+1): #save hindcasts for years in border, plus center year of first xval window
			hindcasts.append(xval_preds[year])

		#now get hindcasts for all xval windows until last
		for year in range(border+1, data.shape[0]-border-1):
			x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self._preprocess(data, year-border, year+border, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
			model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
			model.train(x_train, y_train)
			models.append(models)
			xval_preds = model.predict(x_test)
			xval_preds = train_scaler.recover(xval_preds, method=standardization)
			hindcasts.append(xval_preds[border])

		x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self._preprocess(data, data.shape[0]- 2*border-1, data.shape[0]-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
		model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
		model.train(x_train, y_train)
		models.append(models)
		xval_preds = model.predict(x_test)
		xval_preds = train_scaler.recover(xval_preds, method=standardization)

		for year in range(border, xval_window): #save hindcasts for years in border, plus center year of first xval window
			hindcasts.append(xval_preds[year])

		hindcasts = np.asarray(hindcasts).reshape(1,-1)[0]
		return hindcasts, models

	def cross_validate(self,  model_type, fcst='hindcasts', xval_window=3, hidden_layer_neurons=5, activation='sigm', standardization='minmax', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', pca_variability=-1, W=None, minmax_range=[-1,1] ):
		"""wrapper function for creating cross-validated hindcasts using leave-N-out xval method, either for one point or for multiple points """
		cast = getattr(self, fcst)
		key = cast.available_data()[0]

		total = cast.lats['Obs'].shape[0] *  cast.lons['Obs'].shape[0]
		steps = [total / 25.0 * i for i in range(26)]
		ndx, step_ndx = 0, 0
		if self.verbose:
			print(model_type + ' [' + ' '*25 + ']', end='\r')
		for i in range(cast.lats['Obs'].shape[0]):
			for j in range(cast.lons['Obs'].shape[0]):
				if ndx > steps[step_ndx] and self.verbose:
					print(model_type + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
					step_ndx += 1
				ndx += 1
				lat_ndx, lon_ndx = i if cast.lats['Obs'].shape[0] > 1 else -1, j if cast.lons['Obs'].shape[0] > 1 else -1
				data = cast.assemble_training_data(lat_ndx=lat_ndx, lon_ndx=lon_ndx)
				hindcasts, models = self._xval1D(data, model_type, xval_window=xval_window, standardization=standardization, pca_variability=pca_variability, hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, minmax_range=minmax_range)
				cast.add_data(model_type, hindcasts, lat_ndx=lat_ndx, lon_ndx=lon_ndx, mm=False)
				cast.add_lats(model_type, cast.lats['Obs'])
				cast.add_lons(model_type, cast.lons['Obs'])
				cast.add_years( np.squeeze(cast.years))

				for k in range(hindcasts.shape[0]):
					cast.add_spm(model_type, models[k], lat_ndx=lat_ndx, lon_ndx=lon_ndx, year_ndx=k )
				cast.replace_nans()
		if self.verbose:
			print(model_type + ' [' + '*'*25+ ']')


	#------------------ Stuff for Evaluating Skill ------------------------#
	def measure_skill(self, metrics):
		if self.verbose:
			print('\nAnalyzing Skill')
		spearman_flag, pearson_flag = 0, 0
		for metric in metrics:
			if self.verbose:
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
				assert False, 'Invalid Metric {}'.format(metric)

	def __index_of_agreement(self, y, x):
		"""implements index of agreement metric"""
		return 1 - np.sum((y-x)**2) / np.sum(  (np.abs(y - np.nanmean(x)) + np.abs(x - np.nanmean(x)))**2)

	def IOA(self, fcst='hindcasts'):
		"""computes IOA for each point"""
		cast = getattr(self, fcst)
		for key in cast.available_data():
			cast.replace_nans()
			total = cast.lats['Obs'].shape[0] *  cast.lons['Obs'].shape[0]
			steps = [total / 25.0 * i for i in range(26)]
			ndx, step_ndx = 0, 0
			if self.verbose:
				print('  '+ key + ' [' + ' '*25 + ']', end='\r')
			for i in range(cast.lats[key].shape[0]):
				for j in range(cast.lons[key].shape[0]):
					if ndx > steps[step_ndx] and self.verbose:
						print('  '+key + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
						step_ndx += 1
					ndx += 1
					obs = cast.data['Obs'].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data['Obs'][i,j,:].reshape(-1,1)
					comp = cast.data[key].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data[key][i,j,:].reshape(-1,1)
					ioa = np.asarray([self.__index_of_agreement(obs, comp)])
					cast.save_point_skill(key, 'IOA', ioa, lat_ndx=i, lon_ndx=j)
			cast.mask_nans_skill()
			if self.verbose:
				print('  '+key + ' [' + '*'*25+ ']')

	def MAE(self, fcst='hindcasts'):
		"""compute mean absolute error for each point"""
		cast = getattr(self, fcst)
		for key in cast.available_data():
			cast.replace_nans()
			total = cast.lats['Obs'].shape[0] *  cast.lons['Obs'].shape[0]
			steps = [total / 25.0 * i for i in range(26)]
			ndx, step_ndx = 0, 0
			if self.verbose:
				print('  '+key + ' [' + ' '*25 + ']', end='\r')
			for i in range(cast.lats[key].shape[0]):
				for j in range(cast.lons[key].shape[0]):
					if ndx > steps[step_ndx] and self.verbose:
						print('  '+key + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
						step_ndx += 1
					ndx += 1
					obs = cast.data['Obs'].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data['Obs'][i,j,:].reshape(-1,1)
					comp = cast.data[key].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data[key][i,j,:].reshape(-1,1)
					mae = np.asarray([mean_absolute_error(obs, comp)])
					cast.save_point_skill(key, 'MAE', mae, lat_ndx=i, lon_ndx=j)
			cast.mask_nans_skill()
			if self.verbose:
				print('  '+key + ' [' + '*'*25+ ']')

	def MSE(self, fcst='hindcasts', squared=True):
		"""compute mean squared error (or Root Mean Squared Error) for each point """
		cast = getattr(self, fcst)
		save_key = 'MSE' if squared else 'RMSE'

		for key in cast.available_data():
			total = cast.lats['Obs'].shape[0] *  cast.lons['Obs'].shape[0]
			steps = [total / 25.0 * i for i in range(26)]
			ndx, step_ndx = 0, 0
			if self.verbose:
				print('  '+key + ' [' + ' '*25 + ']', end='\r')
			cast.replace_nans()
			for i in range(cast.lats[key].shape[0]):
				for j in range(cast.lons[key].shape[0]):
					if ndx > steps[step_ndx] and self.verbose:
						print('  '+key + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
						step_ndx += 1
					ndx += 1
					obs = cast.data['Obs'].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data['Obs'][i,j,:].reshape(-1,1)
					comp = cast.data[key].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data[key][i,j,:].reshape(-1,1)
					mae = np.asarray([mean_squared_error(obs, comp, squared=squared)])
					cast.save_point_skill(key, save_key, mae, lat_ndx=i, lon_ndx=j)
			cast.mask_nans_skill()
			if self.verbose:
				print('  '+key + ' [' + '*'*25+ ']')

	def Pearson(self, fcst='hindcasts'):
		"""computes pearson coefficient and pearson p-score for each point """
		cast = getattr(self, fcst)
		for key in cast.available_data():
			cast.replace_nans()
			total = cast.lats['Obs'].shape[0] *  cast.lons['Obs'].shape[0]
			steps = [total / 25.0 * i for i in range(26)]
			ndx, step_ndx = 0, 0
			if self.verbose:
				print('  '+key + ' [' + ' '*25 + ']', end='\r')
			for i in range(cast.lats[key].shape[0]):
				for j in range(cast.lons[key].shape[0]):
					if ndx > steps[step_ndx] and self.verbose:
						print('  '+key + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
						step_ndx += 1
					ndx += 1
					obs = cast.data['Obs'].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data['Obs'][i,j,:].reshape(-1,1)
					comp = cast.data[key].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data[key][i,j,:].reshape(-1,1)
					coef, p = stats.pearsonr(np.squeeze(obs).astype(float), np.squeeze(comp).astype(float))
					cast.save_point_skill(key, 'PearsonCoef', np.asarray([coef]), lat_ndx=i, lon_ndx=j)
					cast.save_point_skill(key, 'PearsonP', np.asarray([p]), lat_ndx=i, lon_ndx=j)
			cast.mask_nans_skill()
			if self.verbose:
				print('  '+key + ' [' + '*'*25+ ']')

	def Spearman(self, fcst='hindcasts'):
		"""computes spearman coefficient, and spearman p-value for each point """
		cast = getattr(self, fcst)
		for key in cast.available_data():
			cast.replace_nans()
			total = cast.lats['Obs'].shape[0] *  cast.lons['Obs'].shape[0]
			steps = [total / 25.0 * i for i in range(26)]
			ndx, step_ndx = 0, 0
			if self.verbose:
				print('  '+key + ' [' + ' '*25 + ']', end='\r')
			for i in range(cast.lats[key].shape[0]):
				for j in range(cast.lons[key].shape[0]):
					if ndx > steps[step_ndx] and self.verbose:
						print('  '+key + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
						step_ndx += 1
					ndx += 1
					obs = cast.data['Obs'].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data['Obs'][i,j,:].reshape(-1,1)
					comp = cast.data[key].reshape(-1,1) if cast.lats[key].shape[0] == 1 and cast.lons[key].shape[0] == 1 else cast.data[key][i,j,:].reshape(-1,1)
					coef, p = stats.spearmanr(np.squeeze(obs).astype(float), np.squeeze(comp).astype(float))
					cast.save_point_skill(key, 'SpearmanCoef', np.asarray([coef]), lat_ndx=i, lon_ndx=j)
					cast.save_point_skill(key, 'SpearmanP', np.asarray([p]), lat_ndx=i, lon_ndx=j)
			cast.mask_nans_skill()
			if self.verbose:
				print('  '+key + ' [' + '*'*25+ ']')


	#------------------ Stuff for training RTF models------------------------#
	def train_rtf_models(self, forecast_methodologies, args):
		if self.verbose:
			print('\nTraining RTF Models')
		for method in forecast_methodologies:
			if method == 'MLR':
				self.forecast_model(method,  standardization=args['mlr_standardization'], fit_intercept=args['mlr_fit_intercept'])
			elif method == 'ELM':
				self.forecast_model(method,  hidden_layer_neurons=args['elm_hidden_layer_neurons'], activation=args['elm_activation'], standardization=args['elm_standardization'], minmax_range=args['elm_minmax_range'])
			elif method == 'EM':
				self.forecast_model(method, standardization=None)
			else:
				assert False, 'Invalid MME {}'.format(method)

	def forecast_model(self, model_type, hcst='hindcasts', fcst='training_forecasts', hidden_layer_neurons=5, activation='sigm', standardization='minmax', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', pca_variability=-1, W=None, minmax_range=[-1,1] ):
		"""wrapper class for training models on all available data for use in making real time forecasts """
		hcst, tfcst = getattr(self, hcst), getattr(self, fcst)
		key = hcst.available_data()[0]
		total = hcst.lats['Obs'].shape[0] *  hcst.lons['Obs'].shape[0]
		steps = [total / 25.0 * i for i in range(26)]
		ndx, step_ndx = 0, 0
		if self.verbose:
			print(model_type + ' [' + ' '*25 + ']', end='\r')
		for i in range(hcst.lats[key].shape[0]):
			for j in range(hcst.lons[key].shape[0]):
				if ndx > steps[step_ndx] and self.verbose:
					print(model_type + ' [' + '*'*step_ndx + (25-step_ndx)*' ' + ']', end='\r')
					step_ndx += 1
				ndx += 1
				lat_ndx, lon_ndx = i if hcst.lats[key].shape[0] > 1 else -1, j if hcst.lats[key].shape[0] > 1 else -1

				data = hcst.assemble_training_data(lat_ndx=lat_ndx, lon_ndx=lon_ndx)
				forecast, model, scaler, pca = self.forecast_model1D(data, model_type,  standardization=standardization, pca_variability=pca_variability, hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, minmax_range=minmax_range)
				tfcst.add_data(model_type, forecast, lat_ndx=lat_ndx, lon_ndx=lon_ndx, mm=False)
				tfcst.add_lats(model_type, hcst.lats['Obs'])
				tfcst.add_lons(model_type, hcst.lons['Obs'])
				tfcst.add_years(np.squeeze(hcst.years))
				tfcst.add_spm(model_type, model, lat_ndx=lat_ndx, lon_ndx=lon_ndx, year_ndx=0 )
				tfcst.add_scaler(model_type, scaler, lat_ndx=lat_ndx, lon_ndx=lon_ndx, year_ndx=0)
				tfcst.add_pca(model_type, pca, lat_ndx=lat_ndx, lon_ndx=lon_ndx, year_ndx=0)
				tfcst.replace_nans()
		if self.verbose:
			print(model_type + ' [' + '*'*25+ ']')

	def forecast_model1D(self, data, model_type, standardization='None', pca_variability=-1, hidden_layer_neurons=5, activation='sigm', max_iter=200, normalize=False, fit_intercept=True, alpha=1.0, solver='auto', W=None, minmax_range=[-1,1]):
		"""train model on all available data, rather than use leave-n-out xvalidation"""
		x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self._preprocess(data, 0, data.shape[0]-1, standardization, pca_variability=pca_variability, minmax_range=minmax_range)
		model = SPM(model_type, xtrain_shape=x_train.shape[1], ytrain_shape=y_train.shape[1], hidden_layer_neurons=hidden_layer_neurons, activation=activation, max_iter=max_iter, normalize=normalize, fit_intercept=fit_intercept, alpha=alpha, solver=solver, W=W, pca=pca)
		model.train(x_train, y_train)
		training_forecast = model.predict(x_train)
		training_forecast = train_scaler.recover(training_forecast, method=standardization)
		return training_forecast, model, train_scaler, pca





	#------------------ Stuff for constructing RTFs------------------------#
	def make_RTFs(self, forecast_methodologies, tfcst='training_forecasts' ):
		tfcsts = getattr(self, tfcst)
		for method in forecast_methodologies:
			if method in tfcsts.available_mmes():
				self.forecast(method)
			else:
				assert False, 'MME {} Not Trained in {} yet'.format(method, tfcst)

	def forecast(self, model_type, model_cast='training_forecasts', data_cast='forecasts'):
		"""use models trained on all available training data to make real-time-forecasts based on new data """
		fcst, modelcst = getattr(self, data_cast), getattr(self, model_cast)

		for i in range(modelcst.lats['Obs'].shape[0]):
			for j in range(modelcst.lons['Obs'].shape[0]):
				lat_ndx, lon_ndx = i if modelcst.lats['Obs'].shape[0] > 1 else -1, j if modelcst.lons['Obs'].shape[0] > 1 else -1
				data = fcst.assemble_training_data(lat_ndx=lat_ndx, lon_ndx=lon_ndx)
				x_train, y_train, x_test, y_test, train_scaler, test_scaler, pca = self._preprocess(data, 0, data.shape[0]-1, train_scaler=modelcst.scalers[model_type][lat_ndx][lon_ndx][0], pca=modelcst.pcas[model_type][lat_ndx][lon_ndx][0])
				out = modelcst.models[model_type][lat_ndx][lon_ndx][0].predict(x_train)
				out = modelcst.scalers[model_type][lat_ndx][lon_ndx][0].recover(out)
				fcst.add_data(model_type, out, lat_ndx=lat_ndx, lon_ndx=lon_ndx, mm=False)
