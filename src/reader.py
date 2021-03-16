from pathlib import Path
import numpy as np

def get_transpose_order(ds):
	desired = ['longitude', 'latitude', 'time']
	cur = [coord for coord in ds.coords if len(ds[coord].shape) > 0] #this is the current shape, we want xyt
	tp_order = [cur.index(desired[i]) for i in range(len(desired))]
	return tp_order


class Reader:
	"""handles reading tsv/csv/ncdf files wth data"""
	def __init__(self):
		self.easteregg = 'close the iris'


	def read_txt(self, filepath, is_forecast=False):
		"""Reads data in format csv, tsv, txt for a Single-Lat/Long-Point MME
		-----------------------------------------------------------------------
		filepath: string - describes path to csv, tsv, txt training data file
		is_forecast: Boolean - whether data file is for real time forecast or not
		-------------------------------------------------------------------------
		Training Data Format should be like this (is_forecast=False):

			Year1,Observation1,Model1_1,Model2_1,Model3_1,Model4_1,Model5_1,....ModelN_1
			.   .  .
			YearM,ObservationM,Model1_M,Model2_M,Model3_M,Model4_M,Model5_M,....ModelN_M
		------------------------------------------------------------------------
		Real Time Forecast Data Format should be like this (is_forecast=True):

			Year1,Model1_1,Model2_1,Model3_1,Model4_1,Model5_1,....ModelN_1
			.  .  .
			YearM,Model1_M,Model2_M,Model3_M,Model4_M,Model5_M,....ModelN_M
		------------------------------------------------------------------------
		returns tuple of (data_type, years, observations, model_data, lats, lons, hindcasts, training_forecasts)"""

		assert Path(filepath).is_file(), "Not a valid file path {}".format(filepath) #make sure its a valid file

		delimiter = ',' if filepath.split('.')[-1] == 'csv' else '\t'
		data_type = 'Single-Point'
		lats, lons = [1], [1]
		data = np.genfromtxt(filepath, delimiter=delimiter, dtype=float)
		if len(data.shape) == 1:
			data = data.reshape(1,-1)

		years = data[:,0].reshape(-1,1)

		if not is_forecast:
			observations = data[:, 1].reshape(-1,1)
			model_data = data[:, 2:]
			training_forecasts, hindcasts = {}, {}
			for model in range(model_data.shape[1]):
				if 'Model {}'.format(model+1) not in hindcasts.keys():
					hindcasts['Model {}'.format(model+1)] = model_data[:, model]
					training_forecasts['Model {}'.format(model+1)] =model_data[:, model]
			hindcasts['Obs']= observations
			training_forecasts['Obs']= observations
			return (data_type, years, observations, model_data, lats, lons, hindcasts, training_forecasts)
		else:
			model_data = data[:, 1:]
			return (data_type, years, [],  model_data, lats, lons, {}, {} )


	def read_multiple_ncdf(self, dir_path, observations_filename=None, latitude_key='latitude', longitude_key='longitude', time_key='time'):
		"""reads all ncdf files in your directory path
		---------------------------------------------------------------------------
		dir_path (string) - path to directory storing separate ncdf files for each model, and observations
		observations_filename (string) - name of ncdf file in dir_path containing observations
		latitude_key (string) - name of coordinate field in NCDF files representing latitude
		longitude_key (string) - name of coordinate field in NCDF files representing longitude
		time_key (string) - name of coordinate field in NCDF files representing years
		obs_time_key (string) - name of coordinate field in observation NCDF file representing years
		using_datetime (boolean) - whether year coordinates in NCDF files are in datetime format, or not
		axis_order (string) - represnts order of axes in NCDF variable.values - if 'txy', we will reshape to 'xyt'.
		is_forecast (Boolean) - is this RTF data? then observations filename will not be used , observations wont be loaded
		--------------------------------------------------------------------------
		NCDF File Format:
			> coords: latitude (n_latitudes), longitude (n_longitudes), time (n_years)
			> should have 1 dataArray with dataArray.values of shape (n_lats, n_lons, n_years) - if (n_years, n_lats, n_lons), use axis_order='txy' to reshape
			> should be 1 file for each model, 1 for observations.
			> All files must have same dimensions
		---------------------------------------------------------------------------"""

		assert Path(dir_path).is_dir(), "Not a valid directory path {}".format(dir_path) #make sure its a valid file
		data_type= 'Multi-Point'

		observations = []
		if observations_filename is not None:
			assert (Path(dir_path) / observations_filename).is_file(), "Not a valid file path {}".format(str(Path(dir_path) / observations_filename)) #make sure its a valid file

			#open the observations nc file
			Obs_DS = xr.open_dataset(Path(dir_path) / observations_filename, decode_times=False)
			Obs_DS = Obs_DS.rename_vars({latitude_key:'latitude', longitude_key:'longitude', time_key:'time'})
			if 'M' in Obs_DS.coords:
				Obs_DS = Obs_DS.mean(dim='M')
			Obs_DS = Obs_DS.squeeze()
			for var_name, data in DS.data_vars.items():
				dv = data.values.squeeze()
				dv = dv.transpose(*get_transpose_order(Obs_DS))
				observations = np.asarray([dv])

		#open all model data nc files
		model_data_files, model_data = [], []
		for file in Path('.').glob('{}/*'.format(dir_path)):
			if str(file).split('/')[-1] != observations_filename:
				model_data_files.append(file)
		DS = xr.open_mfdataset(model_data_files, decode_times=False)
		DS = DS.rename_vars({latitude_key:'latitude', longitude_key:'longitude', time_key:'time'})
		if 'M' in DS.coords:
			DS = DS.mean(dim='M')
		DS = DS.squeeze()
		for var_name, data in DS.data_vars.items():
			dv = data.values.squeeze()
			dv = dv.transpose(*get_transpose_order(DS)) if axis_order == 'txy' else data.values
			model_data.append(dv)

		lats, lons = DS.latitude.values, DS.longitude.values
		years = np.asarray([str(i) for i in DS.time.values]).reshape(-1,1)

		model_data=np.asarray(model_data)
		nanmask = np.sum(np.isnan(model_data), axis=(0,3))
		goodmask = np.isnan(nanmask / nanmask).astype(bool)
		nanmask = nanmask.astype(bool)
		model_data[:,nanmask,:] = np.random.randint(-1000, -900, model_data.shape)[:,nanmask,:]
		observations[:,nanmask,:] = np.random.randint(-1000, -900, observations.shape)[:,nanmask,:]
		hindcasts, training_forecasts = {}, {}
		hindcasts['Obs'] = observations.reshape(observations.shape[1], observations.shape[2], observations.shape[3])
		training_forecasts['Obs'] = observations.reshape(observations.shape[1], observations.shape[2], observations.shape[3])
		return (data_type, model_data, lats, lons, years, nanmask, goodmask, observations, hindcasts, training_forecasts)
