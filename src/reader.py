from pathlib import Path
import numpy as np
import xarray as xr

from .casts import Cast

def get_transpose_order(ds):
	desired = ['longitude', 'latitude', 'time']
	cur = [coord for coord in ds.coords if len(ds[coord].shape) > 0] #this is the current shape, we want xyt
	tp_order = [cur.index(desired[i]) for i in range(len(desired))]
	return tp_order


class Reader:
	"""handles reading tsv/csv/ncdf files wth data"""
	def __init__(self):
		self.easteregg = 'close the iris'


	def read_txt(self, filepath, has_years=True, has_obs=True, has_header=False):
		"""Reads data in format csv, tsv, txt for a Single-Lat/Long-Point MME
		-----------------------------------------------------------------------
		filepath: string - describes path to csv, tsv, txt training data file
		has_years: Boolean - whether data file is labeled wth years
		has_obs: Boolean - whether data file has observations
		has_header: Boolean - whether or not data file has a header (we will throw it away)
		-------------------------------------------------------------------------
		returns: Cast Object"""

		assert Path(filepath).is_file(), "Not a valid file path {}".format(filepath) #make sure its a valid file
		delimiter = ',' if filepath.split('.')[-1] == 'csv' else '\t'
		data = np.genfromtxt(filepath, delimiter=delimiter, dtype=float, skip_header=1 if has_header else 0)

		ret, model_start_ndx = Cast(), 0
		if has_years:
			years = data[:,0]
			model_start_ndx += 1

		if has_obs:
			ret.add_obs( data[:, 1])
			model_start_ndx += 1

		for i in range(model_start_ndx, data.shape[1]):
			ret.add_data('Model{}'.format(i), data[:,i])
		ret.add_years(years)
		return 'Single-Point', ret


	def read_multiple_ncdf(self, dir_path, observations_filename=None, latitude_key='latitude', longitude_key='longitude', time_key='time', obs_time_key='time'):
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

		ret = Cast()

		if observations_filename is not None:
			assert (Path(dir_path) / observations_filename).is_file(), "Not a valid file path {}".format(str(Path(dir_path) / observations_filename)) #make sure its a valid file
			#open the observations nc file
			Obs_DS = xr.open_dataset(Path(dir_path) / observations_filename, decode_times=False)
			Obs_DS = Obs_DS.rename_vars({latitude_key:'latitude', longitude_key:'longitude', obs_time_key:'time'})
			if 'M' in Obs_DS.coords:
				Obs_DS = Obs_DS.mean(dim='M')
			if 'L' in Obs_DS.coords:
				Obs_DS = Obs_DS.mean(dim='L')
			Obs_DS = Obs_DS.squeeze()
			for var_name, data in Obs_DS.data_vars.items(): #there should be only one - we're not worryign if there's more than that
				dv = data.values.squeeze()
				dv = dv.transpose(*get_transpose_order(Obs_DS))
				ret.add_obs( dv)

		#open all model data nc files
		model_data_files, model_data = [], []
		for file in Path('.').glob('{}/*'.format(dir_path)):
			if str(file).split('/')[-1] != str(observations_filename):
				model_data_files.append(file)
		DS = xr.open_mfdataset(model_data_files, combine='nested', compat='override', decode_times=False)
		DS = DS.rename_vars({latitude_key:'latitude', longitude_key:'longitude', time_key:'time'})
		if 'M' in DS.coords:
			DS = DS.mean(dim='M')
		if 'L' in DS.coords:
			DS = DS.mean(dim='L')
		DS, ndx = DS.squeeze(), 1
		for var_name, data in DS.data_vars.items():
			dv = data.values.squeeze()
			dv = dv.transpose(*get_transpose_order(DS))
			ret.add_data('Model{}'.format(ndx), dv)

		#for now we are assuming all have same dimensions
		lats, lons = DS.latitude.values, DS.longitude.values
		years = np.asarray([str(i) for i in DS.time.values])
		for data_key in ret.available_data():
			ret.add_lats(data_key, lats)
			ret.add_lons(data_key, lons)
			ret.add_years( years)

		return 'Multi-Point', ret
