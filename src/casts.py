import numpy as np
import xarray as xr

#for now not supporting xarray DataSets
class Cast:
	def __init__(self):
		self.easteregg = "Tek ma tay bray'tac"
		self.data, self.skill = {}, {}
		self.model_members = []
		self.models, self.scalers = {}, {}
		self.lats, self.lons = {}, {}
		self.years, self.pcas = np.arange(1), {}


	def point_to_ndx(self, point):
		plot_latkey, plot_lonkey = {}, {}
		for key in self.available_data():
			plot_latkey[key], found_yet = -1000, False
			for lat in range(len(self.lats[key])):
				if self.lats[key][lat] >= point[0] and not found_yet:
					plot_latkey[key] = lat
					found_yet = True


			plot_lonkey[key], found_yet = -1000, False
			for lon in range(len(self.lons[key])):
				if self.lons[key][lon] >= point[1] and not found_yet:
					plot_lonkey[key] = lon
					found_yet = True
		return plot_latkey, plot_lonkey


	def get_nanmask(self):
		model_data = []
		for key in self.available_data():
			model_data.append(self.data[key])
			axes = (0,3) if len(self.data[key].shape) == 3 else (0)

		model_data = np.asarray(model_data)
		self.nanmask = np.sum(np.isnan(model_data), axis=axes).astype(bool)
		return True

	def replace_nans(self):
		for key in self.available_data():
			if len(self.data[key].shape) == 3:
				self.data[key][self.nanmask, :] = np.random.randint(-1000, -900, self.data[key].shape)[self.nanmask,:]
			else:
				self.data[key][self.nanmask] = np.random.randint(-1000, -900, self.data[key].shape)[self.nanmask]
		return True

	def replace_nans_skill(self):
		for key in self.available_skill1():
			for method in self.skill[key].keys():
				if len(self.skill[key][method].shape) == 3:
					self.skill[key][method][self.nanmask, :] = np.random.randint(-1000, -900, self.skill[key].shape)[self.nanmask,:]
				else:
					self.skill[key][method][self.nanmask] = np.random.randint(-1000, -900, self.skill[key].shape)[self.nanmask]
		return True

	def mask_nans(self):
		for key in self.available_data():
			if len(self.data[key].shape) == 3:
				self.data[key][self.nanmask, :] = np.nan
			else:
				self.data[key][self.nanmask] = np.nan
		return True

	def mask_nans_var(self, var):
		try:
			x  = self.nanmask
		except:
			self.get_nanmask()
		var[self.nanmask] = np.nan
		return var

	def mask_nans_skill(self):
		for key in self.available_skill1():
			for method in self.skill[key].keys():
				if len(self.skill[key][method].shape) == 3:
					self.skill[key][method][self.nanmask] = np.nan
				else:
					pass
		return True

	def assemble_training_data(self, lat_ndx=-1, lon_ndx=-1):
		model_data = []
		for key in self.model_members:
			time_series_data_at_pt_xy = self.data[key] if lat_ndx==-1 and lon_ndx == -1 else self.data[key][lat_ndx, lon_ndx,:].reshape(-1,1)
			model_data.append(time_series_data_at_pt_xy)
		model_data=np.asarray(model_data).squeeze()
		if len(model_data.shape) == 1:
			model_data = model_data.reshape(-1,1)
		model_data = np.asarray(model_data).transpose(1,0) if lat_ndx==-1 and lon_ndx == -1 else np.asarray(model_data).transpose(1,0)
		if 'Obs' in self.available_data():
			obs = self.data['Obs'] if lat_ndx == -1 and lon_ndx == -1 else self.data['Obs'][lat_ndx, lon_ndx, :].reshape(-1,1)
			data = np.hstack((self.years, obs, model_data))
		else:
			obs = np.arange(model_data.shape[0]).reshape(-1,1)
			data = np.hstack((self.years, obs, model_data))


		return data

	def add_spm(self, key, model, lat_ndx=-1, lon_ndx=-1, year_ndx=-1):
		self.models[key][lat_ndx][lon_ndx][year_ndx] = model
		#this works for both 1D and 2D because list[-1] gets list[0] if len(list) == 1

	def add_scaler(self, key, scaler, lat_ndx=-1, lon_ndx=-1, year_ndx=-1 ):
		self.scalers[key][lat_ndx][lon_ndx][year_ndx] = scaler
		#this works for both 1D and 2D because list[-1] gets list[0] if len(list) == 1

	def add_pca(self, key, pca, lat_ndx=-1, lon_ndx=-1, year_ndx=-1 ):
		self.pcas[key][lat_ndx][lon_ndx][year_ndx] = pca
		#this works for both 1D and 2D because list[-1] gets list[0] if len(list) == 1


	def add_years(self, years):
		assert len(years.shape) == 1, 'years must be 1d array'
		self.years = years.reshape(-1,1)

	def add_lats(self, key, lats):
		assert len(lats.shape) == 1, 'lats must be 1d array'
		self.lats[key] = lats#shape (n)

	def add_lons(self, key, lons):
		assert len(lons.shape) == 1, 'lats must be 1d array'
		self.lons[key] = lons # shape (m)

	def available_skill2(self, key):
		return sorted([key for key in self.skill[key].keys()])

	def available_skill1(self):
		return sorted([key for key in self.skill.keys()])


	def available_members(self):
		return sorted([key for key in self.data.keys() if key in self.model_members ])

	def available_data(self):
		return sorted([key for key in self.data.keys() ])

	def available_mmes(self):
		return sorted([key for key in self.data.keys() if key not in self.model_members and key != 'Obs'])

	def obs_available(self):
		return 'Obs' in self.data.keys()

	def add_obs(self, data):
		key = 'Obs'
		if len(data.shape) == 1: #for (m,) data as a list , reshape to (1, m)
			data = data.reshape(-1, 1)
		assert len(data.shape) in [2,3]

		if len(data.shape) == 2: #1 x years
			self.data[key] = data 		#data[key] can either be (1,m) or [lat, lon, time]
			self.add_lats(key, np.arange(1)) #setting default
			self.add_lons(key, np.arange(1)) #setting defaults
			self.add_years( np.arange(data.shape[1]))

		if len(data.shape) == 3: #lat x lon x years

			self.data[key] = data
			self.add_lats(key, np.arange(data.shape[0])) #setting defaults
			self.add_lons(key, np.arange(data.shape[1])) #setting defaults
			self.add_years( np.arange(data.shape[2])) #setting defaults
		return True

	def add_data(self, key, data, lat_ndx=-1, lon_ndx=-1, mm=True, domain=None):
		if len(data.shape) == 1: #for (m,) data as a list , reshape to (1, m)
			data = data.reshape(-1, 1)
		assert (len(data.shape) == 2 and data.shape[1] == 1) or len(data.shape) == 3, 'input data must be column vector or 3D matrix'




		if mm:
			self.model_members.append(key)

		if lat_ndx == -1 and lon_ndx == -1:
			self.data[key] = data  		#data[key] can either be (1,m) or [lat, lon, time]
		else:
			if key not in self.available_data() and key != 'Obs' and 'Obs' in self.available_data():
				self.data[key] = np.zeros(self.data['Obs'].shape)
			elif domain is not None:
				self.data[key] = [[0 for j in range(domain[1])] for k in range(domain[2])]
			self.data[key][lat_ndx, lon_ndx, :] = np.squeeze(data)

		try:
			lons, lats = self.lons[[key for key in self.lons][0]], self.lats[[key for key in self.lats][0]]
		except:
			lons, lats = [1], [1]

		try:
			x = self.scalers[key]
		except:
			if 'Obs' in self.available_data():
				self.scalers[key] = [[[0 for i in range(self.years.shape[0])] for j in range(len(self.lons['Obs']))] for k in range(len(self.lats['Obs']))]
			else:
				self.scalers[key] = [[[0 for i in range(self.years.shape[0])] for j in range(len(lons))] for k in range(len(lats))]

		try:
			x = self.models[key]
		except:
			if 'Obs' in self.available_data():
				self.models[key] = [[[j+i+k for i in range(self.years.shape[0])] for j in range(len(self.lons['Obs']))] for k in range(len(self.lats['Obs']))]
			else:
				self.models[key] = [[[0 for i in range(self.years.shape[0])] for j in range(len(lons))] for k in range(len(lats))]

		try:
			x = self.pcas[key]
		except:
			if 'Obs' in self.available_data():
				self.pcas[key] = [[[j+i+k for i in range(self.years.shape[0])] for j in range(len(self.lons['Obs']))] for k in range(len(self.lats['Obs']))]
			else:
				self.pcas[key] = [[[0 for i in range(self.years.shape[0])] for j in range(len(lons))] for k in range(len(lats))]


	def save_point_skill(self, method_key, metric_key, data, lat_ndx=-1, lon_ndx=-1):
		assert len(data.shape) == 1 and data.shape[0] == 1, 'must be singular array'
		assert lon_ndx < len(self.lons[method_key]), 'lon ndx out of range'
		assert lat_ndx < len(self.lats[method_key]), 'lat ndx out of range'
		if method_key not in self.available_skill1():
			self.skill[method_key] = {}

		if metric_key not in self.skill[method_key].keys():
			if len(self.data['Obs'].shape) == 2:
				self.skill[method_key][metric_key] = np.zeros((1,1))
			else:
				self.skill[method_key][metric_key] = np.zeros(self.data['Obs'].shape[:-1])
		self.skill[method_key][metric_key][lat_ndx, lon_ndx] = data[0]
