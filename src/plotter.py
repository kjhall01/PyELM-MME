import matplotlib.pyplot as plt
import numpy as np
import warnings

import cartopy.crs as ccrs
from cartopy import feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
warnings.filterwarnings("ignore")

class Plotter:

	def __init__(self, mme):
		self.mme = mme
		self.easteregg = 'daniel!!'

	def timeline(self, fcst='hindcasts', methods=['EM', 'ELM', 'MLR'], members=[], point=None, obs=True):
		assert self.mme.type == 'Single-Point' or point is not None, 'How would you plot a timeline for 2D data?'
		casts= getattr(self.mme, fcst)

		if point is not None:
			assert len(point) == 2, 'point must be a lat/long pair in form of a list'
			plot_latkey, plot_lonkey = casts.point_to_ndx(point)

		plt.figure(figsize=(9,6))
		markers = ['o', '*', '^', '+']
		colors = ['r', 'g', 'm', 'b' ]
		ndx = 0

		not_present = []
		for key in methods:
			if key in casts.available_mmes():
				data = casts.data[key] if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]
				plt.plot(data,  color=colors[ndx % 4], label=key, marker=markers[ndx % 4],ms=3, linewidth=1)
				ndx += 1
			else:
				not_present.append(key)

		for key in casts.available_members():
			if members:
				data = casts.data[key] if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]
				plt.plot(data,  color=colors[ndx % 4], label=key, marker=markers[ndx % 4],ms=3, linewidth=1)
				ndx += 1
		if obs:
			obse = casts.data['Obs'] if point is None else casts.mask_nans_var(casts.data['Obs'])[plot_latkey['Obs']][plot_lonkey['Obs']]
			plt.plot(obse,  color='b', label='Obs',marker='o',ms=3,  linewidth=1)


		for metric in not_present:
			print('{} Not Found - Probably typo, or not calculated yet'.format(metric))

		plt.title('{}'.format(fcst.upper()))
		plt.xlabel('Year')
		if type(casts.years[0]) != np.str_:
			plt.xticks(labels=["{0:.0f}".format(i[0]) for i in casts.years], ticks=[i for i in range(len(casts.years))], rotation=90)
		else:
			plt.xticks(labels=[i[0] for i in casts.years], ticks=[i for i in range(len(casts.years))], rotation=90)

		plt.ylabel('Precipitation')
		plt.legend()
		plt.show()

	def skill_matrix(self, fcst='hindcasts', methods=['Obs', 'EM', 'ELM', 'MLR'], metrics=['SpearmanCoef', 'PearsonCoef', 'RMSE', 'MSE',  'MAE', 'IOA'], members=False, point=None, obs=True):
		casts= getattr(self.mme, fcst)
		columns = [metric for metric in metrics]
		columns.insert(0, 'Std Dev')
		columns.insert(0, 'Mean')
		if point is not None:
			assert len(point) == 2, 'point must be a lat/long pair in form of a list'
			plot_latkey, plot_lonkey = casts.point_to_ndx(point)

		rows = []
		for member in casts.available_members():
			if members:
				rows.append(member)

		for method in methods:
			if method in casts.available_mmes() and method != 'Obs':
				rows.append(method)
		if obs:
			rows.insert(0, 'Observations')

		table = []
		for rowlabel in rows:
			rowkey = 'Obs' if rowlabel == 'Observations' else rowlabel
			mean = "{:.2f}".format(np.nanmean(casts.data[rowkey])) if point is None else "{:.2f}".format(np.nanmean(casts.mask_nans_var(casts.data[rowkey])[plot_latkey[rowkey]][plot_lonkey[rowkey]]))
			std = "{:.2f}".format(np.nanstd(casts.data[rowkey])) if point is None else "{:.2f}".format(np.nanstd(casts.mask_nans_var(casts.data[rowkey])[plot_latkey[rowkey]][plot_lonkey[rowkey]]))
			table.append([mean, std ])


		for row in range(len(rows)):
			for i in range(2, len(columns)): #skip mean & std
				metric = columns[i]
				method = rows[row]
				if method != 'Observations':
					value = casts.skill[method][metric][-1][-1] if point is None else casts.mask_nans_var(casts.skill[method][metric])[plot_latkey[rowkey]][plot_latkey[rowkey]]
					table[row].append("{:.2f}".format(value))
				else:
					table[row].append('--')


		hcell, wcell = 0.3, 1.
		hpad, wpad = 0, 0
		#fig=plt.figure(figsize=(2*len(rows)*wcell+wpad, 3*len(rows)*hcell+hpad))
		fig=plt.figure(figsize=(18,6))

		ax = fig.add_subplot(111)
		ax.axis('off')
		#do the table
		the_table = ax.table(cellText=table, rowLabels=rows, colLabels=columns,loc='center')
		the_table.set_fontsize(20)
		the_table.scale(1,4)
		plt.show()

	def box_plot(self, fcst='hindcasts', methods=[ 'EM', 'ELM', 'MLR'], members=False, point=None, obs=True):
		casts, data = getattr(self.mme, fcst), None
		labels = []
		if point is not None:
			assert len(point) == 2, 'point must be a lat/long pair in form of a list'
			plot_latkey, plot_lonkey = casts.point_to_ndx(point)

		for key in casts.available_mmes():
			if key in methods:
				labels.append(key)
				if data is None:
					data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]
				else:
					data = np.hstack((data, casts.data[key].ravel().reshape(-1,1) if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]))

		for key in casts.available_members():
			if members:
				labels.append(key)
				if data is None:
					data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]
				else:
					data = np.hstack((data, casts.data[key].ravel().reshape(-1,1) if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]))

		if obs:
			labels.append('Obs')
			if data is None:
				data = casts.data['Obs'].ravel().reshape(-1,1) if point is None else casts.data['Obs'][plot_latkey['Obs']][plot_lonkey['Obs']]
			else:
				data = np.hstack((data, casts.data['Obs'].ravel().reshape(-1,1) if point is None else casts.data['Obs'][plot_latkey['Obs']][plot_lonkey['Obs']]))
		fig, ax = plt.subplots(figsize=(10,6))
		ax.boxplot(data, whis=255)
		plt.xticks(labels=labels, ticks=[i+1 for i in range(len(labels))] )
		ax.set_xlim(-0.5, len(labels)+1.5)
		plt.show()

	def bar_plot(self, fcst='forecasts', methods=[ 'EM', 'ELM', 'MLR'], members=False, point=None, obs=True):
		casts = getattr(self.mme, fcst)

		if point is not None:
			assert len(point) == 2, 'point must be a lat/long pair in form of a list'
			plot_latkey, plot_lonkey = casts.point_to_ndx(point)

		plt.figure(figsize=(10,5))
		colors = ['r', 'g', 'm', 'b']
		width = 0.35
		keys = [key for key in casts.available_mmes() if key in methods]
		if obs:
			if 'Obs' in casts.available_data():
				keys.append('Obs')

		hfl = width / (len(keys))
		ndx = 1
		if obs:
			if 'Obs' in casts.available_data():
				obse = casts.data['Obs'] if point is None else casts.mask_nans_var(casts.data['Obs'])[plot_latkey['Obs']][plot_lonkey['Obs']]
				plt.bar(np.arange(obse.shape[0]) - (width /2) + hfl * (ndx-1) , np.squeeze(obse), hfl, label='Obs')

		for key in casts.available_mmes():
			if key in methods:
				data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]
				years = np.arange(np.squeeze(data).shape[0])
				plt.bar(np.arange(len(data)) - (width /2) + hfl * ndx , np.squeeze(data) , hfl,  label=key)
				ndx += 1

		for key in casts.available_members():
			if members:
				data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.mask_nans_var(casts.data[key])[plot_latkey[key]][plot_lonkey[key]]
				years = np.arange(np.squeeze(data).shape[0])
				plt.bar(np.arange(len(data)) - (width /2) + hfl * ndx , np.squeeze(data) , hfl,  label=key)
				ndx += 1

		plt.title('{}{}'.format(fcst, '' if point is None else ' - ({},{})'.format(point[0], point[1])))
		plt.xlabel('Year')
		try:
			plt.xticks(labels=[str(int(yr)) for yr in np.squeeze(casts.years)], ticks=[i for i in range(len(casts.years))] )
		except:
			print(years)
			plt.xticks(labels=years, ticks =[i for i in range(len(years))])
		plt.ylabel(fcst.upper())
		plt.legend()
		plt.show()

	def map(self, fcst='hindcasts', data='skill', methods=[ 'EM', 'ELM', 'MLR'], members=False, obs=False, metrics=['SpearmanCoef', 'PearsonCoef', 'RMSE', 'MSE',  'MAE', 'IOA']):
		casts = getattr(self.mme, fcst)
		assert data in ['skill', 'data'], 'invalid data to map selection'
		x_keys, method_keys = [], []

		if data == 'skill':
			for skill in casts.available_skill2('Obs'):
				if skill in metrics:
					x_keys.append(skill)
		else:
			for year in range(len(casts.years)):
				x_keys.append(year)

		for method in casts.available_mmes():
			if method in methods:
				method_keys.append(method)
		for member in casts.available_members():
			if members:
				method_keys.append(member)
		if obs:
			method_keys.append('Obs')


		fig, ax = plt.subplots(nrows=len(method_keys), ncols=len(x_keys), figsize=(4*len(method_keys),4* len(x_keys)),sharex=True,sharey=True, subplot_kw={'projection': ccrs.PlateCarree()}) #creates pyplot plotgrid with maps
		if len(x_keys) == 1:
			ax = [ax]
		if len(method_keys) == 1:
			ax = [ax]
		states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries',scale='10m',facecolor='none')#setting more variables
		for i in range(len(method_keys)): # for each model, but this is always one because were only doing one model
			for j in range(len(x_keys)): #for each season
				ax[i][j].set_extent([np.min(casts.lons['Obs']),np.max(casts.lons['Obs']), np.min(casts.lats['Obs']), np.max(casts.lats['Obs'])], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
				ax[i][j].add_feature(feature.LAND) #adds predefined cartopy land feature - gets overwritten
				pl=ax[i][j].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
				pl.ylabels_left, pl.xlabels_top, pl.xlabels_bottom,  pl.ylabels_right  = True, False , True, False #adds labels to dashed gridlines on left and bottom
				pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff

				ax[i][j].add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

				if j == 0: # if this is the leftmost plot
					ax[i][j].text(-0.25, 0.5, '{}'.format(str(method_keys[i]).upper()),rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i][j].transAxes) #print title vertially on the left side
				if i == 0: # if this is the top plot
					ax[i][j].set_title("{}".format(x_keys[j]).upper()) #print season on top of each plot

				var = getattr(casts, data)[method_keys[i]][x_keys[j]] if data == 'skill' else getattr(casts, data)[method_keys[i]][:,:,x_keys[j]]
				var = casts.mask_nans_var(var)
				if x_keys[j] in ['SpearmanCoef', 'PearsonCoef']:
					CS1 = ax[i][j].pcolormesh(np.linspace(casts.lons['Obs'][0], casts.lons['Obs'][-1],num=casts.lons['Obs'].shape[0]), np.linspace(casts.lats['Obs'][0], casts.lats['Obs'][-1], num=casts.lats['Obs'].shape[0]), var, vmin=-1, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
				elif x_keys[j] in ['IOA']:
					CS1 = ax[i][j].pcolormesh(np.linspace(casts.lons['Obs'][0], casts.lons['Obs'][-1],num=casts.lons['Obs'].shape[0]), np.linspace(casts.lats['Obs'][0], casts.lats['Obs'][-1], num=casts.lats['Obs'].shape[0]), var, vmin=0, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
				elif x_keys[j] in ['RMSE']:
					CS1 = ax[i][j].pcolormesh(np.linspace(casts.lons['Obs'][0], casts.lons['Obs'][-1],num=casts.lons['Obs'].shape[0]), np.linspace(casts.lats['Obs'][0], casts.lats['Obs'][-1], num=casts.lats['Obs'].shape[0]), var, cmap='Reds') #adds probability of below normal where below normal is most likely  and nan everywhere else
				else:
					CS1 = ax[i][j].pcolormesh(np.linspace(casts.lons['Obs'][0], casts.lons['Obs'][-1],num=casts.lons['Obs'].shape[0]), np.linspace(casts.lats['Obs'][0], casts.lats['Obs'][-1], num=casts.lats['Obs'].shape[0]), var, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else

				axins = inset_axes(ax[i][j], width="100%", height="5%",  loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1), bbox_transform=ax[i][j].transAxes, borderpad=0.15,) #describes where colorbar should go
				cbar_bdet = fig.colorbar(CS1, ax=ax[i][j],  cax=axins, orientation='horizontal', pad = 0.02) #add colorbar based on hindcast data
		plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
		plt.show()

	def map_skill(self, fcst='hindcasts', methods=['EM', 'ELM', 'MLR'], members=False, obs=False, metrics=['SpearmanCoef', 'PearsonCoef', 'RMSE', 'MAE', 'IOA']):
		self.map(methods=methods, fcst=fcst, members=members, metrics=metrics, data='skill', obs=obs)

	def map_forecasts(self, fcst='hindcasts', methods=['EM', 'ELM', 'MLR'], members=False, obs=False):
		self.map(methods=methods, fcst=fcst, members=members, data='data' , obs=obs)
