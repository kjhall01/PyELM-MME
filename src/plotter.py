import matplotlib.pyplot as plt
import numpy as np

class Plotter:

	def __init__(self, mme):
		self.mme = mme
		self.easteregg = 'daniel!!'

	def plot_timeline(self, fcst='hindcasts', methods=['Obs', 'EM', 'ELM', 'MLR'], members=[], point=None):
		assert self.mme.type == 'Single-Point', 'How would you plot a timeline for 2D data?'
		casts= getattr(self.mme, fcst)
		plt.figure(figsize=(9,6))
		markers = ['o', '*', '^', '+']
		colors = ['r', 'g', 'm', 'b' ]
		ndx = 0
		print(casts.available_data())
		for key in casts.available_data():
			if key in methods:
				methods.pop(methods.index(key))
				if key == 'Obs':
					obs = casts.data[key] if point is None else casts.data[key][point[0]][point[1]]
					plt.plot(obs,  color='b', label='Obs',marker='o',ms=3,  linewidth=1)
				else:
					data = casts.data[key] if point is None else casts.data[key][point[0]][point[1]]
					plt.plot(data,  color=colors[ndx], label=key, marker=markers[ndx % 4],ms=3, linewidth=1)
					ndx += 1

		for metric in methods:
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

	def skill_matrix(self, fcst='hindcasts', methods=['Obs', 'EM', 'ELM', 'MLR'], metrics=['SpearmanCoef', 'PearsonCoef', 'RMSE', 'MAE', 'IOA'], members=[], point=None, obs=True):
		casts= getattr(self.mme, fcst)
		columns = [metric for metric in metrics]
		columns.insert(0, 'Std Dev')
		columns.insert(0, 'Mean')

		rows = []
		for member in members:
			if member in casts.available_members():
				rows.append(member)

		for method in methods:
			if method in casts.available_mmes() and method != 'Obs':
				rows.append(method)
		if obs:
			rows.insert(0, 'Observations')


		table = []
		for rowlabel in rows:
			rowkey = 'Obs' if rowlabel == 'Observations' else rowlabel
			mean = "{:.2f}".format(np.nanmean(casts.data[rowkey])) if point is None else "{:.2f}".format(np.nanmean(casts.data[rowkey][point[0]][point[1]]))
			std = "{:.2f}".format(np.nanstd(casts.data[rowkey])) if point is None else "{:.2f}".format(np.nanstd(casts.data[rowkey][point[0]][point[1]]))
			table.append([mean, std ])


		for row in range(len(rows)):
			for i in range(2, len(columns)): #skip mean & std
				metric = columns[i]
				method = rows[row]
				if method != 'Observations':
					value = casts.skill[method][metric][-1][-1] if point is None else casts.skill[method][metric][point[0]][point[1]]
					table[row].append("{:.2f}".format(value))
				else:
					table[row].append('--')


		hcell, wcell = 0.3, 1.
		hpad, wpad = 0, 0
		#fig=plt.figure(figsize=(2*len(rows)*wcell+wpad, 3*len(rows)*hcell+hpad))
		fig=plt.figure(figsize=(10,7))

		ax = fig.add_subplot(111)
		ax.axis('off')
		#do the table
		the_table = ax.table(cellText=table, rowLabels=rows, colLabels=columns,loc='center')
		the_table.set_fontsize(20)
		the_table.scale(1,4)
		plt.show()

	def box_plot(self, fcst='hindcasts', methods=[ 'EM', 'ELM', 'MLR'], members=[], point=None, obs=True):
		casts, data = getattr(self.mme, fcst), None
		labels = []
		for key in casts.available_mmes():
			if key in methods:
				labels.append(key)
				if data is None:
					data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.data[key][point[0]][point[1]]
				else:
					data = np.hstack((data, casts.data[key].ravel().reshape(-1,1) if point is None else casts.data[key][point[0]][point[1]]))

		for key in casts.available_members():
			if key in members:
				labels.append(key)
				if data is None:
					data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.data[key][point[0]][point[1]]
				else:
					data = np.hstack((data, casts.data[key].ravel().reshape(-1,1) if point is None else casts.data[key][point[0]][point[1]]))

		if obs:
			labels.append('Obs')
			if data is None:
				data = casts.data['Obs'].ravel().reshape(-1,1) if point is None else casts.data['Obs'][point[0]][point[1]]
			else:
				data = np.hstack((data, casts.data['Obs'].ravel().reshape(-1,1) if point is None else casts.data['Obs'][point[0]][point[1]]))

		plt.boxplot(data, whis=255)
		plt.xticks(labels=labels, ticks=[i+1 for i in range(len(labels))] )
		plt.show()

	def bar_plot(self, fcst='forecasts', methods=[ 'EM', 'ELM', 'MLR'], members=[], point=None, obs=True):
		casts = getattr(self.mme, fcst)

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
				obse = casts.data['Obs'] if point is None else casts.data['Obs'][point[0]][point[1]]
				plt.bar(np.arange(obse.shape[0]) - (width /2) + hfl * (ndx-1) , np.squeeze(obse), hfl, label='Obs')

		for key in casts.available_mmes():
			if key in methods:
				data = casts.data[key].ravel().reshape(-1,1) if point is None else casts.data[key][point[0]][point[1]]
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
