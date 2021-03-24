# PyELM-MME: A Python Module for Constructing and Comparing Multi-Model Ensemble Methodologies
### Authors: Nachiketa Acharya (Climate Forecasting Science, Statistics &AI/ML), Kyle Hall (AI/ML & Software Dev)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4515069.svg)](https://doi.org/10.5281/zenodo.4515069)

Based on the original code written in Matlab by Nachiketa Acharya

## Install with Anaconda:
```conda install -c hallkjc01 pyelmmme```

## Build from Source with Dependencies:
	* cartopy: conda install -c conda-forge cartopy
	* xarray: conda install xarray
	* scipy, numpy, matplotlib, pandas
	* scikit-learn: conda install sklearn
	* hpelm: pip install hpelm
#####
 
# Data I/O
## Initializing an MME Object using the Reader() class: 

### Read 1D Data (GCM hindcasts & observations, spatially aggregated) with the Reader().read_txt() method. 
```
import pyelmmme as pm
reader = pm.Reader()
hindcast_data = reader.read_txt('your_hindcast_file.csv', has_obs=True, has_years=True, has_header=False) 
mme = pm.MME(hindcast_data)
```
- 'your_hindcast_file.csv' will be a .csv file with the following format, for N models and M years. 

Year 1 | Observation 1 | Model 1_1 | ... | Model N_1
 --- | --- | --- | --- | ---
 Year 2 | Observation 2 | Model 1_2 | ... | Model N_2
 Year 3 | Observation 3 | Model 1_3 | ... | Model N_3
 . | | | | .
 . | | | | .
 Year M | Observation M | Model 1_M | ... | Model N_M
 
- If your file doesn't have year labels, or has a header, you'll need to adjust the 'has_header' and 'has_years keyword arguments, whose defaults are has_header=False and has_years=True. 

- Note that for initializing an MME object requires historical observations - how would one train statistical models without them? 

### Read 2D Data (model hindcasts & observations, latxlongxtime) with the Reader().read_multiple_ncdf() method. 
```
import pyelmmme as pm
reader = pm.Reader()
hindcast_data = reader.read_multiple_ncdf('your_hindcast_directory', observations_filename='test_obs.nc', latitude_key='Y', longitude_key='X',obs_time_key='T', time_key='S') 
mme = pm.MME(hindcast_data)
```
- your data files should be under the your_hindcast_directory directory - your_hindcast_directory/test_obs.nc 
- You need to provide the names of the coordinates in your netCDF Files with the latitude_key, longitude_key, time_key and obs_time_key keyword arguments. PyELM-MME will dynamically rename them .
- PyELM-MME doesn't care about any dimension but latitude, longitude and time. 'M' (model member) and 'L' (lead time) are other commmon ones- if they are present in your data, they will be removed by averaging over those dimensions. 
- again, note that observations data is required (statistics need a Y vector) 

## Adding Forecast Data to an MME Object:
### After an MME Object is initialized, input data for real time forecasts can be added. 
#### 1D: 
```
fcst_data = reader.read_txt('your_forecast_file.csv', has_obs=True, has_years=True, has_header=False) 
mme.add_forecast(fcst_data) 
```

#### 2D: 
```
fcst_data = reader.read_multiple_ncdf('your_forecast_directory', observations_filename='test_obs.nc', latitude_key='Y', longitude_key='X',obs_time_key='T', time_key='S') 
mme = pm.MME(fcst_data)
```

- Note that for forecast input data, observations are optional. But you won't be able to examine skill if they arent there. 

## Exporting MME Data:
### After you've trained models and calculated cross-validated hindcasts, you will be able to export data. 
- If you've been working with 1D (spatially aggregated) data, you'll want to export to csv: 
```mme.export_csv('outputfile.csv', fcst='hindcasts') ```
- And if youve been working with 2D (lat x long x time) data, you'll want to export to netCDF: 
```mme.export_ncdf('outputfile.ncdf', fcst='hindcasts') ```
- if you want to export real time forecasts, set the keyword argument fcst='forecasts'
```mme.export_ncdf('outputfile.ncdf', fcst='forecasts') ```


# Training MME Models
## Calculating Cross-Validated Hindcasts:
### Whether you've been working with spatially aggregated data or not, you'll train models the same way. 
### Use the mme.train_mmes() convenience function to train MMES: 
```mme.train_mmes(['EM', 'ELM', 'MLR'], args)```

### args will be a python dict object with the following keys: 
```
args = {
    #EnsembleMean settings
    'em_xval_window': 1,               #odd number - behavior undefined for even number

    #MLR Settings
    'mlr_fit_intercept': True,         #Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered) (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    'mlr_xval_window': 1,               #odd number - behavior undefined for even number
    'mlr_standardization': None,        #'std_anomaly' or None

    #ELM Settings 
    'elm_xval_window': 1,              #odd number - behavior undefined for even number
    'elm_hidden_layer_neurons':3,     #number of hidden layer neurons - overridden if using PCA init
    'elm_activation': 'sigm',          #“lin” for linear, “sigm” or “tanh” for non-linear, “rbf_l1”, “rbf_l2” or “rbf_linf” for radial basis function neurons (https://hpelm.readthedocs.io/en/latest/api/elm.html)
    'elm_standardization' : 'std_anomaly',  #'minmax' or 'std_anomaly' or None
    'elm_minmax_range': [-1, 1]        #choose [minimum, maximum] values for minmax scaling. ignored if not using minmax scaling 
}
```

### or, you can use the cross_validate() function to do each model individually if you want. 

#### Each methodology is referred to by a code (a string) and takes a different set of keyword arguments. Hindcasts created are stored internally.

#### Standard Ensemble Mean (EM) - simple arithmetic mean
```
mme.cross_validate('EM', xval_window=3)
```
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.


#### Multiple Linear Regression (MLR) - Standard multiple regression between model values and observations
```
mme.cross_validate('MLR', xval_window=3, standardization=None, fit_intercept=False)
```
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly' or None
- fit_intercept (bool, default=True): whether to include intercepts in calculation of model coefficients  


#### Extreme Learning Machine (ELM) - Artificial Neural Network with randomized, untrained hidden layer weights and output weights solved by ELM method.
```
mme.cross_validate('ELM', hidden_layer_neurons=5, activation='sigm', standardization='minmax', minmax_range=[-1,1])
```
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly', 'minmax' or None
- minmax_range (list, default=[-1,1]): minimum and maximum values to scale data to when using minmax scaling.
- hidden_layer_neurons (int, default=5): Number of neurons in ELM's hidden layer
- activation (str, default='sigm'): string representing activation function. 'sigm', 'lin', 'tanh', 'rbf_l1', 'rbf_l2', rbf_linf'
#### The mme.forecast_model() function has the same interface as mme.cross_validate(), except it will train on all available data, not cross-validated data. These are the functions you would use to train RTF models. 
#### Or, you can use the mme.train_rtf_models() convenience function to train many at once: 
```mme.train_rtf_models(['EM', 'ELM', 'MLR'], args)```
- args should probably be the same as it was before, otherwise the skill evaluations you probably did on the cross validated hindcasts won't be relevant.

### After training real time forecast models, you can make real time forecasts! yay. Use the mme.make_RTFs() convenience function:
```mme.make_RTFs(['EM', 'ELM', 'MLR']) #no args dictionary necessary, because we saved those when we trained the models```

- you can also use the mme.forecast() method to do the MME methodologies individually: 
```mme.forecast('ELM', model_cast='training_forecasts', data_cast='forecasts') ```
- model_cast refers to an internal 'Cast' data structure that holds the real-time-forecast models we trained on all available hindcast data. 
- data_cast refers to the internal 'Cast' structure that holds the forecast data that we read and added with add_forecast() 


# Assessing Skill
## Calculate Skill Metrics as Follows:
```
mme.Pearson() # Pearson Correlation 
mme.Spearman() # Spearman Correlation 
mme.MAE() # Mean Absolute Error 
mme.MSE() # Mean Squared Error 
mme.MSE(squared=False) # Root Mean Squared Error 
mme.IOA() #Index of Agreement 
```
- Or, pass a list of keys to the convenience function: 
```
mme.measure_skill(['Pearson', 'Spearman', 'MAE', MSE', 'RMSE', 'IOA'])'
```

# Plotting Results 
### Each mapping function takes a list of methodologies, three keyword args named 'obs' and 'members' and 'fcst, and a list of skill metrics if applicable 
- obs=True will plot observations - if there are not observations, clearly this will cause a problem 
- members=True will plot individual model members 
-  fcst='hindcasts' will plot cross validated hindcasts. fcst='forecasts' will plot real-time forecasts. fcst='training_forecasts' will plot non-cross-validated hindcasts, produced by the real-time-forecast models. 
-  the 'variable' kw argument will set the Y-Axis of the plots, if they're 1D graphics. 
- the 'point' keyword argument will plot the 1D values at a given lat/long point. it should be a python list or tuple
#### Plotting functions include: 
```
ptr = Plotter(mme)

#2D only 
ptr.map_skill(methods=['EM', 'ELM', 'MLR'], metrics=['IOA', 'RMSE', (etc) ], members=True, obs=True) 
ptr.map_forecasts(methods=['EM', 'ELM', 'MLR'], metrics=['IOA', 'RMSE', (etc) ], members=True, obs=True) 

#1D or provide a point arugment: 
ptr.box_plot(methods=['EM', 'ELM', 'MLR'], members=True, obs=True, variable='Precip (mm/day)', point=[9, -36]) 
ptr.bar_plot(methods=['EM', 'ELM', 'MLR'], members=True, obs=True, variable='Precip (mm/day)', point=[9, -36]) 
ptr.timeline(methods=['EM', 'ELM', 'MLR'], members=True, obs=True, variable='Precip (mm/day)', point=[9, -36]) 
ptr.skill_matrx(methods=['EM', 'ELM', 'MLR'], members=True, obs=True, point=[9, -36, metrics=['IOA', 'RMSE', (etc) ]) 
```
#### if you're working with 1D data (GCMS x Years) make sure to leave out the point argument! 

# See PyMME-1D.ipynb and PyMME-2D.ipynb for examples, or just use those!

# Good Luck!

# Works Cited

1. Acharya N, Kar SC, Kulkarni MA, Mohanty UC, Sahoo LN (2011) Multi-model ensemble schemes for predicting northeast mon- soon rainfall over peninsular India. J Earth Syst Sci 120:795–805
2. Acharya N, Srivastava N.A., Panigrahi B.K. and Mohanty U.C. (2014): Development of an artificial neural network based multi-model ensemble to estimate the northeast monsoon rainfall over south peninsular India: an application of extreme learning machine. Climate Dynamics. 43(5):1303-1310.
2. A. Akusok, K. Björk, Y. Miche and A. Lendasse, "High-Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications," in IEEE Access, vol. 3, pp. 1011-1025, 2015, doi: 10.1109/ACCESS.2015.2450498.
3. L. L. C. Kasun, Y. Yang, G. Huang and Z. Zhang, "Dimension Reduction With Extreme Learning Machine," in IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3906-3918, Aug. 2016, doi: 10.1109/TIP.2016.2570569.
4. Anaconda Software Distribution. (2020). Anaconda Documentation. Anaconda Inc. Retrieved from https://docs.anaconda.com/
5. Cartopy. v0.11.2. 22-Aug-2014. Met Office. UK. https://github.com/SciTools/cartopy/archive/v0.11.2.tar.gz
6. Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2.
7. Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: http://doi.org/10.5334/jors.148
8. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.
9. Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232
10. K. Jarrod Millman and Michael Aivazis. Python for Scientists and Engineers, Computing in Science & Engineering, 13, 9-12 (2011), DOI:10.1109/MCSE.2011.36
11. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
12. Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
13. Travis E. Oliphant. Python for Scientific Computing, Computing in Science & Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58
14. Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)
