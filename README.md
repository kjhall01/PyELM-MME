# PyELM-MME: A Python Module for Constructing and Comparing Multi-Model Ensemble Methodologies
### Authors: Nachiketa Acharya (Model Science & Statistics), Kyle Hall (AI/ML & Software Dev)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4515069.svg)](https://doi.org/10.5281/zenodo.4515069)

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
## Initializing an MME Object using the Reader() class

### Read 1D Data (model hindcasts & observations, spatially aggregated) with the Reader().read_txt() method. 
```
import pyelmmme as pm
reader = pm.Reader()
hindcast_data = reader.read_txt('your_hindcast_file.csv', has_obs=True, has_years=True, has_header=False) 
mme = pm.MME(hindcast_data)
```

#### 'your_hindcast_file.csv' will be a .csv file with the following format, for N models and M years. 

Year 1 | Observation 1 | Model 1_1 | ... | Model N_1
 --- | --- | --- | --- | ---
 Year 2 | Observation 2 | Model 1_2 | ... | Model N_2
 Year 3 | Observation 3 | Model 1_3 | ... | Model N_3
 . | | | | .
 . | | | | .
 Year M | Observation M | Model 1_M | ... | Model N_M
 
#### If your file doesn't have year labels, or has a header, you'll need to adjust the 'has_header' and 'has_years keyword arguments, whose defaults are has_header=False and has_years=True. 

#### Note that for initializing an MME object requires historical observations - how would one train statistical models without them? 

### Read 2D Data (model hindcasts & observations) with the Reader().read_txt() method. 
```
import pyelmmme as pm
reader = pm.Reader()
hindcast_data = reader.read_txt('your_hindcast_file.csv', has_obs=True, has_years=True, has_header=False) 
mme = pm.MME(hindcast_data)
```

#### After reading in data, MME's internal variables will have been initialized. Next, we can use whichever MME methodologies we want to call the construct_crossvalidated_mme_hindcasts method.
#### 'Multi-Point' MMEs accept data in ncdf format only. There are two methods, depending on the format of the data. If all model data and observations are DataArrays within one DataSet in one file, use:
```
mme.read_full_ncdf('name_of_ncdf_file', latitude_key='latitude', longitude_key='longitude', time_key='time', observations_key='observations', using_datetime=True, axis_order='xyt', is_forecast=False)
```
#### Note that you must pass the names of the coordinates within the ncdf file to the appropriate keyword argument, and provide teh name of the observations DataArray.
#### Set using_datetime to False if not using standard datatime objects as time indexes.
#### axis_order refers to the shape of the DS.variable.values array - it needs to be of shape (longitude, latitude, time) ('xyt'). if yours is of shape (time, longitude, latitude) pass axis_order='txy', we have automated the reshaping for you.

#### Similarly, if you have one model per ncdf file, and an observations file separately, place them all in the same subfolder and pass the path to it to:

```
mme.read_multiple_ncdf('name_of_ncdf_dir', 'name_of_obs_file' latitude_key='latitude', longitude_key='longitude', time_key='time', obs_time_key='time', using_datetime=True, axis_order='xyt', is_forecast=False)
```
#### Each methodology is referred to by a code (a string) and takes a different set of keyword arguments. Hindcasts created are stored internally.

## MME Methodologies

### Standard Ensemble Mean (EM)
##### simple arithmetic mean
```
mme.construct_crossvalidated_mme_hindcasts('EM', xval_window=3)
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.


### Multiple Linear Regression (MLR)
##### Standard multiple regression between model values and observations
```
mme.construct_crossvalidated_mme_hindcasts('MLR'))
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly' or None
- fit_intercept (bool, default=True): whether to include intercepts in calculation of model coefficients  


### Extreme Learning Machine (ELM)
##### Artificial Neural Network with randomized hidden layer weights and output weights solved by ELM method.
```
mme.construct_crossvalidated_mme_hindcasts('ELM', hidden_layer_neurons=5, activation='sigm', standardization='minmax', minmax_range=[-1,1])
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly', 'minmax' or None
- minmax_range (list, default=[-1,1]): minimum and maximum values to scale data to when using minmax scaling.
- hidden_layer_neurons (int, default=5): Number of neurons in ELM's hidden layer
- activation (str, default='sigm'): string representing activation function. 'sigm', 'lin', 'tanh', 'rbf_l1', 'rbf_l2', rbf_linf'



### Once you have constructed all the MME hindcasts you want to examine, calculated their skills using the following:

## Skill Metrics
#### Calculate Skill Metrics as Follows
```
mme.Pearson() # Pearson Correlation 
mme.Spearman() # Spearman Correlation 
mme.MAE() # Mean Absolute Error 
mme.MSE() # Mean Squared Error 
mme.MSE(squared=False) # Root Mean Squared Error 
mme.IOA() #Index of Agreement 
```
#### Or, pass a list of keys to the convenience function: 
```
mme.train_mmes(['





## You can train forecast models on all available data with:
```
mme.train_forecast_model(...)
```
#### This takes the same arguments as 'construct_crossvalidated_mme_hindcasts' - call multpile times for multiple mme methodologies

#### and then you can calculate real time forecasts by reading in new data in the same way as above, but with the keyword argument 'is_forecast' = True, and then calling
```
mme.forecast(...)
```
#### with the same args as train_forecast_model. You won't be able to plot results  that you havent calculated yet.

## See PyMME-1D.ipynb and PyMME-2D.ipynb for examples, or just use those!

##Good Luck!

## Works Cited

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
