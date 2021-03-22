# PyELM-MME: A Python Module for Constructing and Comparing Multi-Model Ensemble Methodologies
### Authors: Nachiketa Acharya (Model Science & Statistics), Kyle Hall (AI/ML & Software Dev)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4515069.svg)](https://doi.org/10.5281/zenodo.4515069)

## Requirements:
	* cartopy: conda install -c conda-forge cartopy
	* xarray: conda install xarray
	* scipy, numpy, matplotlib, pandas: should be standard anaconda distribution
	* scikit-learn: conda install sklearn
	* hpelm: pip install hpelm
## Tutorial:
#### PyMME is centered around the Multi-Model Ensemble (MME) class.

```
import pymme as pm
```

```
mme = pm.MME()
```
#### Each MME() object is either of type 'Single-Point', or 'Multi-Point'. This is determined by the format of the input data, and the method used to load it.

#### 'Single-Point' MMEs handle time series data for one lat/long point.
```
mme.read_txt('example.csv', delimiter=',')
```
#### The read_txt method loads csv data into MME's internals. It also sets the .type attribute to 'Single-Point', to ensure MME's Multi-Point methods aren't called accidentally.

#### 'example.csv' will be a csv file with the following format, for N models and M years:

Year 1 | Observation 1 | Model 1_1 | ... | Model N_1
 --- | --- | --- | --- | ---
 Year 2 | Observation 2 | Model 1_2 | ... | Model N_2
 Year 3 | Observation 3 | Model 1_3 | ... | Model N_3
 . | | | | .
 . | | | | .
 Year M | Observation M | Model 1_M | ... | Model N_M

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

#### After reading in data, MME's internal variables will have been initialized. Next, we can use whichever MME methodologies we want to call the construct_crossvalidated_mme_hindcasts method.

#### Each methodology is referred to by a code (a string) and takes a different set of keyword arguments. Hindcasts created are stored internally.

## MME Methodologies

### Multiple Linear Regression (MLR)
##### Standard multiple regression between model values and observations
```
mme.construct_crossvalidated_mme_hindcasts('MLR'))
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly' or None
- fit_intercept (bool, default=True): whether to include intercepts in calculation of model coefficients  

### Principle Components Regression (PCR)
##### Multiple regression between PCA-transformed orthogonal model data and observations
```
mme.construct_crossvalidated_mme_hindcasts('PCR', pca_variability=0.9)
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly' or None
- fit_intercept (bool, default=True): whether to include intercepts in calculation of model coefficients
- pca_variability (float [0.0, 1.0)): percent of variability to retain during PCA transformation

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


### PCA-ELM
##### ELM but hidden layer weights are initialized to Eigenvectors calculated during PCA, and number of hidden layer neurons chosen by n_components retained during PCA to keep X% variability
```
mme.construct_crossvalidated_mme_hindcasts('PCA-ELM', hidden_layer_neurons=5, activation='sigm', standardization='minmax', minmax_range=[-1,1], pca_variability=0.9, W=True)
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly', 'minmax' or None
- minmax_range (list, default=[-1,1]): minimum and maximum values to scale data to when using minmax scaling.
- hidden_layer_neurons (int, default=5): Number of neurons in ELM's hidden layer
- activation (str, default='sigm'): string representing activation function. 'sigm', 'lin', 'tanh', 'rbf_l1', 'rbf_l2', rbf_linf'
- pca_variability (float [0.0, 1.0)): percent of variability to retain during PCA transformation
- W (boolean, default=False): True Required for PCA-ELM - whether to use PCA-ELM weight initialization scheme


### ELM with PCA (ELM-PCA)
##### ELM between PCA-transformed orthogonal model data and untouched observations. PCA-ELM weight initialization or hidden layer neuron count methodology not applied
```
mme.construct_crossvalidated_mme_hindcasts('PCA-ELM', hidden_layer_neurons=5, activation='sigm', standardization='minmax', minmax_range=[-1,1], pca_variability=0.9, W=False)
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly', 'minmax' or None
- minmax_range (list, default=[-1,1]): minimum and maximum values to scale data to when using minmax scaling.
- hidden_layer_neurons (int, default=5): Number of neurons in ELM's hidden layer
- activation (str, default='sigm'): string representing activation function. 'sigm', 'lin', 'tanh', 'rbf_l1', 'rbf_l2', rbf_linf'
- pca_variability (float [0.0, 1.0)): percent of variability to retain during PCA transformation
- W (boolean, default=False): False Required for ELM-PCA - whether to use PCA-ELM weight initialization scheme

### Singular Value Decomposition (SVD)
##### MLR using Singular Value Decomposition
```
mme.construct_crossvalidated_mme_hindcasts('SVD', xval_window=3, standardization='std_anomaly')
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly' or None

### Standard Ensemble Mean (EM)
##### simple arithmetic mean
```
mme.construct_crossvalidated_mme_hindcasts('EM', xval_window=3)
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.

### Bias-Corrected Ensemble Mean (BCEM)
##### mean of models projected onto observation distribution
```
mme.construct_crossvalidated_mme_hindcasts('BCEM', xval_window=3, standardization='std_anomaly')
```
##### accepts keyword arguments:
- xval_window (int, default=3): number of years to leave out in each cross validation round. Must be odd.
- standardization (str or None, default=None): scaling method to apply to data. Either 'std_anomaly' or None. 'std_anomaly' required for BCEM

### Once you have constructed all the MME hindcasts you want to examine, calculated their skills using the following:

## Skill Metrics

##### Pearson Correlation
```
mme.Pearson()
```

##### Spearman Correlation
```
mme.Spearman()
```

##### Mean Absolute Error
```
mme.MAE()
```

##### Mean Squared Error
```
mme.MSE()
```

##### Root Mean Squared Error
```
mme.MSE(squared=False)
```

##### Index of Agreement
```
mme.IOA()
```


## Once you've calculated all skill metrics you want, plot using
```
mme.plot()
```
### with keyword argument 'setting' equal to one of the following settings:
##### for single-point MME's, or if you pass point=[lat index, lon index]:
- 'Cross-Validated Hindcasts': timeline of xvalidated forecasts
- 'Real Time Forecasts': timeline of realtime forecasts, if more than one
- 'Training Forecasts': timeline  of non-xvalidated forecasts calculated by forecast models trained on all data
- 'xval_hindcast_skill': grid of skills of xvalidated hindacsts
- 'training_forecast_skill': grid of skills of forecasts models on training data
- 'boxplot': show boxplots of distributions of xvalidated hindcasts data
- 'training_forecast_boxplot': show boxplots of distributions of non-xvaldated hindcasts data
##### for multi-point MMEs:
- "Real Time Deterministic Forecast": map of deterministic forecast over lat/long grid
- 'xval_hindcast_skill': map of skills of xvalidated hindacsts
- 'training_forecast_skill': map of skills of forecasts models on training data


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

1. A. Akusok, K. Björk, Y. Miche and A. Lendasse, "High-Performance Extreme Learning Machines: A Complete Toolbox for Big Data Applications," in IEEE Access, vol. 3, pp. 1011-1025, 2015, doi: 10.1109/ACCESS.2015.2450498.
2. Anaconda Software Distribution. (2020). Anaconda Documentation. Anaconda Inc. Retrieved from https://docs.anaconda.com/
3. Cartopy. v0.11.2. 22-Aug-2014. Met Office. UK. https://github.com/SciTools/cartopy/archive/v0.11.2.tar.gz
4. Castaño, A., Fernández-Navarro, F. & Hervás-Martínez, C. PCA-ELM: A Robust and Pruned Extreme Learning Machine Approach Based on Principal Component Analysis. Neural Process Lett 37, 377–392 (2013). https://doi.org/10.1007/s11063-012-9253-x
5. Castaño, A., Fernández-Navarro, F., Riccardi, A. et al. Enforcement of the principal component analysis–extreme learning machine algorithm by linear discriminant analysis. Neural Comput & Applic 27, 1749–1760 (2016). https://doi.org/10.1007/s00521-015-1974-0
6. Fernando Pérez and Brian E. Granger. IPython: A System for Interactive Scientific Computing, Computing in Science & Engineering, 9, 21-29 (2007), DOI:10.1109/MCSE.2007.53
7. Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2.
8. Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: http://doi.org/10.5334/jors.148
9. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.
10. J. Kim, H. Shin, Y. Lee and M. Lee, "Algorithm for Classifying Arrhythmia using Extreme Learning Machine and Principal Component Analysis," 2007 29th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, Lyon, 2007, pp. 3257-3260, doi: 10.1109/IEMBS.2007.4353024.
11. Jeff Reback, Wes McKinney, jbrockmendel, Joris Van den Bossche, Tom Augspurger, Phillip Cloud, … Mortada Mehyar. (2020, March 18). pandas-dev/pandas: Pandas 1.0.3 (Version v1.0.3). Zenodo. http://doi.org/10.5281/zenodo.3715232
12. K. Jarrod Millman and Michael Aivazis. Python for Scientists and Engineers, Computing in Science & Engineering, 13, 9-12 (2011), DOI:10.1109/MCSE.2011.36
13. L. L. C. Kasun, Y. Yang, G. Huang and Z. Zhang, "Dimension Reduction With Extreme Learning Machine," in IEEE Transactions on Image Processing, vol. 25, no. 8, pp. 3906-3918, Aug. 2016, doi: 10.1109/TIP.2016.2570569.
14. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
15. Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
16. Travis E. Oliphant. Python for Scientific Computing, Computing in Science & Engineering, 9, 10-20 (2007), DOI:10.1109/MCSE.2007.58
17. Wes McKinney. Data Structures for Statistical Computing in Python, Proceedings of the 9th Python in Science Conference, 51-56 (2010)
