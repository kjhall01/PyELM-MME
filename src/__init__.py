from .mme import *
from .reader import *
from .scaler import *
from .plotter import *
from .svd import *
from .spm import *

from subprocess import PIPE, Popen
from pathlib import Path
import os, json

__author__ = 'Kyle Hall'
__version__ = '0.1.18'

pyelm1d = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. File I/O Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "hindcast_data_file = 'test_data/NMME_data_BD.csv' #data used for cross-validated hindcast skill analysis, and to train forecast model\\n",
    "hindcast_has_years = True"\\n,
    "hindcast_has_header = False\\n",
    "hindcast_has_obs = True #NOTE: This is mandatory \\n",
    "hindcast_export_file = 'bd.csv' #'None' or the name of a file to save cross validated hindcasts \\n",
    "\\n",
    "forecast_data_file = 'test_data/NMME_data_BD_forecast.csv' #data fed to trained model to produce forecasts, or None\\n",
    "forecast_has_years = True\\n",
    "forecast_has_header = False\\n",
    "forecast_has_obs = True #NOTE: for Forecasting, observations are optional\\n",
    "forecast_export_file = 'bd_rtf.csv'\\n",
    "\\n",
    "variable = 'Precipitation (mm/day)'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Cross-Validated Hindcast Skill Evaluation\\n",
    "#### 2a. Analysis Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "mme_methodologies = ['EM', 'MLR', 'ELM'] #list of MME methodologies to use \\n",
    "skill_metrics = [ 'MAE', 'IOA', 'MSE', 'RMSE', 'PearsonCoef', 'SpearmanCoef'] #list of metrics to compute - available: ['SpearmanCoef', 'SpearmanP', 'PearsonCoef', 'PearsonP', 'MSE', 'MAE', 'RMSE', 'IOA']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2b. Model Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "args = {\\n",
    "    #EnsembleMean settings\\n",
    "    'em_xval_window': 1,               #odd number - behavior undefined for even number\\n",
    "\\n",
    "    #MLR Settings\\n",
    "    'mlr_fit_intercept': True,         #Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered) (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)\\n",
    "    'mlr_xval_window': 1,               #odd number - behavior undefined for even number\\n",
    "    'mlr_standardization': None,        #'std_anomaly' or None\\n",
    "\\n",
    "    #ELM Settings \\n",
    "    'elm_xval_window': 1,              #odd number - behavior undefined for even number\\n",
    "    'elm_hidden_layer_neurons':10,     #number of hidden layer neurons - overridden if using PCA init\\n",
    "    'elm_activation': 'sigm',          #“lin” for linear, “sigm” or “tanh” for non-linear, “rbf_l1”, “rbf_l2” or “rbf_linf” for radial basis function neurons (https://hpelm.readthedocs.io/en/latest/api/elm.html)\\n",
    "    'elm_standardization' : 'minmax',  #'minmax' or 'std_anomaly' or None\\n",
    "    'elm_minmax_range': [-1, 1]        #choose [minimum, maximum] values for minmax scaling. ignored if not using minmax scaling\\n",
    "}\\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2c. Model Construction - Do Not Edit "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "from pyelmmme import * \\n",
    "\\n",
    "reader = Reader()  #Object that will handle our input data\\n",
    "data = reader.read_txt(hindcast_data_file, has_years=hindcast_has_years, has_obs=hindcast_has_obs, has_header=hindcast_has_header)\\n",
    "mme = MME(data)\\n",
    "mme.train_mmes(mme_methodologies, args)\\n",
    "mme.measure_skill(skill_metrics)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2d. Cross-Validated Hindcast Timeline - Do Not Edit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
	 "outputs": [],
   "source": [
    "ptr = Plotter(mme)\\n",
    "ptr.timeline(methods=mme_methodologies, members=False, obs=True, var=variable)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2e. Cross-Validated Hindcast Skill Metrics & Distributions - Do Not Edit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.skill_matrix(methods=mme_methodologies, metrics=skill_metrics, obs=True, members=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.box_plot(methods=mme_methodologies, obs=True, members=False, var=variable)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2f. Saving MME & Exporting Cross-Validated Hindcasts - Do Not Edit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "mme.export_csv(hindcast_export_file)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Real Time Forecasting\\n",
    "#### 3a. RTF Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "forecast_methodologies = ['EM', 'MLR', 'ELM' ]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 3b. Computation - do not edit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "fcst_data = reader.read_txt(forecast_data_file, has_years=forecast_has_years, has_obs=forecast_has_obs)\\n",
    "mme.add_forecast(fcst_data)\\n",
    "mme.train_rtf_models(forecast_methodologies, args)\\n",
    "mme.make_RTFs(forecast_methodologies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "ptr.bar_plot(methods=mme_methodologies, members=False, obs=forecast_has_obs, var=variable)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "mme.export_csv(forecast_export_file, fcst='forecasts', obs=forecast_has_obs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py38",
   "language": "python",
   "name": "py38"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""


pyelm2d = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Data I/O Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Directory data loading parameters - training data\\n",
    "hindcast_directory = 'test_data/hindcasts/'              #name of directory with separate ncdf files for observations and each model \\n",
    "hcst_observations_file = 'chirps_hindcast.nc'    #name of file within ncdf_directory with observations data\\n",
    "hindcast_export_file = 'tanzania.nc'\\n",
    "\\n",
    "#Directory data loading parameters - forecast data\\n",
    "forecast_directory = 'test_data/forecasts/'              #name of directory with separate ncdf files for observations and each model \\n",
    "fcst_observations_file = 'chirps_forecast.nc' #or 'None'\\n",
    "forecast_export_file = 'tanzania_fcst.nc'\\n",
    "\\n",
    "#shared loading parameters\\n",
    "latitude_key = 'Y'                        #name of latitude coordinate in your ncdf file\\n",
    "longitude_key='X'                         #name of longitude coordinate in your ncdf file \\n",
    "time_key = 'S'                            #name of time coordinate in your ncdf file \\n",
    "obs_time_key = 'T'\\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. MME Skill Evaluation\\n",
    "#### 2a. Analysis Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#analysis parameters \\n",
    "mme_methodologies = ['EM', 'MLR', 'ELM' ] #'MLR', 'PCR', 'SVD', 'ELM', 'EWP',list of MME methodologies to use - available: ['EM', 'MLR', 'ELM', 'Ridge'] 'SLFN' also works but is slow for 2D data\\n",
    "skill_metrics = ['SpearmanCoef', 'PearsonCoef', 'RMSE', 'MSE',  'MAE', 'IOA'] #list of metrics to compute - available: ['SpearmanCoef', 'SpearmanP', 'PearsonCoef', 'PearsonP', 'MSE', 'MAE', 'RMSE', 'IOA']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2b. Model Parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "args = {\\n",
    "    #EnsembleMean settings\\n",
    "    'em_xval_window': 1,               #odd number - behavior undefined for even number\\n",
    "\\n",
    "    #MLR Settings\\n",
    "    'mlr_fit_intercept': True,         #Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered) (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)\\n",
    "    'mlr_xval_window': 1,               #odd number - behavior undefined for even number\\n",
    "    'mlr_standardization': None,        #'std_anomaly' or None\\n",
    "\\n",
    "    #ELM Settings \\n",
    "    'elm_xval_window': 1,              #odd number - behavior undefined for even number\\n",
    "    'elm_hidden_layer_neurons':3,     #number of hidden layer neurons - overridden if using PCA init\\n",
    "    'elm_activation': 'sigm',          #“lin” for linear, “sigm” or “tanh” for non-linear, “rbf_l1”, “rbf_l2” or “rbf_linf” for radial basis function neurons (https://hpelm.readthedocs.io/en/latest/api/elm.html)\\n",
    "    'elm_standardization' : 'std_anomaly',  #'minmax' or 'std_anomaly' or None\\n",
    "    'elm_minmax_range': [-1, 1]        #choose [minimum, maximum] values for minmax scaling. ignored if not using minmax scaling\\n",
    "}\\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2c. Model Construction\\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\\n",
      "Computing MMEs\\n",
      "EM [*************************]\\n",
      "MLR [*************************]\\n",
      "ELM [*************************]\\n"
     ]
    }
   ],
   "source": [
    "from pyelmmme import * \\n",
    "\\n",
    "reader = Reader()  #Object that will handle our input data\\n",
    "data = reader.read_multiple_ncdf(hindcast_directory, observations_filename=hcst_observations_file, latitude_key=latitude_key, longitude_key=longitude_key,obs_time_key=obs_time_key, time_key=time_key)\\n",
    "mme = MME(data, verbose=True)\\n",
    "mme.train_mmes(mme_methodologies, args)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2d. Skill Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\\n",
      "Analyzing Skill\\n",
      "SpearmanCoef\\n",
      "  ELM [*************************]\\n",
      "  EM [*************************]\\n",
      "  MLR [*************************]\\n",
      "  Model1 [*************************]\\n",
      "  Model2 [*************************]\\n",
      "  Model3 [*************************]\\n",
      "  Obs [*************************]\\n",
      "PearsonCoef\\n",
      "  ELM [*************************]\\n",
      "  EM [*************************]\\n",
      "  MLR [*************************]\\n",
      "  Model1 [*************************]\\n",
      "  Model2 [*************************]\\n",
      "  Model3 [*************************]\\n",
      "  Obs [*************************]\\n",
      "RMSE\\n",
      "  ELM [*************************]\\n",
      "  EM [*************************]\\n",
      "  MLR [*************************]\\n",
      "  Model1 [*************************]\\n",
      "  Model2 [*************************]\\n",
      "  Model3 [*************************]\\n",
      "  Obs [*************************]\\n",
      "MSE\\n",
      "  ELM [*************************]\\n",
      "  EM [*************************]\\n",
      "  MLR [*************************]\\n",
      "  Model1 [*************************]\\n",
      "  Model2 [*************************]\\n",
      "  Model3 [*************************]\\n",
      "  Obs [*************************]\\n",
      "MAE\\n",
      "  ELM [*************************]\\n",
      "  EM [*************************]\\n",
      "  MLR [*************************]\\n",
      "  Model1 [*************************]\\n",
      "  Model2 [*************************]\\n",
      "  Model3 [*************************]\\n",
      "  Obs [*************************]\\n",
      "IOA\\n",
      "  ELM [*************************]\\n",
      "  EM [*************************]\\n",
      "  MLR [*************************]\\n",
      "  Model1 [*************************]\\n",
      "  Model2 [*************************]\\n",
      "  Model3 [*************************]\\n",
      "  Obs [*************************]\\n"
     ]
    }
   ],
   "source": [
    "mme.measure_skill(skill_metrics)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2e. Cross-Validated Hindcast Skill Maps "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr = Plotter(mme)\\n",
    "ptr.map_skill(methods=mme_methodologies, metrics=skill_metrics, members=True, obs=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2e. Cross-Validated Hindcast Single-Point Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.timeline(methods=mme_methodologies, members=False, obs=True, point=[-6, 36]) #point in degrees lat/long"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.box_plot(methods=mme_methodologies, members=False, obs=True, point=[-6, 36]) #point in degrees lat/long"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.skill_matrix(methods=mme_methodologies, metrics=skill_metrics, members=True, obs=True, point=[-9, 34]) #point in degrees lat/long"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 2f. Saving MME - Do Not Edit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "saved to tanzania.nc\\n"
     ]
    }
   ],
   "source": [
    "mme.export_ncdf(hindcast_export_file)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Real Time Forecasting\\n",
    "#### 4a. Real Time Forecast Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#forecast analysis parameters \\n",
    "forecast_methodologies = ['EM', 'MLR', 'ELM' ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\\n",
      "Training RTF Models\\n",
      "EM [*************************]\\n",
      "MLR [*************************]\\n",
      "ELM [*************************]\\n"
     ]
    }
   ],
   "source": [
    "fcst_data = reader.read_multiple_ncdf(forecast_directory, observations_filename=fcst_observations_file, latitude_key=latitude_key, longitude_key=longitude_key,obs_time_key=obs_time_key, time_key=time_key)\\n",
    "mme.add_forecast(fcst_data)\\n",
    "mme.train_rtf_models(forecast_methodologies, args)\\n",
    "mme.make_RTFs(forecast_methodologies)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4b. Deterministic Forecasts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.map_forecasts(methods=mme_methodologies, fcst='forecasts', obs=True,  members=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 4b. Deterministic Forecasts - Pointwise Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "ptr.bar_plot(methods=mme_methodologies, fcst='forecasts', members=False, obs=False, point=[-9,37])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "py38",
   "language": "python",
   "name": "py38"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""

def pyelm1d_main():
	wd = os.path.dirname(os.path.abspath(__file__))
	wd = Path(wd).parents[0] / 'pyelmmme'
	with open(str((wd / 'PyELM-MME-1D.ipynb').absolute()), 'w') as f:

		f.write(pyelm1d)

	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-1D.ipynb').absolute()) ])


def pyelm2d_main():
	wd = os.path.dirname(os.path.abspath(__file__))
	wd = Path(wd).parents[0] / 'pyelmmme'
	with open(str((wd / 'PyELM-MME-2D.ipynb').absolute()), 'w') as f:
		f.write(pyelm2d)
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-2D.ipynb').absolute()) ])
