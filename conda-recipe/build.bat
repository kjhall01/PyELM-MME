"%PYTHON%" "%RECIPE_DIR%"/setup.py install
"%PYTHON%" -m pip install https://files.pythonhosted.org/packages/a2/9a/e68e1fd4ec6388979737be537d05b34626840b353e5db6e05951286614c0/hpelm-1.0.10-py3-none-any.whl
if errorlevel 1 exit 1
