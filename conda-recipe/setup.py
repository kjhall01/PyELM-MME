from setuptools import *

with open('README.md', 'r', encoding='utf-8') as fh:
	long_description= fh.read()


setup(
    name = "pyelmmme",
    version = "0.1.5",
    author = "Kyle Hall, Nachiketa Acharya",
    author_email = "hallkjc01@gmail.com",
    description = ("Using HPELM and Extreme Learning Machine to make climate forecasts and Multi Model Ensembles "),
    license = "MIT",
    keywords = "AI HPELM ELM Extreme Learning Machine Climate Forecasting Multi Model Ensemble",
    url = "https://github.com/kjhall01/PyELM-MME",
    packages=['pyelmmme'],
	package_dir={'pyelmmme': 'src'},
	python_requires=">=3.0",
    long_description=long_description,
	long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
    ],
)
