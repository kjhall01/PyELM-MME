from .mme import *
from .reader import *
from .scaler import *
from .plotter import *
from .svd import *
from .spm import *

from subprocess import PIPE, Popen
from pathlib import Path
import os

__author__ = 'Kyle Hall'
__version__ = '0.1.16'

def pyelm1d_main():
	wd = os.path.dirname(os.path.abspath(__file__))
	wd = Path(wd).parents[0] / 'pyelmmme'
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-1D.ipynb').absolute()) ])


def pyelm2d_main():
	wd = os.path.dirname(os.path.abspath(__file__))
	wd = Path(wd).parents[0] / 'src' / 'key'
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-2D.ipynb').absolute()) ])
