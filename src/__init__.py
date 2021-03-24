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
__version__ = '0.1.13'

def pyelm1d_main():
	wd = os.path.dirname(os.path.abspath(__file__))
	wd = Path(wd).parents[0] / 'src' / 'key'
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-1D.ipynb').absolute()) ], stdout=PIPE, stderr=PIPE)


def pyelm2d_main():
	wd = os.path.dirname(os.path.abspath(__file__))
	wd = Path(wd).parents[0] / 'src' / 'key'
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-2D.ipynb').absolute()) ], stdout=PIPE, stderr=PIPE)
