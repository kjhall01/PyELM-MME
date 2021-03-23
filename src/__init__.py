from .mme import *
from .reader import *
from .scaler import *
from .plotter import *
from .svd import *
from .spm import *

from subprocess import PIPE, Popen
from pathlib import Path

__author__ = 'Kyle Hall'
__version__ = '0.1.5'

def pyelm1d_main():
	from . import key as _

	wd = Path(_.__path__[0])
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-1D.ipynb').absolute()) ], stdout=PIPE, stderr=PIPE)


def pyelm2d_main():
	from . import key as _

	wd = Path(_.__path__[0])
	proc = Popen(['jupyter', 'notebook', str( (wd / 'PyELM-MME-2D.ipynb').absolute()) ], stdout=PIPE, stderr=PIPE)
