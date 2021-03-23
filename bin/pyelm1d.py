from subprocess import Popen, PIPE
from pathlib import Path
import pyelmmme as _


if __name__=='__main__':
	wd = Path(_.__path__)
	proc = Popen(['jupyter', 'notebook', str( (wd / 'bin' / PyELM-MME-1D.ipynb').absolute()) ] stdout=PIPE, stderr=PIPE) 
 
