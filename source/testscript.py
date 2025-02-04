import numpy as np
from pathlib import Path
from constants import *

from thermodynamics import shomate, get_Gibbs, calculate_GRT
from activity import get_activity
from model import optimisationfunction
from utilities import *

## Reading model input from config file
projectdir = Path(__file__).parent
path = projectdir / 'defaultconfig.txt'
print(path)
D = readconfig(path)

Fs = optimisationfunction(D)
print(Fs)
