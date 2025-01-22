import numpy as np
from pathlib import Path
from constants import *

from thermodynamics import shomate, get_Gibbs, calculate_GRT
from activity import get_activity
from model import optimisationfunction
from utilities import readconfig

## Reading model input from config file
projectdir = Path(__file__).parent
path = projectdir / 'defaultconfig.txt'
print(path)
D = readconfig(path)

T_min = 1300
T_max = D["T_eq"]
T = np.linspace(T_min, T_max, 200)

f1, f2, f3 = optimisationfunction(D)
print(f1, f2, f3)
