import numpy as np
from pathlib import Path
from constants import *

from thermodynamics import shomate, get_Gibbs, calculate_GRT
from activity import get_activity
from utilities import readconfig

## Reading model input from config file
projectdir = Path(__file__).parent
path = projectdir / 'defaultconfig.txt'
print(path)
D = readconfig(path)

T_min = 1300
T_max = D["T_eq"]
T = np.linspace(T_min, T_max, 200)

GRTs = calculate_GRT(T)

GRT1, GRT2, GRT3, GRT4, GRT5, GRT6, GRT7, GRT8, GRT9, GRT10 = GRTs[:10]
GRT11, GRT12, GRT13, GRT14, GRT15, GRT16, GRT17, GRT18, GRT19 = GRTs[10:]


