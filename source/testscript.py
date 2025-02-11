import numpy as np
import numpy.testing as npt
from pathlib import Path
from constants import *

from thermodynamics import shomate, get_Gibbs, calculate_GRT
from activity import get_activity
from model import *
from utilities import *

## Reading model input from config file
sourcedir = Path(__file__).parent
path_to_config = sourcedir / 'defaultconfig.txt'
D = readconfig(path_to_config)

nSi = moles_in_system('Si', D)
nMg = moles_in_system('Mg', D)
nNa = moles_in_system('Na', D)
nFe = moles_in_system('Fe', D)
nC = moles_in_system('C', D)
nH = moles_in_system('H', D)
nO = moles_in_system('O', D)

moles_initial = {'Si' : nSi,
                 'Mg' : nMg,
                 'Fe' : nFe,
                 'Na' : nNa,
                 'H'  : nH,
                 'O'  : nO,
                 'C'  : nC}

F_ini = optimisationfunction_initial(D, moles_initial)
w_gas = 1.0/np.max(np.abs(F_ini[:19])) # Weight for thermodynamic equations from maximum objective function from the 19 reaction values

Value = optimisationfunction(D, moles_initial, w_gas)
print(Value)







