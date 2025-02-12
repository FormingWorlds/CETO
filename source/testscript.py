import numpy as np
import numpy.testing as npt
from copy import copy, deepcopy
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

P_estimate = calculate_pressure(D)
D_new = deepcopy(D)
D_new["P_penalty"] = P_estimate


w_gas = 1.0/np.max(np.abs(F_ini[:19])) # Weight for thermodynamic equations from maximum objective function from the 19 reaction values

Value = optimisationfunction(D_new, moles_initial, w_gas)

## Estimate mean const by sampling optimisationfunction over random variables
modelkeys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "P_penalty"]

np.random.seed(D["seed"])

bounds = get_bounds(D)

n_iters = 50
costs = np.zeros(n_iters)
for i in range(n_iters):
    for j in range(len(modelkeys)):
        D_new[modelkeys[j]] = np.random.uniform(bounds[j,0], bounds[j,1])
    costs[i] = optimisationfunction(D_new, moles_initial, w_gas)

cost_smoothed = smoothTriangle(costs, 5)
mean_cost = np.mean(cost_smoothed)

print(mean_cost)