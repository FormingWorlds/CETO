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

T_min = 1300
T_max = D["T_eq"]
T = np.linspace(T_min, T_max, 200)

gpm_gas, gpm_melt, gpm_metal = gpm_phases(D)
moles_total = D["moles_atm"] + D["moles_melt"] + D["moles_metal"]

molefrac_atm = D["moles_atm"] / moles_total
molefrac_melt = D["moles_melt"] / moles_total
molefrac_metal = D["moles_metal"] / moles_total

grams_atm = molefrac_atm * gpm_gas
grams_melt = molefrac_melt * gpm_melt
grams_metal = molefrac_metal * gpm_metal

totalmass = grams_atm + grams_melt + grams_metal

massfrac_atm = grams_atm / totalmass
massfrac_melt = grams_melt / totalmass
massfrac_metal = grams_metal / totalmass

fratio = massfrac_atm/(1.0-massfrac_atm)
P_initial = 1.2e6*fratio*(D["M_p"])**(2/3)
print(f"Estimated initial surface pressure: {P_initial} bar ({P_initial*0.0001} GPa)")
