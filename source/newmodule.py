from __future__ import annotations
import numpy as np
from pathlib import Path

from readconfig import readconfig

## Obtain input from config file
projectdir = Path(__file__).parent
path = projectdir / 'defaultconfig.txt'
D = readconfig(path)

## total moles of chemical constituents, extensive constraints on system
def moles_in_system(el, D):
    """Computes the number of moles of a certain element present within the system based on input data.
       Parameters:
       el (string)              : string representing chemical symbol of element to be computed for.
                                  valid examples are H, C, O, Si, Mg, Na, Fe. 
       D  (dict)                : Dictionary representing input for model; must contain at least fractional
                                  compositions in terms of molecular species in melt, metal and gas phases and
                                  amount of moles in each phase.
    
       Returns:
       nElement (float)         : Number of moles of requested element in system across melt, metal and gas phases.
       """
    el_inmelt = 0
    el_ingas = 0
    el_inmetal = 0
    for i in D:
        if el in i and '_melt' in i:
            if (el+'2') not in i and (el+'3') not in i and (el+'4') not in i:
                el_inmelt += D[i]
            else:
                for j in range(2,5):
                    if (el+str(j)) in i:
                        el_inmelt += j*D[i]
                    else:
                        pass
        elif el in i and '_metal' in i:
            if (el+'2') not in i and (el+'3') not in i and (el+'4') not in i:
                el_inmetal += D[i]
            else:
                for j in range(2,5):
                    if (el+str(j)) in i:
                        el_inmetal += j*D[i]
                    else:
                        pass
        elif el in i and '_gas' in i:
            if (el+'2') not in i and (el+'3') not in i and (el+'4') not in i:
                el_ingas += D[i]
            else:
                for j in range(2,5):
                    if (el+str(j)) in i:
                        el_ingas += j*D[i]
                    else:
                        pass
            
    return D["moles_melt"]*el_inmelt + D["moles_metal"]*el_inmetal + D["moles_atm"]*el_ingas

nSi = moles_in_system('Na', D)
nMg = moles_in_system('Mg', D)
nNa = moles_in_system('Na', D)
nFe = moles_in_system('Fe', D)
nH = moles_in_system('H', D)
nC = moles_in_system('C', D)
nO = moles_in_system('O', D)

## Boundary Conditions, hardcoded for now. Conditions are imposed on species mole fractions (indices 0-25), 
## moles of phases (indices 26, 27, 28) and pressure (index 29).
bounds = np.zeros((30))
for i in range(0, 26):       # we set boundary conditions on mole fractions for all species, and use modified
    if i == 7:               # conditions for a few reactions; including H2 in melt (i==7) and H2O (i==8)
        bounds[i,0] = 1.0e-15
        bounds[i,1] = 0.4
    elif i == 8:
        bounds[i,0] = 1.0e-12
        bounds[i,1] = 0.4

    elif i == 21 or i == 22 or i == 23:
        bounds[i, 0] = 1.0e-15
        bounds[i, 1] = 0.99999
    else:
        bounds[i,0] = 1.0e-20 
        bounds[i,1] = 0.99999

bounds[26, 0] = D["moles_atm"]*1.0e-20
bounds[26, 1] = D["moles_atm"]*50.0

bounds[27, 0] = D["moles_melt"]*0.5
bounds[27, 1] = D["moles_melt"]*2.0

bounds[28, 0] = D["moles_metal"]*0.5
bounds[28, 1] = D["moles_metal"]*2.0

bounds[29, 0] = 1.0e-3
bounds[29, 1] = 9.0e5