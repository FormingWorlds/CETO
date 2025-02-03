from __future__ import annotations
import numpy as np
from numpy import log as ln
from numpy import log10 as log

from pathlib import Path

from constants import *

def readconfig(path):
    """"Reads model config file (.txt) and converts input into dictionary for easy handling.
        Parameters:
        path (str)              : absolute path to model config file.
        
        Returns:
        D (dict)                : dictionary containing model input"""
    input = np.genfromtxt(path, dtype=str)
    data = []
    for i in range(len(input)):
        item = input[i]
        if '.' in item or 'e+' in item or 'e-' in item:
            try:
                newitem = float(item)
            except:
                newitem = item
        elif item == 'True' or item == "true":
            try:
                newitem = True
            except:
                newitem = item
        elif item == 'False' or item == 'false':
            try:
                newitem = False
            except:
                newitem = item
        else:
            try:
                newitem = int(item)
            except:
                newitem = item
        data.append(newitem)

    datakeys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "M_p", "T_surface", "T_eq", 
            "P_penalty", "wt_massbalance", "wt_summing", "wt_atm", "wt_solub", "wt_melt", "wt_evap", "wt_f1",
            "wt_f2", "wt_f3", "wt_f4", "wt_f5", "wt_f6", "wt_f7", "wt_f8", "wt_f9", "wt_f10", "wt_f11",
            "wt_f12", "wt_f13", "wt_f14", "wt_f15", "wt_f16", "wt_f17", "wt_f18", "wt_f19", "wt_f20", "wt_f21",
            "wt_f22", "wt_f23", "wt_f24", "wt_f25", "wt_f26", "wt_f27", "wt_f28", "wt_f29", "seed", "niters",
            "offset_MCMC", "bool_unreactive_metal", "bool_nonideal_mixing"]
    
    D = dict(zip(datakeys, data))
    return D

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
    for key in D:
        if el in key and '_melt' in key:
            if (el+'2') not in key and (el+'3') not in key and (el+'4') not in key:
                el_inmelt += D[key]
            else:
                for coeff in range(2,5):
                    if (el+str(coeff)) in key:
                        el_inmelt += coeff*D[key]
                    else:
                        pass
        elif el in key and '_metal' in key:
            if (el+'2') not in key and (el+'3') not in key and (el+'4') not in key:
                el_inmetal += D[key]
            else:
                for coeff in range(2,5):
                    if (el+str(coeff)) in key:
                        el_inmetal += coeff*D[key]
                    else:
                        pass
        elif el in key and '_gas' in key:
            if (el+'2') not in key and (el+'3') not in key and (el+'4') not in key:
                el_ingas += D[key]
            else:
                for j in range(2,5):
                    if (el+str(coeff)) in key:
                        el_ingas += coeff*D[key]
                    else:
                        pass
            
    return D["moles_melt"]*el_inmelt + D["moles_metal"]*el_inmetal + D["moles_atm"]*el_ingas

def gpm_phases(D):
    """Computes gram per mole of phase for atmosphere, silicate melt and metal core via species
       molecular weights.
       Parameters:
       D (dict)             : Dictionary containing model input, retrieves mole fractions of species
                            in the three respective phases.
       Returns:
       gpm_gas (float)      : total number of grams per mole of atmosphere
       gpm_melt (float)     : total number of grams per mole of silicate mantle
       gpm_metal (float)    : total number of grams per mole of metal core."""
    gpm_gas = 0
    gpm_melt = 0
    gpm_metal = 0
    for key in D:
        if "_gas" in key and 'moles' not in key and 'wt' not in key:
            speciesname = key.replace('_gas','')
            gpm_gas += D[key]*molwts[speciesname]
        elif "_melt" in key and 'moles' not in key and 'wt' not in key:
            speciesname = key.replace('_melt','')
            gpm_melt += D[key]*molwts[speciesname]
        elif "_metal" in key and 'moles' not in key and 'wt' not in key and 'bool' not in key:
            speciesname = key.replace('_metal','')
            gpm_metal += D[key]*molwts[speciesname]
        else:
            pass
    return gpm_gas, gpm_melt, gpm_metal

def get_bounds(D):
    """Creates 2d array containing upper and lower bounds for each element in optimisation function based
       on model input. Conditions are imposed on species mole fractions (elements 0-25), moles of phases 
       (elements 26, 27, 28) and the pressure (element 29). Boundary conditions are adapted from original
       version of code by Edward Young.
       Parameters:
       D (dict)                 : dictionary containing model input, used to scale boundary condition for
                                elements 27, 27 and 28.
       Returns:
       bounds (np.2darray)      : boundary conditions (low, high) for each of the 30 elements of the model's
                                optimisation function."""
    bounds = np.zeros((30,2))
    for i in range(0,26):           # Set boundary conditions over first 25 elements through loop, we set 
        if i == 7:                  # specific boundaries for several reactions.
            bounds[i, 0] = 1.0e-15 
            bounds[i, 1] = 0.4
        elif i == 8:
            bounds[i, 0] = 1.0e-12
            bounds[i, 1] = 0.4
        elif i == 21 or i == 22 or i == 23:
            bounds[i, 0] = 1.0e-15
            bounds[i, 1] = 0.99999
        else:
            bounds[i, 0] = 1.0e-20
            bounds[i, 1] = 0.99999

    ## Boundaries on total # moles per phase
    bounds[26, 0] = D["moles_atm"]*1.0e-20 
    bounds[26, 1] = D["moles_atm"]*50.0
    bounds[27, 0] = D["moles_melt"]*0.5
    bounds[27, 1] = D["moles_melt"]*2.0
    bounds[28, 0] = D["moles_metal"]*0.5
    bounds[28, 1] = D["moles_metal"]*2.0

    ## Boundaries on pressure
    bounds[29, 0] = 1.0e-3
    bounds[29, 1] = 9.0e5

    return bounds

def sigmoidal_penalty(f, val, sharp, tol, mag):
    """Assigns a penalty function to provided function f according to a 2-sided logistic function.
       Penalty is applied to the region (-tol : 0.0 : tol).
       Parameters:
       f ()                     : 
       val (float)              : x value of f midpoint.
       sharp (float)            : sharpness of logistic function.
       tol (float)              : tolerance or width of area on which the penalty applies
       mag (float)              : scale factor for magnitude of the penalty
       
       Returns:
       f_penalty ()             : calculated penalty applied to function f """
    f_penalty = 1.0 - 1.0/(1.0 + np.exp(-sharp*((val+tol)-f))) + 1.0/(1.0 + np.exp(-sharp*((val-tol)-f)))
    return f_penalty*mag


    