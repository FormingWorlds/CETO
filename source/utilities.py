from __future__ import annotations
import numpy as np
from numpy import log as ln
from numpy import log10 as log

from pathlib import Path


try:
    from constants import *
except:
    from source.constants import *

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
            "offset_MCMC", "bool_unreactive_metal", "bool_ideal_mixing"]
    
    config = dict(zip(datakeys, data))
    return config

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
            el_inmetal += D[key]

        elif el in key and '_gas' in key:
            if (el+'2') not in key and (el+'3') not in key and (el+'4') not in key:
                el_ingas += D[key]
            else:
                for coeff in range(2,5):
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

def calculate_pressure(D, config):
    """Calculates pressure according to equation 8 from Schlichting & Young (2022).
    Parameters:
    D (dict)                : Dictionary of model variables, retrieves moles in
                            each phase to calculate surface pressure in bar
    config (dict)           : Dictionary of global model input, unchanged from 
                            model initialisation, retrieves planet mass
    Returns:
    P (float)               : Surface pressure in bar"""
    
    gpm_gas, gpm_melt, gpm_metal = gpm_phases(D)
    moles_total = D["moles_atm"] + D["moles_melt"] + D["moles_metal"]

    molefrac_atm = D["moles_atm"] / moles_total
    molefrac_melt = D["moles_melt"] / moles_total
    molefrac_metal = 1.0 - molefrac_atm - molefrac_melt

    grams_atm = gpm_gas*molefrac_atm
    grams_melt = gpm_melt*molefrac_melt
    grams_metal = gpm_metal*molefrac_metal


    totalmass = grams_atm + grams_melt + grams_metal

    massfrac_atm = grams_atm / totalmass

    fratio = massfrac_atm/(1.0-massfrac_atm)
    P = 1.2e6*fratio*(config["M_p"])**(2/3) # surface pressure in bar
    return P


def get_bounds(config):
    """Creates 2d array containing upper and lower bounds for each element in optimisation function based
       on model input. Conditions are imposed on species mole fractions (elements 0-25), moles of phases 
       (elements 26, 27, 28) and the pressure (element 29). Boundary conditions are adapted from original
       version of code by Edward Young.
       Parameters:
       config (dict)            : dictionary containing global model input, used to scale boundary condition for
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
            bounds[i, 1] = 1.0
        else:
            bounds[i, 0] = 1.0e-20
            bounds[i, 1] = 0.99999

    ## Boundaries on total # moles per phase
    bounds[26, 0] = config["moles_atm"]*1.0e-20 
    bounds[26, 1] = config["moles_atm"]*50.0
    bounds[27, 0] = config["moles_melt"]*0.5
    bounds[27, 1] = config["moles_melt"]*2.0
    bounds[28, 0] = config["moles_metal"]*0.5
    bounds[28, 1] = config["moles_metal"]*2.0

    ## Boundaries on pressure
    bounds[29, 0] = 1.0e-3
    bounds[29, 1] = 900000

    return bounds

def sigmoidal_penalty(f, val, sharp, tol, mag):
    """Assigns a penalty function to provided function f according to a 2-sided logistic function.
       Penalty is applied to the region (-tol : 0.0 : tol).
       Parameters:
       f (float or array)           : float value of function or array of function values.
       val (float)                  : x value of f midpoint.
       sharp (float)                : sharpness of logistic function.
       tol (float)                  : tolerance or width of area on which the penalty applies.
       mag (float)                  : scale factor for magnitude of the penalty.
       
       Returns:
       f_penalty (float or array)   : calculated penalty applied to function f."""
    f_penalty = 1.0 - 1.0/(1.0 + np.exp(-sharp*((val+tol)-f))) + 1.0/(1.0 + np.exp(-sharp*((val-tol)-f)))
    return f*(1 + f_penalty*mag)

def smoothTriangle(data, degree):
    """Function used in Young code to estimate cost (or mean value) of the objective function.
       Its precise workings and intentions are still unclear; this docstring will be more descriptive
       in the future.
       Parameters:
       data (list or array)     : list or array of values
       degree (int)             : integer that controls dimensions of triangle to be drawn from

       Returns:
       smoothed (array)         : array of 'smoothed' values."""
    triangle = np.concatenate((np.arange(degree+1), np.arange(degree)[::-1]))
    smoothed = []
    for i in range(degree, len(data)-degree*2):
        point = data[i:i+len(triangle)]*triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle Boundaries
    smoothed = [smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

def progress_callback(x, fun, context):
    a_file = open('progress.txt', 'a')
    a_file.write("%10.5e \n" % fun)
    a_file.close()

def gelman_rubin(samples):
    """Compute the Gelman-Rubin statistic for all variables in an emcee MCMC sample.
       Function obtains chain length, number of walkers and number of variable from the sample.
       For each variable, the Gelman-Rubin statistic is then computed by finding the variance
       within each chain and the variance across all chains.
       Parameters:
       samples (numpy.ndarray)        : array representing samples from an MCMC run. Expected dimensions are
                                       [[number of iterations],[number of walkers],[number of variables]].
       Returns:
       statistics (numpy.1darray)     : array with the computed Gelman-Rubin statistic for each variable."""
    S = np.shape(samples)
    chain_length = S[0]    #length of chains
    num_walkers = S[1]     #number of walkers
    num_variables = S[2]   #number of variables

    statistics = np.zeros(num_variables)
    
    for i in range(num_variables):
        chain_mean = np.zeros(num_walkers)
        chain_variance = np.zeros(num_walkers)
        
        for j in range(num_walkers):
            walker = samples[:,j,i]       # iterate over each variable in each walker
            chain_mean[j] = np.mean(walker)
            chain_variance[j] = np.var(walker)
            
        grand_mean = np.mean(chain_mean)  # Mean across all walkers
        B = (chain_length / (num_walkers-1))*np.sum((chain_mean - grand_mean)**2) # Variance between chains
        W = np.mean(chain_variance) # mean variance within chains
        
        R = ( ((chain_length-1)/chain_length)*W + (1/chain_length)*B ) / W # will tend to 1 as B->0 and L-> inf
        statistics[i] = R
    return statistics






    