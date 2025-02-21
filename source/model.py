from __future__ import annotations
import numpy as np
from numpy import log as ln

try:
    from thermodynamics import calculate_GRT
    from utilities import *
    from activity import get_activity
except:
    from source.thermodynamics import calculate_GRT
    from source.utilities import *
    from source.activity import get_activity


def objectivefunction_initial(var, varkeys, config, initial_moles, G, Pstd=1.0):
    """Function establishes and computes initial values of 29 equations in 29 variables, taken from input file D. The first 19 reactions are equilibrium reactions for the 19 basis reactions of the model.
       Reactions 20 - 26 constrain the mass balance of the elements (H, C, O, Si, Mg, Fe). Reactions 27, 28 and 29 set summing constraints on the mole fractions of melt, atmosphere and metal phases.
       Function returns values f1 - f29 of each equation.
       Parameters:
        var (array)             : 1D array of model variables. Includes mole fractions of species in gas, melt and metal phases, total moles in gas, melt and metal phases and pressure.
        varkeys (array or list) : 1D array of keys (strings) from which a dictionary D will be formed from values in var.
        config (dict)           : dictionary containing global model input. Used to retrieve global parameters that do not vary through the model run.
        initial_moles (dict)    : dictionary containing initial amount of moles for Si, Mg, O, C, H, Fe, Na in the system.
        G (dict)                : dictionary containing G / RT terms for reactions 1-19 at relevant temperatures.
        Pstd (float, optional)  : Pressure at standard state, chosen to be 1.0 bar by default (As described in Schlichting&Young (2022))
       Returns:
        F (list)                : list containing initial values of each equation (length 29), in order."""

    ## Create dictionary of input variables for legibility
    D = dict(zip(varkeys, var))

    ## Surface pressure
    P = D["P_penalty"]
    #P = calculate_pressure(D, config)
    
    ## Get activities for Si, O, H2, H2O in melt phase, H in metal phase.
    lngSi, lngO, lngH2, lngH2O_melt, lngH_metal, xB = get_activity(D, config)

    ## Equilibrium equations for reactions R1 - R19
    f1 = ln(D["Na2O_melt"]) + ln(D["SiO2_melt"]) - ln(D["Na2SiO3_melt"]) + G["R1"]
    f2 = 0.5*ln(D["Si_metal"]) + 0.5*lngSi + ln(D["FeO_melt"]) - 0.5*ln(D["SiO2_melt"]) - \
        ln(D["Fe_metal"]) + G["R2"]
    f3 = ln(D["MgO_melt"]) + ln(D["SiO2_melt"]) - ln(D["MgSiO3_melt"]) + G["R3"]
    f4 = 0.5*ln(D["SiO2_melt"]) - ln(D["O_metal"]) - lngO -0.5*ln(D["Si_metal"]) -0.5*lngSi + G["R4"]
    f5 = ln(D["H2_melt"]) + lngH2 - 2*ln(D["H_metal"]) - 2*lngH_metal + G["R5"]
    f6 = ln(D["FeO_melt"]) + ln(D["SiO2_melt"]) - ln(D["FeSiO3_melt"]) + G["R6"]

    if xB != 0.0:
        f7 = ln(D["SiO2_melt"]) + 2*ln(D["H2_melt"]) + 2*lngH2 -4*ln(xB) - 2*lngH2O_melt - \
        ln(D["Si_metal"]) - lngSi + G["R7"]
    else:
        f7 = ln(D["SiO2_melt"]) + 2*ln(D["H2_melt"]) + 2*lngH2 - 2*ln(D["H2O_melt"]) - 2*lngH2O_melt - \
        ln(D["Si_metal"]) - lngSi + G["R7"]

    f8 = ln(D["CO2_gas"]) - ln(D["CO_gas"]) - 0.5*ln(D["O2_gas"]) + G["R8"] - 0.5*ln(P/Pstd)
    f9 = 2*ln(D["H2_gas"]) + ln(D["CO_gas"]) - ln(D["CH4_gas"]) - 0.5*ln(D["O2_gas"]) + G["R9"] + 1.5*ln(P/Pstd)
    f10 = ln(D["H2O_gas"]) - 0.5*ln(D["O2_gas"]) - ln(D["H2_gas"]) + G["R10"] - 0.5*ln(P/Pstd)
    f11 = 0.5*ln(D["O2_gas"]) + ln(D["Fe_gas"]) - ln(D["FeO_melt"]) + G["R11"] + 1.5*ln(P/Pstd)
    f12 = 0.5*ln(D["O2_gas"]) + ln(D["Mg_gas"]) - ln(D["MgO_melt"]) + G["R12"] + 1.5*ln(P/Pstd)
    f13 = 0.5*ln(D["O2_gas"]) + ln(D["SiO_gas"]) - ln(D["SiO2_melt"]) + G["R13"] + 1.5*ln(P/Pstd)
    f14 = 0.5*ln(D["O2_gas"]) + 2*ln(D["Na_gas"]) - ln(D["Na2O_melt"]) + G["R14"] + 2.5*ln(P/Pstd)
    f15 = ln(D["H2_melt"]) + lngH2 - ln(D["H2_gas"]) + G["R15"] - ln(1.0e4/Pstd) #Fixed at 3GPa, Young line 1612

    if xB != 0.0:
        f16 = 2*ln(xB) - ln(D["H2O_gas"]) + G["R16"] - ln(P/Pstd)
    else:
        f16 = ln(D["H2O_melt"]) + lngH2O_melt - ln(D["H2O_gas"]) + G["R16"] - ln(P/Pstd)
    
    f17 = ln(D["CO_melt"]) - ln(D["CO_gas"]) + G["R17"] - ln(P/Pstd)
    f18 = ln(D["CO2_melt"]) - ln(D["CO2_gas"]) + G["R18"] - ln(P/Pstd)
    f19 = ln(D["SiH4_gas"]) + 0.5*ln(D["O2_gas"]) - ln(D["SiO_gas"]) - 2*ln(D["H2_gas"]) + \
        G["R19"] - 1.5*ln(P/Pstd)
    
    ## Mass balance for elements Si, Mg, O, Fe, H, Na, C, must be zero initially
    f20 = initial_moles["Si"] - moles_in_system('Si', D)
    f21 = initial_moles["Mg"] - moles_in_system('Mg', D)
    f22 = initial_moles["O"] - moles_in_system('O', D)
    f23 = initial_moles["Fe"] - moles_in_system('Fe', D)
    f24 = initial_moles["H"] - moles_in_system('H', D)
    f25 = initial_moles["Na"] - moles_in_system('Na', D)
    f26 = initial_moles["C"] - moles_in_system('C', D)

    ## Summing constraint on mole fractions for gas, melt and metal phases
    ## Doing it now 'by hand' to force no rounding errors compared to Young
    f27 = 1.0 - D["CH4_gas"] - D["H2_gas"] - D["CO2_gas"] - D["CO_gas"] - D["Fe_gas"] - D["H2O_gas"] - D["SiH4_gas"] - D["SiO_gas"] - D["Mg_gas"] - D["Na_gas"] - D["O2_gas"]
    f28 = 1.0 - D["MgO_melt"] - D["MgSiO3_melt"] - D["SiO2_melt"] - D["FeO_melt"] - D["FeSiO3_melt"] - D["Na2O_melt"] - D["Na2SiO3_melt"] - D["H2O_melt"] - D["CO2_melt"] - D["CO_melt"] - D["H2_melt"]
    f29 = 1.0 - D["H_metal"] - D["O_metal"] - D["Fe_metal"] - D["Si_metal"]

    #f27 = 1.0 - np.sum([D[key] for key in D if "_gas" in key and "moles" not in key and "wt" not in key])
    #f28 = 1.0 - np.sum([D[key] for key in D if "_melt" in key and "moles" not in key and "wt" not in key])
    #f29 = 1.0 - np.sum([D[key] for key in D if "_metal" in key and "moles" not in key and "wt" not in key and "bool" not in key])

    F_initial = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, 
             f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29])
    
    return F_initial

def objectivefunction(var, varkeys, config, initial_moles, G, w_gas, Pstd=1.0):
    """Objective function to be minimised by dual_annealing / MCMC search containing the 29 nonlinear equations in 29 variables
       describing the system. Function computes values for the 29 equations using objectivefunction_initial and assigns weights and
       penalties for each function based on the model config and sigmoidal penalty functions. Function finally computes the sum of 
       squared values and returns it.
       Parameters:
        var (1darray)                : 1D array of model variables. Includes mole fractions of species, total # moles in each phase and pressure.
        varkeys (list)               : list of keys (strings) from which dictionary D will be formed using values in var.
        config (dict)                : Dictionary containing global model input. Used to retrieve global parameters that do not vary in model run.
        initial_moles (dict)         : Dictionary containing initial amount of moles of Si, Mg, Fe, Na, H, C, O in system.
        G (dict)                     : Dictionary containing G / RT terms for reactions 1-19 at relevant temperatures.
        w_gas (float)                : Scale factor for weights on model equations.
        Pstd (float, optional)       : Pressure at standard state, chosen to be 1.0 by default (as in Schlichting & Young (2022))
       Returns: 
        sum (float)                  : Sum of squared errors"""
    D = dict(zip(varkeys, var))

    wt_massbalance = w_gas*config["wt_massbalance"]
    wt_summing = w_gas*config["wt_summing"]
    wt_atm = w_gas*config["wt_atm"]
    wt_solub = w_gas*config["wt_solub"]
    wt_melt = w_gas*config["wt_melt"]
    wt_evap = w_gas*config["wt_evap"]

    ## Create list of initial number of moles for each element to ensure proper indexing when assigning weights later
    nElements = list(initial_moles.values())
    wtm = 5.0                                                       # Extra weight factor for mass balance equations

    F_ini = objectivefunction_initial(var, varkeys, config, initial_moles, G, Pstd=Pstd)
    F_list = []                                                     # List to contain values for the 29 equations

    ## Assign weights to all equations
    for i in range(1,(len(F_ini)+1)): # Runs over a range of i in (1 - 29)
        f_i_ini = F_ini[(i-1)]
        if i in (1, 2, 3, 4, 5, 6, 7):                              # F1 - F7 are in-melt reactions
            f_i = wt_melt*config[f"wt_f{i}"]*f_i_ini
        elif i in (8, 9, 10, 19):                                   # F8, F9, F10 and F19 are atmospheric reactions
            f_i = wt_atm*config[f"wt_f{i}"]*f_i_ini 
        elif i in (11, 12, 13, 14):                                 # F11 - F14 are reactions for evaporation of melt
            f_i = wt_evap*config[f"wt_f{i}"]*f_i_ini
        elif i in (15, 16, 17, 18):                                 # F15 - F18 are solubility reactions
            f_i = wt_solub*config[f"wt_f{i}"]*f_i_ini
        elif i in (20, 21, 22, 23, 24, 25, 26):                     # Mass Balance Equations Si, Mg, O, Fe, H, Na, C
            f_i = wt_massbalance*config[f"wt_f{i}"]*(wtm/nElements[int(i-20)]) * f_i_ini
        elif i in (27, 28, 29):                                     # Summing constraints on phases
            f_i = wt_summing*config[f"wt_f{i}"] * f_i_ini
        F_list.append(f_i)

    P_guess = calculate_pressure(D, config)
    P = (P_guess - D["P_penalty"]) / P_guess
    F_list.append(P)

    F_unpenalized = np.array(F_list) #convert to array before applying penalties

    ## Assign penalties
    F = np.zeros(30)
    F[0:19] = sigmoidal_penalty(F_unpenalized[0:19], 0, 5, 1, 10000) #value=0, sharpness=5, tolerance=1, magnitude=10,000
    F[19:26] = sigmoidal_penalty(F_unpenalized[19:26], 0, 1, 0.01, 1000) #value=0, sharpness=1, tolerance=0.01, magnitude=1,000
    F[26:29] = sigmoidal_penalty(F_unpenalized[26:29], 0, 1, 0.005, 100000) #value=0, sharpness=1, tolerance=0.005, magnitude=100,000
    F[29] = sigmoidal_penalty(F_unpenalized[29], 0, 1, 0.2, config["P_penalty"]) #value=0, sharpness=1, tolerance=0.2, take magnitude from input

    ## Sum of squared errors in F
    sum = np.sum((F**2))
    return sum

def model(theta, thetakeys, config, initial_moles, G, Pstd=1.0):
    """Function sets up vector y_model of variable vector theta for later use in MCMC solver. y_model values are linked to model system of equations,
       without pressure corrections or thermodynamics.
       Parameters:
       theta (1darray)          : 1darray of model variables
       thetakeys (list)         : list of keys (strings) from which dictionary T will be formed using values contained in theta.
       config (dict)            : Dictionary containing global model input.
       inital_moles (dict)      : Dictionary containing initial amount of moles of Si, Mg, O, Fe, H, Na, C in system.
       G (dict)                 : Dictionary containing G / RT terms for reactions 1-19 at relevant temperatures
       Pstd (float, optional)   : Pressure at standard state, chosen to be 1.0 by default in keeping with Schlichting & Young (2022)
       
       Returns:
       y_model (1darray)        : array of model values in order (len 30) """
    T = dict(zip(thetakeys, theta))

    F = objectivefunction_initial(theta, thetakeys, config, initial_moles, G, Pstd=Pstd)
    y_model = np.zeros(len(theta))
    for i in range(19):
        if i == 14:
            y_model[i] = F[i] - list(G.values())[i] + ln(1e4 / Pstd) #Model term without thermodynamics or pressure correction
        else:
            y_model[i] = F[i] - list(G.values())[i]                  #Model term without thermodynamics

    for j in range(19, 26):
        y_model[j] = -F[j] + list(initial_moles.values())[(j-19)]

    for k in range(26, 29):
        y_model[k] = -F[k] + 1.0

    P_guess = (calculate_pressure(T, config))
    y_model[-1] = 100*(P_guess - theta[-1])/P_guess

    return y_model
