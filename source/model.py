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


def optimisationfunction_initial(D, initial_moles, Pstd=1.0, T_low=1300.0, T_num=200):
    """Function establishes and computes initial values of 29 equations in 29 variables, taken from input file D. The first 19 reactions are equilibrium reactions for the 19 basis reactions of the model.
       Reactions 20 - 26 constrain the mass balance of the elements (H, C, O, Si, Mg, Fe). Reactions 27, 28 and 29 set summing constraints on the mole fractions of melt, atmosphere and metal phases.
       Function returns values f1 - f29 of each equation.
       Parameters:
        D (dict)                : input dictionary containing model input. Function uses mole fractions of planet constituents, moles of atmosphere, melt and metal, surface temperature
                                and core-mantle equilibration temperature.
        initial_moles (dict)    : dictionary containing initial amount of moles for Si, Mg, O, C, H, Fe, Na in the system
        Pstd (float, optional)  : Pressure at standard state, chosen to be 1.0 bar by default (As described in Schlichting&Young (2022))
        T_low (float, optional) : Lower bound for temperature array over which thermodynamics are calculated, default is 1300 K
        T_num (int, optional)   : Number of points in temperature array over which Gibbs free energies are calculated, default is 200.
       Returns:
        F (list)                : list containing initial values of each equation (length 29), in order."""

    ## Get thermodynamics
    T_array  = np.linspace(T_low, D["T_eq"], T_num)
    GRT_list = calculate_GRT(T_array)
    GRT_vals = []; GRT_keys = []
    for i in range(len(GRT_list)):
        if i == 1 or i == 3 or i == 4 or i == 6:
            GRT_T = GRT_list[i][np.argmin(np.abs(T_array - D["T_eq"]))] # For reactions 2, 4, 5, 7 (partitioning reactions between melt/metal) use T_eq
        else:
            GRT_T = GRT_list[i][np.argmin(np.abs(T_array - D["T_surface"]))] # Otherwise just find Gibbs free energy at closest T to surface temperature
        GRT_vals.append(GRT_T)
        GRT_keys.append(f"GRT{i+1}_T")
    G = dict(zip(GRT_keys, GRT_vals))

    ## Surface pressure
    P = calculate_pressure(D)
    
    ## Get activities for Si, O, H2, H2O in melt phase, H in metal phase.
    lngSi, lngO, lngH2, lngH2O_melt, lngH_metal, xB = get_activity(D)

    ## Equilibrium equations for reactions R1 - R19
    f1 = ln(D["Na2O_melt"]) + ln(D["SiO2_melt"]) - ln(D["Na2SiO3_melt"]) + G["GRT1_T"]
    f2 = 0.5*ln(D["Si_metal"]) + 0.5*lngSi + ln(D["FeO_melt"]) - 0.5*ln(D["SiO2_melt"]) - \
        ln(D["Fe_metal"]) + G["GRT2_T"]
    f3 = ln(D["MgO_melt"]) + ln(D["SiO2_melt"]) - ln(D["MgSiO3_melt"]) + G["GRT3_T"]
    f4 = 0.5*ln(D["SiO2_melt"]) - ln(D["O_metal"]) - lngO -0.5*ln(D["Si_metal"]) -0.5*lngSi + G["GRT4_T"]
    f5 = ln(D["H2_melt"]) + lngH2 - 2*ln(D["H_metal"]) - 2*lngH_metal + G["GRT5_T"]
    f6 = ln(D["FeO_melt"]) + ln(D["SiO2_melt"]) - ln(D["FeSiO3_melt"]) + G["GRT6_T"]

    if xB != 0.0:
        f7 = ln(D["SiO2_melt"]) + 2*ln(D["H2_melt"]) + 2*lngH2 -4*ln(xB) - 2*lngH2O_melt - \
        ln(D["Si_metal"]) - lngSi + G["GRT7_T"]
    else:
        f7 = ln(D["SiO2_melt"]) + 2*ln(D["H2_melt"]) + 2*lngH2 - 2*ln(D["H2O_melt"]) - 2*lngH2O_melt - \
        ln(D["Si_metal"]) - lngSi + G["GRT7_T"]

    f8 = ln(D["CO2_gas"]) - ln(D["CO_gas"]) - 0.5*ln(D["O2_gas"]) + G["GRT8_T"] - 0.5*ln(P/Pstd)
    f9 = 2*ln(D["H2_gas"]) + ln(D["CO_gas"]) - ln(D["CH4_gas"]) - 0.5*ln(D["O2_gas"]) + G["GRT9_T"] + 1.5*ln(P/Pstd)
    f10 = ln(D["H2O_gas"]) - 0.5*ln(D["O2_gas"]) - ln(D["H2_gas"]) + G["GRT10_T"] - 0.5*ln(P/Pstd)
    f11 = 0.5*ln(D["O2_gas"]) + ln(D["Fe_gas"]) - ln(D["FeO_melt"]) + G["GRT11_T"] + 1.5*ln(P/Pstd)
    f12 = 0.5*ln(D["O2_gas"]) + ln(D["Mg_gas"]) - ln(D["MgO_melt"]) + G["GRT12_T"] + 1.5*ln(P/Pstd)
    f13 = 0.5*ln(D["O2_gas"]) + ln(D["SiO_gas"]) - ln(D["SiO2_melt"]) + G["GRT13_T"] + 1.5*ln(P/Pstd)
    f14 = 0.5*ln(D["O2_gas"]) + 2*ln(D["Na_gas"]) - ln(D["Na2O_melt"]) + G["GRT14_T"] + 2.5*ln(P/Pstd)
    f15 = ln(D["H2_melt"]) + lngH2 - ln(D["H2_gas"]) + G["GRT15_T"] - ln(1.0e4/Pstd) #Fixed at 3GPa, Young line 1612

    if xB != 0.0:
        f16 = 2*ln(xB) - ln(D["H2O_gas"]) + G["GRT16_T"] - ln(P/Pstd)
    else:
        f16 = ln(D["H2O_melt"]) + lngH2O_melt - ln(D["H2O_gas"]) + G["GRT16_T"] - ln(P/Pstd)
    
    f17 = ln(D["CO_melt"]) - ln(D["CO_gas"]) + G["GRT17_T"] - ln(P/Pstd)
    f18 = ln(D["CO2_melt"]) - ln(D["CO2_gas"]) + G["GRT18_T"] - ln(P/Pstd)
    f19 = ln(D["SiH4_gas"]) + 0.5*ln(D["O2_gas"]) - ln(D["SiO_gas"]) - 2*ln(D["H2_gas"]) + \
        G["GRT19_T"] - 1.5*ln(P/Pstd)
    
    ## Mass balance for elements H, C, O, Na, Mg, Si, Fe, must be zero initially
    f20 = initial_moles["Si"] - moles_in_system('Si', D)
    f21 = initial_moles["Mg"] - moles_in_system('Mg', D)
    f22 = initial_moles["O"] - moles_in_system('O', D)
    f23 = initial_moles["Fe"] - moles_in_system('Fe', D)
    f24 = initial_moles["H"] - moles_in_system('H', D)
    f25 = initial_moles["Na"] - moles_in_system('Na', D)
    f26 = initial_moles["C"] - moles_in_system('C', D)

    ## Summing constraint on mole fractions for gas, melt and metal phases
    f27 = 1 - np.sum([D[key] for key in D if "_melt" in key and "moles" not in key and "wt" not in key])
    f28 = 1 - np.sum([D[key] for key in D if "_metal" in key and "moles" not in key and "wt" not in key and "bool" not in key])
    f29 = 1 - np.sum([D[key] for key in D if "_gas" in key and "moles" not in key and "wt" not in key])

    F_initial = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, 
             f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29])
    
    return F_initial

def optimisationfunction(D, initial_moles, w_gas, Pstd=1.0, T_low=1300.0, T_num=200):
    wt_massbalance = w_gas*D["wt_massbalance"]
    wt_summing = w_gas*D["wt_summing"]
    wt_atm = w_gas*D["wt_atm"]
    wt_solub = w_gas*D["wt_solub"]
    wt_melt = w_gas*D["wt_melt"]
    wt_evap = w_gas*D["wt_evap"]

    P_guess = calculate_pressure(D)
    P = (P_guess - D["P_penalty"]) / P_guess

    ## Create list of initial number of moles for each element to ensure proper indexing when assigning weights later
    nElements = [initial_moles["Si"], initial_moles["Mg"], initial_moles["O"], initial_moles["Fe"], initial_moles["H"],
                 initial_moles["Na"], initial_moles["C"]]
    wtm = 5.0                                                       # Extra weight factor for mass balance equations

    F_ini = optimisationfunction_initial(D, initial_moles, Pstd=Pstd, T_low=T_low, T_num=T_num)
    F_list = []                                                     # List to contain values for the 29 equations


    ## Assign weights to all equations
    for i in range(1,(len(F_ini)+1)): # Runs over a range of i in (1 - 29)
        f_i_ini = F_ini[(i-1)]
        if i in (1, 2, 3, 4, 5, 6, 7):                              # F1 - F7 are in-melt reactions
            f_i = wt_melt*D[f"wt_f{i}"]*f_i_ini
        elif i in (8, 9, 10, 19):                                   # F8, F9, F10 and F19 are atmospheric reactions
            f_i = wt_atm*D[f"wt_f{i}"]*f_i_ini 
        elif i in (11, 12, 13, 14):                                 # F11 - F14 are reactions for evaporation of melt
            f_i = wt_evap*D[f"wt_f{i}"]*f_i_ini
        elif i in (15, 16, 17, 18):                                 # F15 - F18 are solubility reactions
            f_i = wt_solub*D[f"wt_f{i}"]*f_i_ini
        elif i in (20, 21, 22, 23, 24, 25, 26):                     # Mass Balance Equations
            f_i = wt_massbalance*D[f"wt_f{i}"]*(wtm/nElements[int(i-20)]) * f_i_ini
        elif i in (27, 28, 29):                                     # Summing constraints on phases
            f_i = wt_summing*D[f"wt_f{i}"] * f_i_ini
        F_list.append(f_i)

    F_unpenalized = np.array(F_list) #convert to array before applying penalties

    ## Assign penalties
    F = np.zeros(30)
    F[0:19] = sigmoidal_penalty(F_unpenalized[0:19], 0, 5, 1, 10000) #value=0, sharpness=5, tolerance=1, magnitude=10,000
    F[19:26] = sigmoidal_penalty(F_unpenalized[19:26], 0, 1, 0.01, 1000) #value=0, sharpness=1, tolerance=0.01, magnitude=1,000
    F[26:29] = sigmoidal_penalty(F_unpenalized[26:29], 0, 1, 0.005, 100000) #value=0, sharpness=1, tolerance=0.005, magnitude=100,000
    F[29] = sigmoidal_penalty(P, 0, 1, 0.2, D["P_penalty"]) #value=0, sharpness=1, tolerance=0.2, take magnitude from input

    ## Sum of squared errors in F
    sum = np.sum((F**2))
    return sum