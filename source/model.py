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


def optimisationfunction_initial(D, Pstd=1.0, T_low=1300, T_num=200):
    """Function establishes and computes initial values of 29 equations in 29 variables, taken from input file D. The first 19 reactions are equilibrium reactions for the 19 basis reactions of the model.
       Reactions 20 - 26 constrain the mass balance of the elements (H, C, O, Si, Mg, Fe). Reactions 27, 28 and 29 set summing constraints on the mole fractions of melt, atmosphere and metal phases.
       Function returns values f1 - f29 of each equation.
       Parameters:
        D (dict)                : input dictionary containing model input. Function uses mole fractions of planet constituents, moles of atmosphere, melt and metal, surface temperature
                                and core-mantle equilibration temperature.
        Pstd (float, optional)  : Deviation for pressure terms of evaporation reactions. Default is 1.0, indicating no deviation.
        T_low (float, optional) : Lower bound for temperature array over which thermodynamics are calculated, default is 1300 K
        T_num (float, optional) : Number of points in temperature array over which Gibbs free energies are calculated, default is 200.
       Returns:
        F (list)                : list containing initial values of each equation (length 29), in order."""

    ## Get thermodynamics
    T_array  = np.linspace(T_low, D["T_eq"], T_num)
    GRT_list = calculate_GRT(T_array)
    GRT_vals = []; GRT_keys = []
    for i in range(len(GRT_list)):
        if i == 1 or i == 3 or i == 4 or i == 6:
            GRT_T = GRT_list[i][np.argmin(np.abs(T_array - D["T_eq"]))] # For reactions 2, 4, 5, 7 use Gibbs free energy at higher temperature (T_eq) for optimisationfunction
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
    
    ## Mass balance for elements H, C, O, Na, Mg, Si, Fe
    f20 = 0.0
    f21 = 0.0
    f22 = 0.0
    f23 = 0.0
    f24 = 0.0
    f25 = 0.0
    f26 = 0.0

    ## Summing constraint on mole fractions for gas, melt and metal phases
    f27 = 1 - np.sum([D[i] for i in D if "_melt" in i and "moles" not in i and "wt" not in i])
    f28 = 1 - np.sum([D[i] for i in D if "_metal" in i and "moles" not in i and "wt" not in i and "bool" not in i])
    f29 = 1 - np.sum([D[i] for i in D if "_gas" in i and "moles" not in i and "wt" not in i])

    F_initial = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, 
             f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29])
    
    return F_initial

