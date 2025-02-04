from __future__ import annotations
import numpy as np
from numpy import log as ln

from thermodynamics import calculate_GRT
from utilities import *
from activity import get_activity

def optimisationfunction(D, Pstd=1.0, T_low=1300, T_num=200):
    """Sets up the system of 30 nonlinear equations in 30 variables to be solved for.
    WIP"""

    ## Get thermodynamics
    T_array  = np.linspace(T_low, D["T_eq"], T_num)
    GRT_list = calculate_GRT(T_array)
    GRT_vals = []; GRT_keys = []
    for i in range(len(GRT_list)):
        GRT_T = GRT_list[i][np.argmin(np.abs(T_array - D["T_surface"]))] #Find Gibbs at closest T to surface T
        GRT_vals.append(GRT_T)
        GRT_keys.append(f"GRT{i+1}_T")
    G = dict(zip(GRT_keys, GRT_vals))

    ## Surface pressure
    P = calculate_pressure(D)
    
    ## Get activities
    lngSi, lngO, lngH2, lngH2O_melt, lngH_metal, xB = get_activity(D)

    ## Equilibrium equations for reactions R1 - R19
    f1 = ln(D["Na2O_melt"]) + ln(D["SiO2_melt"]) - ln(D["Na2SiO3_melt"]) + G["GRT1_T"]
    f2 = 0.5*ln(D["Si_metal"]) +0.5*lngSi + ln(D["FeO_melt"]) - 0.5*ln(D["SiO2_melt"]) - \
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

    F = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, 
             f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29])
    
    # wgas = 1.0 / np.max(np.abs(F))
    # wtn = D["wt_massbalance"]*wgas
    # wtx = D["wt_summing"]*wgas

    # weights = np.array([wgas, wtn, wtx])
    
    return F

