from __future__ import annotations
import numpy as np
from numpy import log as ln

from thermodynamics import calculate_GRT
from utilities import readconfig
from activity import get_activity

def optimisationfunction(D):
    """Sets up the system of 30 nonlinear equations in 30 variables to be solved for.
    WIP"""
    T_array = np.linspace(1300.0, D["T_eq"], 200)
    ti = np.argmin(np.abs(T_array - D["T_surface"]))
    GRT_list = calculate_GRT(T_array)
    GRT_vals = []
    GRT_keys = []
    for i in range(len(GRT_list)):
        GRT_T = GRT_list[i][ti]
        GRT_vals.append(GRT_T)
        GRT_keys.append(f"GRT{i+1}_T")
    G = dict(zip(GRT_keys, GRT_vals))
    
    lngSi, lngO, lngH2, lngH2O_melt, lngH_metal, xB = get_activity(D)

    f1 = ln(D["Na2O_melt"])+ln(D["SiO2_melt"])-ln(D["Na2SiO3_melt"])+G["GRT1_T"]
    f2 = 0.5*ln(D["Si_metal"])+0.5*lngSi+ln(D["FeO_melt"])-0.5*ln(D["SiO2_melt"])-ln(D["Fe_metal"])+G["GRT2_T"]
    f3 = ln(D["MgO_melt"])+ln(D["SiO2_melt"])-ln(D["MgSiO3_melt"])+G["GRT3_T"]
    return f1, f2, f3

