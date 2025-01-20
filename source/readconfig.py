from __future__ import annotations
import numpy as np

def readconfig(path):
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
    
    res = dict(zip(datakeys, data))
    return res



