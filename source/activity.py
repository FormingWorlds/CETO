from __future__ import annotations
import numpy as np
from numpy import log as ln

def get_activity(D, get_all=False):
    """Computes activity coefficient for model species. For ideal mixing, all ln(g) terms equal
       zero such that there is no deviation from ideal behaviour. Non-ideal mixing is captured by the
       computed lng (ln gamma) terms below for Si, O, H2, H2O in melt phase, H in metal phase and xB.
       Optional boolean parameter get_all controls whether all activities are computed and returned,
       or if only the activities of Si, O and xB are returned.
       Parameters:
       D (dict)                 : Dictionary representing input for model; mole fractions of species
                                  are used in calculation of activity coefficients. Boolean element
                                  "bool_nonideal_mixing" controls non-ideality of activity coefficients.
                                  If True, lng terms will be computed via empirical formula. If False,
                                  lng terms will be set to zero instead indicating ideal mixing.
       get_all (Bool)           : Boolean controlling whether terms for H2, H2O_melt and H_metal are 
                                computed and returned. These equations are implemented but their 
                                validity is still under review. Default is False.
       
       Returns:
       lng (list)               : List of lng terms for Si, O, H2, H2O (melt), H (metal) and xB."""
    
    if D["bool_nonideal_mixing"] is True:
        lngSi = -6.65*1873.0/D["T_eq"]-(12.41*1873.0/D["T_eq"])*ln(1.0-D["Si_metal"]) - \
        ((-5.0*1873.0/D["T_eq"])*D["O_metal"]*(1.0+ln(1-D["O_metal"])/D["O_metal"]-1.0/(1.0-D["Si_metal"]))) + \
        (-5.0*1873.0/D["T_eq"])*D["O_metal"]**2.0*D["Si_metal"]*(1.0/(1.0-D["Si_metal"])+1.0/(1.0-D["O_metal"]) + \
                                                             D["Si_metal"]/(2.0*(1.0-D["Si_metal"])**2.0)-1.0)
        lngO = (4.29-16500.0/D["T_eq"])-(-1.0*1873.0/D["T_eq"])*ln(1.0-D["O_metal"]) - \
        ((-5.0*1873.0/D["T_eq"])*D["Si_metal"]*(1.0+ln(1-D["Si_metal"])/D["Si_metal"]-1.0/(1.0-D["O_metal"]))) + \
        (-5.0*1873.0/D["T_eq"])*D["Si_metal"]**2.0*D["O_metal"]*(1.0/(1.0-D["O_metal"])+1.0/(1.0-D["Si_metal"])+ \
                                                             D["O_metal"]/(2.0*(1.0-D["O_metal"])**2.0)-1.0)
        
        xB = (D["H2O_melt"]/(1-D["H2O_melt"])*(1/3)) / (1.0 + (D["H2O_melt"]/(1-D["H2O_melt"])*(1/3)))

        if get_all is True:
            lngH2 = (74829.6/(8.3144*D["T_eq"]))*(1.0 - D["H2_melt"])**2.0

            lngH2O_melt = (74826.0/(8.3144*D["T_eq"]))*(1.0 - D["H2O_melt"])**2.0

            lngH_metal_T_indep = -1.9*(1.0 - D["H_metal"])**2.0

            x_light = D["Si_metal"] + D["O_metal"]
            lngH_metal = -3.8*x_light*(1.0 + ln(1.0 - x_light)/x_light - 1.0/(1.0-D["H_metal"])) + 3.8*(x_light**2.0)*D["H_metal"]* \
                (1.0/(1.0 - D["H_metal"]) + 1.0/(1.0 - x_light) + D["H_metal"]/(2.0*(1.0-D["H_metal"])**2.0) - 1.0)
        else:
            lngH2 = 0.0
            lngH2O_melt = 0.0
            lngH_metal = 0.0

    else:
        lngSi = 0.0
        lngO = 0.0
        xB = 0.0
        lngH2 = 0.0
        lngH2O_melt = 0.0
        lngH_metal = 0.0

    return [lngSi, lngO, lngH2, lngH2O_melt, lngH_metal, xB]