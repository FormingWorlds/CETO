from __future__ import annotations
import numpy as np
from numpy import log as ln

def get_activity(D):
    """Computes activity coefficient for model species. For ideal mixing, all ln(g) terms equal
       zero such that there is no deviation from ideal behaviour. Non-ideal mixing is supported
       for Si, O and H2O at present.
       Parameters:
       D (dict)                 : Dictionary representing input for model; mole fractions of species
                                  are used in calculation of activity coefficients. Boolean element
                                  "bool_nonideal_mixing" controls non-ideality of activity coefficients.
                                  If True, lng terms will be computed via empirical formula. If False,
                                  lng terms will be set to zero instead indicating ideal mixing.
       
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
    else:
        lngSi = 0.0
        lngO = 0.0
        xB = 0.0

    lngH2 = 0.0
    lngH2O_melt = 0.0
    lng_H_metal = 0.0


    return [lngSi, lngO, lngH2, lngH2O_melt, lng_H_metal, xB]