from __future__ import annotations

from .constants import (log_to_ln, R_gas)

import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp as exp

def get_Gibbs_melt(T):
    ti = T/1000

    ## MgO (NIST data)
    a, b, c, d, e, f, g = 66.944, 0, 0, 0, 0, -580.9944, 93.74712
    H_MgO_melt = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_MgO_melt = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_MgO_melt = H_MgO_melt - T*S_MgO_melt

    ## MgSiO3 (NIST data)
    a, b, c, d, e, f, g = 146.440, -1.499926e-7, 6.220145e-8, -8.733222e-9, -3.144171e-8, -1563.306, 220.6679
    H_MgSiO3_melt = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_MgSiO3_melt = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_MgSiO3_melt = H_MgSiO3_melt - T*S_MgSiO3_melt

    ## SiO2 (NIST data)
    a, b, c, d, e, f, g = 85.772, -1.6e-5, 4e-6, -3.809081e-7, -1.7e-5, -952.87, 113.344
    H_SiO2_melt = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_SiO2_melt = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_SiO2_melt = H_SiO2_melt - T*S_SiO2_melt

    ## FeO (NIST data)
    a, b, c, d, e, f, g = 68.19920, -4.501232e-10, 1.195227e-10, -1.064302e-11, -3.09268e-10, -281.4326, 137.8377
    H_FeO_melt = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_FeO_melt = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_FeO_melt = H_FeO_melt - T*S_FeO_melt

    ## Fe metal (NIST data)
    a, b, c, d, e, f, g = 46.024, -1.884667e-8, 6.09475e-9, -6.640301e-10, -8.246121e-9, -10.80543, 72.54094
    H_Fe_metal = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_Fe_metal = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_Fe_metal = H_Fe_metal - T*S_Fe_metal

    ## H2 gas
    a, b, c, d, e, f, g = 18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438, 156.288133
    H_H2_gas = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_H2_gas = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_H2_gas = H_Fe_metal - T*S_Fe_metal

    ## H2O gas
    G_H2O_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1700:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 30.092, 6.832514, 6.793435, -2.53448, 0.082139, -250.881, 223.3967
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_H2O_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 41.96426, 8.622053, -1.49978, 0.098119, -11.15764, -272.1797, 219.7809
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_H2O_gas[i] = H-T[i]*s
        i += 1

