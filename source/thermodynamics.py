from __future__ import annotations

from constants import (log_to_ln, R_gas)

import numpy as np
from numpy import log as ln
from numpy import log10 as log
from numpy import exp as exp

def get_Gibbs(T):
    """Computes Gibbs free energies of formation of species at temperature T and 1 bar of pressure via
       Shomate polynomial equations. Coefficient data taken from NIST where available.
       Parameters:
            T (np.1darray)         : array of temperatures between minimum and maximum temperatures
            
       Returns:
            G_dict (dict)          : dictionary containing 1d arrays of Gibbs free energies for supported 
                                     species, indexed as G_{species}_{phase}; eg. G_H2O_gas."""
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

    ## O2 gas
    a, b, c, d, e, f, g = 30.03235, 8.772972, -3.988133, 0.788313, -0.741599, -11.32468, 236.1663
    H_O2_gas = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_O2_gas = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_O2_gas = H_O2_gas - T*S_O2_gas

    ## Na2O melt
    a, b, c, d, e, f, g = 104.600, 9.909135e-10, -6.022074e-10, 1.113058e-10, 2.362827e-11, -404.0296, 218.1902
    H_Na2O_melt = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_Na2O_melt = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_Na2O_melt = H_Na2O_melt - T*H_Na2O_melt

    ## Na gas
    a, b, c, d, e, f, g = 20.80573, 0.277206, -0.392086, 0.119634, -0.008879, 101.0386, 178.7095
    H_Na_gas = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_Na_gas = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_Na_gas = H_Na_gas - T*S_Na_gas

    ## Fe gas
    a, b, c, d, e, f, g = 11.29253, 6.989707, -1.110305, 0.122354, 5.689278, 423.5380, 206.3591
    H_Fe_gas = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
    S_Fe_gas = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
    G_Fe_gas = H_Fe_gas - T*S_Fe_gas

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
            G_H2O_gas[i] = H-T[i]*S
        i += 1
    
    ## CO2 gas
    G_CO2_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1200:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 24.99735, 55.18696, -33.69137, 7.948387, -0.136638, -403.6075, 228.2431
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_CO2_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 58.16639, 2.720074, -0.492289, 0.038844, -6.447293, -425.9186, 263.6125
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_CO2_gas[i] = H-T[i]*S
        i += 1

    ## CO gas
    G_CO_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1300:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 25.56759, 6.096130, 4.054656, -2.671301, 0.131021, -118.0089, 227.3665
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_CO_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 35.15070, 1.300095, -0.205921, 0.013550, -3.282780, -127.8375, 231.7120
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_CO_gas[i] = H-T[i]*S
        i += 1

    ## CH4 gas
    G_CH4_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1300:
            ti = T[i]/1000
            a, b, c, d, e, f, g = -0.703029, 108.4773, -42.52157, 5.862788, 0.678565, -76.84376, 158.7163
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_CH4_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 85.81217, 11.26467, -2.114146, 0.138190, -26.42221, -153.5327, 224.4143
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_CH4_gas[i] = H-T[i]*S
        i += 1
    
    ## H2 gas
    G_H2_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1000:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 43.41356, -4.293079, 1.272428, -0.096876, -20.533862, -38.515158, 162.081354
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_H2_gas[i] = H-T[i]*S

        elif T[i] >= 1000 and T[i] < 2500:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 18.563083, 12.257357, -2.859786, 0.268238, 1.977990, -1.147438, 156.288133
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_H2_gas[i] = H-T[i]*S

        elif T[i] >= 2500 and T[i] <= 6000:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_H2_gas[i] = H-T[i]*S
        i += 1
    
    ## SiO gas
    G_SiO_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1100:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 19.52413, 37.46370, -30.51805, 9.094050, 0.148934, -107.1514, 226.1506
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_SiO_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 35.69893, 1.731252, -0.509348, 0.059404, -1.248055, -114.6019, 249.1911
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_SiO_gas[i] = H-T[i]*S
        i += 1

    ## Mg gas
    G_Mg_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 2200:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 20.77306, 0.035592, -0.031917, 0.009109, 0.000461, 140.9071, 173.7799
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_Mg_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 47.60848, -15.40875, 2.875965, -0.120806, -27.01764, 97.40017, 177.2305
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_Mg_gas[i] = H-T[i]*S
        i += 1
    
    ## SiH4 gas
    G_SiH4_gas = np.zeros(len(T))
    i = 0
    while i < len(T):
        if T[i] < 1300:
            ti = T[i]/1000
            a, b, c, d, e, f, g = 6.060189, 139.9632, -77.88474, 16.24095, 0.135509, 27.39081, 174.3351
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_SiH4_gas[i] = H-T[i]*S
        else:
            a, b, c, d, e, f, g = 99.84949, 4.251530, -0.809269, 0.053437, -20.39005, -40.54016, 266.8015
            H = ((a*ti) + (b*ti**2)/2 + (c*ti**3)/3 + (d*ti**4)/4 - (e/ti) + f)*1000 #J
            S = a*ln(ti) + (b*ti) + (c*ti**2)/2 + (d*ti**3)/3 - (e/(2*ti**2)) + g
            G_SiH4_gas[i] = H-T[i]*S
        i += 1

    G_dict = {
        "G_MgO_melt" : G_MgO_melt,
        "G_MgSiO3_melt" : G_MgSiO3_melt,
        "G_SiO2_melt" : G_SiO2_melt,
        "G_FeO_melt" : G_FeO_melt,
        "G_Na2O_melt" : G_Na2O_melt,
        "G_Fe_metal" : G_Fe_metal,
        "G_H2O_gas" : G_H2O_gas,
        "G_CO2_gas" : G_CO2_gas,
        "G_CO_gas" : G_CO_gas,
        "G_O2_gas" : G_O2_gas,
        "G_CH4_gas" : G_CH4_gas,
        "G_H2_gas" : G_H2_gas,
        "G_SiO_gas" : G_SiO_gas,
        "G_Fe_gas" : G_Fe_gas,
        "G_Mg_gas" : G_Mg_gas,
        "G_Na_gas" : G_Na_gas,
        "G_SiH4_gas" : G_SiH4_gas
    }

    return G_dict