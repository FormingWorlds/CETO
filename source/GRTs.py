from __future__ import annotations

import numpy as np

from thermodynamics import get_Gibbs
from constants import R_gas, log_to_ln

Tmin = 1300
Tmax = 6000
T = np.linspace(Tmin, Tmax, 100)

Gdict = get_Gibbs(T)

## R1: Na2SiO3 = Na2O + SiO2 (in melt)
G1 = (log_to_ln*(-1.33+13870.0/T))*R_gas*T # adapted from Young line 399

## R2: 1/2 SiO2 + Fe_metal = FeO + 1/2 Si_metal (in melt)
G_Corgne = ((-log_to_ln*(2.97-21800/T))*R_gas*T) #adapted from Young line 418, Corgne et al. (2008)
G_Si_metal = G_Corgne - 2*Gdict["G_FeO_melt"] + 2*Gdict["G_Fe_metal"] + Gdict["G_SiO2_melt"]
G2 = 0.5*G_Si_metal+Gdict["G_FeO_melt"] - Gdict["G_Fe_metal"] - 0.5*Gdict["G_SiO2_melt"]

## R3: MgSiO3 = MgO + SiO2 (in melt)
G3 = Gdict["G_MgO_melt"] + Gdict["G_SiO2_melt"] - Gdict["G_MgSiO3"]

## R4: O_metal + 1/2 Si_metal = 1/2 SiO2
G_O_metal = -log_to_ln*(2.736-11439/T)*R_gas*T
G4 = -(G_O_metal + G2)

## R5: 2 H_metal = H2_melt
G_Hirschmann = R_gas*T*(-12.5-0.76*1e-4)   # Delta G from Hirschmann et al (2012), line 475 of Young code
G_H2_melt = Gdict["G_H2_gas"]+G_Hirschmann # Obtain G of H2 in melt by difference

G_Okuchi = 143589.7-T*69.1                 # Delta G for Fe + H2O_melt = FeO + 2H_metal from Okuchi (1997)
G_H2O_melt_to_vapor = -R_gas*T*(2565/T - 14.21)
G_H2O_melt = G_H2O_melt_to_vapor + Gdict["G_H2O_gas"]
G_H_metal = 0.5*(G_Okuchi-Gdict["G_FeO_melt"]+Gdict["G_Fe_metal"]+G_H2O_melt)

G5 = G_H2_melt - 2*G_H_metal

## R6: FeSiO3 = FeO + SiO2 (in melt)
G6 = log_to_ln*R_gas*T*(-0.63+3103.0/T)    # Young Code line 575

## R7: 2 H2O_melt + Si_metal = SiO2_melt + 2 H2_melt
G7 = 2*G_H2_melt+Gdict["G_SiO2_melt"]-G_Si_metal-2*G_H2O_melt

## R8: CO_gas + 1/2 O2_gas = CO2_gas
G8 = Gdict["G_CO2_gas"] - Gdict["G_CO_gas"] - 0.5*Gdict["G_O2_gas"]

## R9: CH4_gas + 1/2 O2_gas = 2 H2_gas + CO_gas
G9 = 2*Gdict["G_H2_gas"] + Gdict["G_CO_gas"] - Gdict["G_CH4_gas"] -0.5*Gdict["G_O2_gas"]

## R10: H2_gas + 1/2 O2_gas = H2O_gas
G10 = Gdict["G_H2O_gas"] - 0.5*Gdict["G_O2_gas"] - Gdict["G_H2_gas"]

## R11: FeO = Fe_gas + 1/2 O2_gas
G11 = 0.5*Gdict["G_O2_gas"] + Gdict["G_Fe_gas"] - Gdict["G_FeO_melt"]

## R12: Mg_gas + 1/2 O2_gas
G12 = 0.5*Gdict["G_O2_gas"] - Gdict["G_Mg_gas"]

## R13: SiO2_melt = SiO_gas +1/2 O2_gas
G13 = 0.5*Gdict["G_O2_gas"] + Gdict["G_SiO_gas"] - Gdict["G_SiO2_melt"]

## R14: Na2O_melt = 2Na_gas + 1/2 O2_gas
G14 = 0.5*Gdict["G_O2_gas"] + 2*Gdict["G_Na_gas"] - Gdict["G_Na2O_melt"]

## R15: H2_gas = H2_melt
G15 = G_H2_melt - Gdict["G_H2_gas"]

## R16: H2O_gas = H2O_melt
G16 = G_H2O_melt - Gdict["G_H2O_gas"]

## R17: CO_gas = CO_melt

## R18: CO2_gas = CO2_melt

## R19: SiO + 2 H2 = SiH4 + 1/2 O2 (in gas)
G19 = 0.5*Gdict["G_O2_gas"] + Gdict["G_SiH4_gas"] - 2*Gdict["G_H2_gas"] - Gdict["G_SiO_gas"]




