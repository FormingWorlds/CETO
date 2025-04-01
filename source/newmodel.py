from __future__ import annotations
import numpy as np
from numpy import log as ln
import math

try:
    from thermodynamics import calculate_GRT
    from utilities import *
    from activity import get_activity
except:
    from source.thermodynamics import calculate_GRT
    from source.utilities import *
    from source.activity import get_activity

def newobjectivefunction(var, config, initial_moles, G, w_gas, Pstd=1.0):
    varkeys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "P_penalty"]
    P = var[29]

    D = dict(zip(varkeys, var))

    wt_massbalance = w_gas*config["wt_massbalance"]
    wt_summing = w_gas*config["wt_summing"]
    wt_atm = w_gas*config["wt_atm"]
    wt_solub = w_gas*config["wt_solub"]
    wt_melt = w_gas*config["wt_melt"]
    wt_evap = w_gas*config["wt_evap"]

    #lngSi, lngO, lngH2, lngH2O_melt, lngH_metal, xB = get_activity(D, config)
    lngSi = -6.65*1873.0/config["T_eq"]-(12.41*1873.0/config["T_eq"])*ln(1.0-D["Si_metal"]) - \
        ((-5.0*1873.0/config["T_eq"])*D["O_metal"]*(1.0+ln(1-D["O_metal"])/D["O_metal"]-1.0/(1.0-D["Si_metal"]))) + \
        (-5.0*1873.0/config["T_eq"])*D["O_metal"]**2.0*D["Si_metal"]*(1.0/(1.0-D["Si_metal"])+1.0/(1.0-D["O_metal"]) + \
                                                             D["Si_metal"]/(2.0*(1.0-D["Si_metal"])**2.0)-1.0)
    lngO = (4.29-16500.0/config["T_eq"])-(-1.0*1873.0/config["T_eq"])*ln(1.0-D["O_metal"]) - \
        ((-5.0*1873.0/config["T_eq"])*D["Si_metal"]*(1.0+ln(1-D["Si_metal"])/D["Si_metal"]-1.0/(1.0-D["O_metal"]))) + \
        (-5.0*1873.0/config["T_eq"])*D["Si_metal"]**2.0*D["O_metal"]*(1.0/(1.0-D["O_metal"])+1.0/(1.0-D["Si_metal"])+ \
                                                             D["O_metal"]/(2.0*(1.0-D["O_metal"])**2.0)-1.0)
        
    xB = (D["H2O_melt"]/(1.0-D["H2O_melt"])*(1.0/3.0)) / (1.0 + (D["H2O_melt"]/(1.0-D["H2O_melt"])*(1.0/3.0)))

    lngH2 = 0.0
    lngH2O_melt = 0.0
    lngH_metal = 0.0

    f1 = config["wt_f1"]*wt_melt * ( ln(D["Na2O_melt"]) + ln(D["SiO2_melt"]) - ln(D["Na2SiO3_melt"]) + G["R1"] )
    f2 = config["wt_f2"]*wt_melt * ( 0.5*ln(D["Si_metal"]) + 0.5*lngSi + ln(D["FeO_melt"]) - 0.5*ln(D["SiO2_melt"]) - \
        ln(D["Fe_metal"]) + G["R2"] )
    f3 = config["wt_f3"]*wt_melt * ( ln(D["MgO_melt"]) + ln(D["SiO2_melt"]) - ln(D["MgSiO3_melt"]) + G["R3"] )
    f4 = config["wt_f4"]*wt_melt * ( 0.5*ln(D["SiO2_melt"]) - ln(D["O_metal"]) - lngO -0.5*ln(D["Si_metal"]) -0.5*lngSi + G["R4"] )
    f5 = config["wt_f5"]*wt_melt * ( ln(D["H2_melt"]) + lngH2 - 2.0*ln(D["H_metal"]) - 2.0*lngH_metal + G["R5"] )
    f6 = config["wt_f6"]*wt_melt * ( ln(D["FeO_melt"]) + ln(D["SiO2_melt"]) - ln(D["FeSiO3_melt"]) + G["R6"] )

    if xB != 0.0:
        f7 = config["wt_f7"]*wt_melt * ( ln(D["SiO2_melt"]) + 2.0*ln(D["H2_melt"]) + 2.0*lngH2 -4.0*ln(xB) - 2.0*lngH2O_melt - \
        ln(D["Si_metal"]) - lngSi + G["R7"] )
    else:
        f7 = config["wt_f7"]*wt_melt * ( ln(D["SiO2_melt"]) + 2.0*ln(D["H2_melt"]) + 2.0*lngH2 - 2.0*ln(D["H2O_melt"]) - 2.0*lngH2O_melt - \
        ln(D["Si_metal"]) - lngSi + G["R7"] )

    f8 = config["wt_f8"]*wt_atm * ( ln(D["CO2_gas"]) - ln(D["CO_gas"]) - 0.5*ln(D["O2_gas"]) + G["R8"] - 0.5*ln(P/Pstd) )
    f9 = config["wt_f9"]*wt_atm * ( 2.0*ln(D["H2_gas"]) + ln(D["CO_gas"]) - ln(D["CH4_gas"]) - 0.5*ln(D["O2_gas"]) + G["R9"] + 1.5*ln(P/Pstd) )
    f10 = config["wt_f10"]*wt_atm * ( ln(D["H2O_gas"]) - 0.5*ln(D["O2_gas"]) - ln(D["H2_gas"]) + G["R10"] - 0.5*ln(P/Pstd) )

    f11 = config["wt_f11"]*wt_evap * ( 0.5*ln(D["O2_gas"]) + ln(D["Fe_gas"]) - ln(D["FeO_melt"]) + G["R11"] + 1.5*ln(P/Pstd) )
    f12 = config["wt_f12"]*wt_evap * ( 0.5*ln(D["O2_gas"]) + ln(D["Mg_gas"]) - ln(D["MgO_melt"]) + G["R12"] + 1.5*ln(P/Pstd) )
    f13 = config["wt_f13"]*wt_evap * ( 0.5*ln(D["O2_gas"]) + ln(D["SiO_gas"]) - ln(D["SiO2_melt"]) + G["R13"] + 1.5*ln(P/Pstd) )
    f14 = config["wt_f14"]*wt_evap * ( 0.5*ln(D["O2_gas"]) + 2.0*ln(D["Na_gas"]) - ln(D["Na2O_melt"]) + G["R14"] + 2.5*ln(P/Pstd) )

    f15 = config["wt_f15"]*wt_solub * ( ln(D["H2_melt"]) + lngH2 - ln(D["H2_gas"]) + G["R15"] - ln(1.0e4/Pstd) ) #Fixed at 3GPa, Young line 1612

    if xB != 0.0:
        f16 = config["wt_f16"]*wt_solub * ( 2.0*ln(xB) - ln(D["H2O_gas"]) + G["R16"] - ln(P/Pstd) )
    else:
        f16 = config["wt_f16"]*wt_solub * ( ln(D["H2O_melt"]) + lngH2O_melt - ln(D["H2O_gas"]) + G["R16"] - ln(P/Pstd) )
    
    f17 = config["wt_f17"]*wt_solub * ( ln(D["CO_melt"]) - ln(D["CO_gas"]) + G["R17"] - ln(P/Pstd) )
    f18 = config["wt_f18"]*wt_solub * ( ln(D["CO2_melt"]) - ln(D["CO2_gas"]) + G["R18"] - ln(P/Pstd) )
    f19 = config["wt_f19"]*wt_atm * ( ln(D["SiH4_gas"]) + 0.5*ln(D["O2_gas"]) - ln(D["SiO_gas"]) - 2.0*ln(D["H2_gas"]) + \
        G["R19"] - 1.5*ln(P/Pstd) )
    
    f20 = config["wt_f20"]*(5.0/initial_moles["Si"])*wt_massbalance * ( initial_moles["Si"] - \
                                (D["moles_atm"]*(D["SiO_gas"]+D["SiH4_gas"]) + \
                                D["moles_melt"]*(D["SiO2_melt"] + D["FeSiO3_melt"] + D["MgSiO3_melt"] + D["Na2SiO3_melt"]) + \
                                D["moles_metal"]*(D["Si_metal"])) )
    
    f21 = config["wt_f21"]*(5.0/initial_moles["Mg"])*wt_massbalance * ( initial_moles["Mg"] - \
                                (D["moles_atm"]*(D["Mg_gas"]) + D["moles_melt"]*(D["MgO_melt"] + D["MgSiO3_melt"])) )
    
    f22 = config["wt_f22"]*(5.0/initial_moles["O"])*wt_massbalance * ( initial_moles["O"] - \
                                (D["moles_atm"]*(2.0*D["O2_gas"] + 2.0*D["CO2_gas"] + D["CO_gas"] + D["H2O_gas"] + D["SiO_gas"]) + \
                                D["moles_melt"]*(D["FeO_melt"] + D["H2O_melt"] + D["MgO_melt"] + 2.0*D["SiO2_melt"] + D["Na2O_melt"] + 3.0*D["FeSiO3_melt"] + 3.0*D["Na2SiO3_melt"] + \
                                                 3.0*D["MgSiO3_melt"] + 2.0*D["CO2_melt"] + D["CO_melt"]) + \
                                D["moles_metal"]*D["O_metal"] ) )
    
    f23 = config["wt_f23"]*(5.0/initial_moles["Fe"])*wt_massbalance * ( initial_moles["Fe"] - \
                                (D["moles_atm"]*(D["Fe_gas"]) + D["moles_metal"]*(D["Fe_metal"]) + D["moles_melt"]*(D["FeO_melt"] + D["FeSiO3_melt"])) )
    
    f24 = config["wt_f24"]*(5.0/initial_moles["H"])*wt_massbalance * ( initial_moles["H"] - \
                                (D["moles_atm"]*(4.0*D["CH4_gas"] + 2.0*D["H2_gas"] + 2.0*D["H2O_gas"] + 4.0*D["SiH4_gas"]) + \
                                 D["moles_melt"]*(2.0*D["H2O_melt"] + 2.0*D["H2_melt"]) + D["moles_metal"]*D["H_metal"]) )

    f25 = config["wt_f25"]*(5.0/initial_moles["Na"])*wt_massbalance * ( initial_moles["Na"] - \
                                (D["moles_atm"]*(D["Na_gas"]) + D["moles_melt"]*(2.0*D["Na2O_melt"] + 2.0*D["Na2SiO3_melt"]) ) )

    f26 = config["wt_f26"]*(5.0/initial_moles["C"])*wt_massbalance * ( initial_moles["C"] - \
                                (D["moles_atm"]*(D["CH4_gas"]+D["CO2_gas"]+D["CO_gas"]) + D["moles_melt"]*(D["CO_melt"]+D["CO2_melt"])) )

    f27 = config["wt_f27"]*wt_summing * ( 1.0 - D["MgO_melt"]-D["MgSiO3_melt"]-D["SiO2_melt"]-D["FeO_melt"]-D["FeSiO3_melt"]-D["Na2O_melt"]-D["Na2SiO3_melt"]-D["H2O_melt"]-D["CO2_melt"]-D["CO_melt"]-D["H2_melt"] )
    f28 = config["wt_f28"]*wt_summing * ( 1.0 - D["H_metal"] - D["O_metal"] - D["Fe_metal"] - D["Si_metal"] )
    f29 = config["wt_f29"]*wt_summing * ( 1.0 - D["CH4_gas"] - D["H2_gas"] - D["CO2_gas"] - D["CO_gas"] - D["Fe_gas"] - D["H2O_gas"] - D["SiH4_gas"] - D["SiO_gas"] - D["Mg_gas"] - D["Na_gas"] - D["O2_gas"] )

    ## Calculate pressure
    gpm_gas = D["CH4_gas"]*molwts["CH4"]+D["H2_gas"]*molwts["H2"]+D["CO2_gas"]*molwts["CO2"]+D["CO_gas"]*molwts["CO"]+D["Fe_gas"]*molwts["Fe"]+D["H2O_gas"]*molwts["H2O"]+ \
    D["SiH4_gas"]*molwts["SiH4"]+D["SiO_gas"]*molwts["SiO"]+D["Mg_gas"]*molwts["Mg"]+D["Na_gas"]*molwts["Na"]+D["O2_gas"]*molwts["O2"]

    gpm_metal = D["Fe_metal"]*molwts["Fe"]+D["Si_metal"]*molwts["Si"]+D["H_metal"]*molwts["H"]+D["O_metal"]*molwts["O"]

    gpm_melt = D["MgO_melt"]*molwts["MgO"]+D["MgSiO3_melt"]*molwts["MgSiO3"]+D["SiO2_melt"]*molwts["SiO2"]+D["FeO_melt"]*molwts["FeO"]+D["FeSiO3_melt"]*molwts["FeSiO3"]+D["Na2O_melt"]*molwts["Na2O"] + \
    D["Na2SiO3_melt"]*molwts["Na2SiO3"]+D["H2O_melt"]*molwts["H2O"]+D["CO2_melt"]*molwts["CO2"]+D["CO_melt"]*molwts["CO"]+D["H2_melt"]*molwts["H2"]


    moles_total = D["moles_atm"] + D["moles_melt"] + D["moles_metal"]
    molefrac_atm = D["moles_atm"] / moles_total
    molefrac_melt = D["moles_melt"] / moles_total
    molefrac_metal = 1.0 - molefrac_atm - molefrac_melt
    grams_atm = gpm_gas*molefrac_atm
    grams_melt = gpm_melt*molefrac_melt
    grams_metal = gpm_metal*molefrac_metal
    totalmass = grams_atm + grams_melt + grams_metal
    massfrac_atm = grams_atm / totalmass
    fratio = massfrac_atm/(1.0-massfrac_atm)
    P_guess = 1.2e6*fratio*(config["M_p"])**(2.0/3.0)
    f30 = (P_guess - D["P_penalty"]) / P_guess

    ## apply penalties
    f1 = sigmoidal_penalty(f1, 0.0, 5.0, 1.0, 10000.0)
    f2 = sigmoidal_penalty(f2, 0.0, 5.0, 1.0, 10000.0)
    f3 = sigmoidal_penalty(f3, 0.0, 5.0, 1.0, 10000.0)
    f4 = sigmoidal_penalty(f4, 0.0, 5.0, 1.0, 10000.0)
    f5 = sigmoidal_penalty(f5, 0.0, 5.0, 1.0, 10000.0)
    f6 = sigmoidal_penalty(f6, 0.0, 5.0, 1.0, 10000.0)
    f7 = sigmoidal_penalty(f7, 0.0, 5.0, 1.0, 10000.0)
    f8 = sigmoidal_penalty(f8, 0.0, 5.0, 1.0, 10000.0)
    f9 = sigmoidal_penalty(f9, 0.0, 5.0, 1.0, 10000.0)
    f10 = sigmoidal_penalty(f10, 0.0, 5.0, 1.0, 10000.0)
    f11 = sigmoidal_penalty(f11, 0.0, 5.0, 1.0, 10000.0)
    f12 = sigmoidal_penalty(f12, 0.0, 5.0, 1.0, 10000.0)
    f13 = sigmoidal_penalty(f13, 0.0, 5.0, 1.0, 10000.0)
    f14 = sigmoidal_penalty(f14, 0.0, 5.0, 1.0, 10000.0)
    f15 = sigmoidal_penalty(f15, 0.0, 5.0, 1.0, 10000.0)
    f16 = sigmoidal_penalty(f16, 0.0, 5.0, 1.0, 10000.0)
    f17 = sigmoidal_penalty(f17, 0.0, 5.0, 1.0, 10000.0)
    f18 = sigmoidal_penalty(f18, 0.0, 5.0, 1.0, 10000.0)
    f19 = sigmoidal_penalty(f19, 0.0, 5.0, 1.0, 10000.0)

    f20 = sigmoidal_penalty(f20, 0.0, 1.0, 0.01, 1000.0)
    f21 = sigmoidal_penalty(f21, 0.0, 1.0, 0.01, 1000.0)
    f22 = sigmoidal_penalty(f22, 0.0, 1.0, 0.01, 1000.0)
    f23 = sigmoidal_penalty(f23, 0.0, 1.0, 0.01, 1000.0)
    f24 = sigmoidal_penalty(f24, 0.0, 1.0, 0.01, 1000.0)
    f25 = sigmoidal_penalty(f25, 0.0, 1.0, 0.01, 1000.0)
    f26 = sigmoidal_penalty(f26, 0.0, 1.0, 0.01, 1000.0)

    f27 = sigmoidal_penalty(f27, 0.0, 1.0, 0.005, 100000.0)
    f28 = sigmoidal_penalty(f28, 0.0, 1.0, 0.005, 100000.0)
    f29 = sigmoidal_penalty(f29, 0.0, 1.0, 0.005, 100000.0)

    f30 = sigmoidal_penalty(f30, 0.0, 1.0, 0.2, config["P_penalty"])

    sum1=f1**2.0+f2**2.0+f3**2.0+f4**2.0+f5**2.0+f6**2.0+f7**2.0+f8**2.0+f9**2.0+f10**2.0
    sum1=sum1+f11**2.0+f12**2.0+f13**2.0+f14**2.0+f15**2.0+f16**2.0+f17**2.0+f18**2.0+f19**2.0
    sum2=f20**2.0+f21**2.0+f22**2.0+f23**2.0+f24**2.0+f25**2.0+f26**2.0
    sum3=(f27**2.0+f28**2.0+f29**2.0)
    sum4=f30**2.0

    sum1=1.0*sum1
    sum2=1.0*sum2
    sum3=1.0*sum3
    sum4=1.0*sum4
    sum=sum1+sum2+sum3+sum4
    return sum





