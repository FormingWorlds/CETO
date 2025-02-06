from __future__ import annotations

## Earth constants
M_earth = 5.972e24           # kg
R_earth = 6.335439e6         # m
R_core_earth = 3.485e6       # m
M_core_earth = 1.94e24       # kg
ocean_moles = 7.688949e22    # moles of H2 (or H20) in one present-day Earth Ocean

## Physical constants
R_gas = 8.31446261815324     # J/K/mole
G = 6.67428e-11              # Gravitational Constant in N m2/kg2
k_b = 1.3806485279e-23       # Boltzmann constant, J/K
mol = 6.02214076e23

## Conversion Factors
log_to_ln = 2.302585093      # change base of logarithms

## Molecular Weights (g/mol)
molwts = {
    "H"       : 1.00794,
    "MgO"     : 40.3044,
    "SiO2"     : 60.08,
    "FeO"     : 71.844,
    "MgSiO3"  : 100.39,
    "FeSiO3"  : 131.9287,
    "Na2O"    : 61.9789,
    "Na2SiO3" : 122.063,
    "H2"      : 2.016,
    "H2O"     : 18.01528,
    "CO"      : 28.01,
    "CO2"     : 44.0095,
    "Fe"      : 55.847,
    "Si"      : 28.0855,
    "O"       : 15.9994,
    "CH4"     : 16.04,
    "O2"      : 31.9988,
    "Mg"      : 24.305,
    "SiO"     : 44.08,
    "Na"      : 22.98977,
    "SiH4"    : 32.12
}

## Molar Volumes (cm3 / mol, approximate)
molvols = {
    "MgO"     : 1.246,
    "SiO2"    : 1.3505,
    "MgSiO3"  : 3.958,
    "FeO"     : 1.319,
    "FeSiO3"  : 4.031,
    "Na2O"    : 2.997,
    "Na2SiO3" : 5.709,
    "H2"      : 1.10,
    "H2O"     : 1.92,
    "CO"      : 3.958,
    "CO2"     : 2.3,
    "Fe"      : 0.776,
    "Si"      : 1.0886,
    "O"       : 0.4,
    "H"       : 0.266
}

