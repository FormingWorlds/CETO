import numpy as np
import numpy.testing as npt
from copy import deepcopy
import pytest

from source.utilities import *
from source.thermodynamics import calculate_GRT
from source.model import *

## Set up function to be tested
projectdir = Path(__file__).parent
path_to_config = projectdir / 'defaultconfig.txt'
config = readconfig(path_to_config)

variable_keys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "P_penalty"]
var_initial = [config[key] for key in variable_keys]

elements_keys = ['Si', 'Mg', 'O', 'Fe', 'H', 'Na', 'C']
elements_values = [moles_in_system(element, config) for element in elements_keys]
moles_initial = dict(zip(elements_keys, elements_values))

T_array  = np.linspace(1300.0, config["T_eq"], 200)
GRT_list = calculate_GRT(T_array)
GRT_vals = []; GRT_keys = []
for i in range(len(GRT_list)):
    if i == 1 or i == 3 or i == 4 or i == 6:
        GRT_T = GRT_list[i][np.argmin(np.abs(T_array - config["T_eq"]))] # For reactions 2, 4, 5, 7 (partitioning reactions between melt/metal) use T_eq
    else:
        GRT_T = GRT_list[i][np.argmin(np.abs(T_array - config["T_surface"]))] # Otherwise just find Gibbs free energy at closest T to surface temperature
    GRT_vals.append(GRT_T)
    GRT_keys.append(f"R{i+1}")
G = dict(zip(GRT_keys, GRT_vals))

bounds = get_bounds(config)
P_estimate = calculate_pressure(config, config)
variables = deepcopy(var_initial)
variables[-1] = P_estimate
F_ini = objectivefunction_initial(variables, variable_keys, config, moles_initial, G)
w_gas = 1.0/np.max(np.abs(F_ini[:19]))

## Define test arrays (output from Young's code; mean, smoothed mean, std, smoothed std, T_ini)
expected_23_500 = [3704990699651723.5, 3705134607282833.0, 6.838297722758838e+16, 2.49862863078713e+16, 1.2367791066400092e+18]             #seed=23, 500 iters
expected_42_5000 = [1.1558406118368118e+18, 1.1558405914019684e+18, 5.025781560331543e+19, 1.8503024845316248e+19, 9.158685791221021e+20]   #seed=42, 5000 iters
expected_77_50000 = [4.786266314930234e+18, 4.786266317736509e+18, 9.938329813941407e+20, 3.6648049335484835e+20, 1.8140167433749946e+22]   #seed=77, 50000 iters

testdata = [
    (23, 500, expected_23_500),
    (42, 5000, expected_42_5000),
    (77, 50000, expected_77_50000)
]

@pytest.mark.parametrize("inputseed, niters, expected", testdata)
def test_cost_estimation(inputseed, niters, expected):
    np.random.seed(inputseed)

    costs = np.zeros(niters)
    random_variables = np.zeros(len(variables))
    for i in range(niters):
        for j in range(len(variables)):
            random_variables[j] = np.random.uniform(bounds[j, 0], bounds[j, 1])
        costs[i] = objectivefunction(random_variables, variable_keys, config, moles_initial, G, w_gas)

    costs_smoothed = smoothTriangle(costs, 5)

    mean_unsmoothed = np.mean(costs)
    std_unsmoothed = np.std(costs)
    mean_smoothed = np.mean(costs_smoothed)
    std_smoothed = np.std(costs_smoothed)
    T_ini = -std_smoothed / np.log(0.98)

    testresult = [mean_unsmoothed, mean_smoothed, std_unsmoothed, std_smoothed, T_ini]
    npt.assert_allclose(np.array(testresult), np.array(expected), rtol=1e-4) #Uses np.assert_allclose to test for relative difference no greater than 0.0001