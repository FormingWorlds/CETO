import numpy as np
import numpy.testing as npt
from source.utilities import *
from source.thermodynamics import calculate_GRT
from source.model import objectivefunction_initial

## Reading model input from config file
projectdir = Path(__file__).parent
path_to_config = projectdir / 'defaultconfig.txt'
config = readconfig(path_to_config)

variable_keys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "P_penalty"]
var_initial = [config[key] for key in variable_keys]

nSi = moles_in_system('Si', config)
nMg = moles_in_system('Mg', config)
nNa = moles_in_system('Na', config)
nFe = moles_in_system('Fe', config)
nC = moles_in_system('C', config)
nH = moles_in_system('H', config)
nO = moles_in_system('O', config)

moles_initial = {'Si' : nSi,
                 'Mg' : nMg,
                 'Fe' : nFe,
                 'Na' : nNa,
                 'H'  : nH,
                 'O'  : nO,
                 'C'  : nC}

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

## Read array of computed values from Young model as test
path_to_expected = projectdir / 'testvalues_initialoptimisation.txt'

def test_objectivefunction():
    test_input = objectivefunction_initial(var_initial, variable_keys, config, moles_initial, G)
    expected_result = np.genfromtxt(path_to_expected)
    npt.assert_array_almost_equal(test_input, expected_result, decimal=6)




