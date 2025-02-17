## Package Imports
import numpy as np
import numpy.testing as npt
from copy import copy, deepcopy
from pathlib import Path
from constants import *
from scipy.optimize import dual_annealing, minimize
import time

## Model Imports
from thermodynamics import *
from activity import *
from model import *
from utilities import *

starttime = time.time()

## Reading model input from config file
sourcedir = Path(__file__).parent
path_to_config = sourcedir / 'defaultconfig.txt'
config = readconfig(path_to_config)

## Seed random generators
#user_seed = D["seed"]
user_seed = 42
if user_seed == 0:
    user_seed = np.random.randint(1, 1000) #draw random seed from uniform distribution

print(f"Seeding random generator with seed={user_seed}")
np.random.seed(user_seed)                  #re-seed generator with user_seed

## Compute initial moles in system
elements_keys = ['Si', 'Mg', 'Na', 'Fe', 'C', 'H', 'O']
elements_values = [moles_in_system(element, config) for element in elements_keys]
moles_initial = dict(zip(elements_keys, elements_values))

## Compute thermodynamics
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

## Initial positions objective function
variable_keys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "P_penalty"]
variables_initial = [config[key] for key in variable_keys]

F_ini = objectivefunction_initial(variables_initial, variable_keys, config, moles_initial, G)


## Estimate mean const by sampling optimisationfunction over random variables


P_estimate = calculate_pressure(config, config)
variables = deepcopy(variables_initial)
variables[-1] = P_estimate

w_gas = 1.0/np.max(np.abs(F_ini[:19])) # Weight for thermodynamic equations 
                                       # from maximum objective function from the 19 reaction values

Value = objectivefunction(variables, variable_keys, config, moles_initial, G, w_gas)
bounds = get_bounds(config)

n_iters = 500
costs = np.zeros(n_iters)
random_variables = np.zeros(len(variables))
for i in range(n_iters):
    for j in range(len(variables)):
        random_variables[j] = np.random.uniform(bounds[j,0], bounds[j,1])
    costs[i] = objectivefunction(random_variables, variable_keys, config, moles_initial, G, w_gas)

cost_smoothed = smoothTriangle(costs, 5)
mean_cost = np.mean(cost_smoothed)
std_cost = np.std(cost_smoothed)

print(f"mean cost: {mean_cost}")
print(f"std cost: {std_cost}")

T_estimate = -std_cost / ln(0.98)
print(f"Estimated initial search T: {T_estimate}")

## Invoke Simulated Annealing
## NOTE: the use of 'seed' in dual_annealing is legacy behaviour and will cease to work at some point
## in the future. keyword 'rng' takes over the functionality, so investigate using it instead soon.

sol = dual_annealing(objectivefunction,bounds,maxiter=config["niters"],args=(variable_keys, config, moles_initial, G, w_gas), 
                     initial_temp=T_estimate, visit=2.98, maxfun=1e8, seed=user_seed, 
                     accept=-500.0, restart_temp_ratio=1.0e-9)

quality = sol.fun / mean_cost
print(sol)
print("Quality: ", quality)

endtime = time.time()
print(f"Script concluded in {endtime-starttime} s")