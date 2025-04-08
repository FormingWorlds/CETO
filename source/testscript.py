## Package Imports
import numpy as np
import numpy.testing as npt
from copy import deepcopy
from pathlib import Path
from constants import *
from scipy.optimize import dual_annealing, minimize
import time
import emcee
from multiprocessing.pool import Pool
from multiprocessing import get_start_method
from multiprocessing import get_context

## Model Imports
from thermodynamics import *
from activity import *
from model import *
from utilities import *
from newmodel import *

starttime = time.time()

## Reading model input from config file
sourcedir = Path(__file__).parent
path_to_config = sourcedir / 'defaultconfig.txt'

config = readconfig(path_to_config)

## Creating random number generators
rng_random = np.random.default_rng()

user_seed=config["seed"]
if user_seed == 0:
    randomseed = rng_random.integers(0, 1000)
    rng_global = np.random.default_rng(randomseed)
    print(f'Provided seed is zero; random seed for run: {randomseed}')
else:
    randomseed = user_seed
    rng_global = np.random.default_rng(randomseed)
    print(f"Seeding random generator with user seed: {user_seed}")

## Compute initial moles in system
elements_keys = ['Si', 'Mg', 'O', 'Fe', 'H', 'Na', 'C']
elements_values = [moles_in_system(element, config) for element in elements_keys]
moles_initial = dict(zip(elements_keys, elements_values))

## Compute thermodynamics
T_array  = np.linspace(1000.0, config["T_eq"], 200)
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
T_used = T_array[np.argmin(np.abs(T_array - config["T_surface"]))]

## Initial positions objective function
variable_keys = ["MgO_melt", "SiO2_melt", "MgSiO3_melt", "FeO_melt", "FeSiO3_melt", "Na2O_melt", "Na2SiO3_melt",
            "H2_melt", "H2O_melt", "CO_melt", "CO2_melt", "Fe_metal", "Si_metal", "O_metal", "H_metal", 
            "H2_gas", "CO_gas", "CO2_gas", "CH4_gas", "O2_gas", "H2O_gas", "Fe_gas", "Mg_gas", "SiO_gas", 
            "Na_gas", "SiH4_gas", "moles_atm", "moles_melt", "moles_metal", "P_penalty"]
variables_initial = [config[key] for key in variable_keys]

## Calculate objective function first value
P_initial = calculate_pressure(config, config)
variables = deepcopy(variables_initial)
variables[-1] = P_initial
F_ini = objectivefunction_initial(variables, variable_keys, config, moles_initial, G)

## Use initial values to calculate scale factor
w_gas = 1.0/np.max(np.abs(F_ini[:19])) # Weight for thermodynamic equations 
                                       # from maximum objective function from the 19 reaction values

Value = newobjectivefunction(variables, config, moles_initial, G, w_gas)
print(f"Initial value of objective function: {Value}")

# Sample objective function over 500 random sets of variables to estimate mean cost of function
bounds = get_bounds(config)
n_iters = 500
costs = np.zeros(n_iters)
random_variables = np.zeros(len(variables))
for i in range(n_iters):
    for j in range(len(variables)):
        random_variables[j] = rng_random.uniform(bounds[j,0], bounds[j,1])
    costs[i] = newobjectivefunction(random_variables, config, moles_initial, G, w_gas)
try:
    cost_smoothed = smoothTriangle(costs, 5)
except:
    print("ERROR: Number of iterations insufficient to apply smoothTriangle.")
    cost_smoothed = costs

mean_cost = np.mean(cost_smoothed)
mean_unsmoothed = np.mean(costs)
std_cost = np.std(cost_smoothed)
std_unsmoothed = np.std(costs)

print(f"Initial objective function: {Value}")
print(f"Caculating statistics over {n_iters} iterations")
print(f"    mean cost (smoothed):   {mean_cost}\n     mean cost (unsmoothed): {mean_unsmoothed}")
print(f"    std cost (smoothed):   {std_cost}\n    std cost (unsmoothed): {std_unsmoothed}")

# ## Invoke Simulated Annealing
#     ## NOTE: the use of 'seed' in dual_annealing is legacy behaviour and will cease to work at some point
#     ## in the future. keyword 'rng' takes over the functionality, so investigate using it instead soon.

T_estimate = -std_cost / ln(0.98)
print(f"Estimated initial search T: {T_estimate}")

sol = dual_annealing(newobjectivefunction,bounds,maxiter=config["niters"],args=(config, moles_initial, G, w_gas), 
                     initial_temp=1e18, visit=2.98, maxfun=1e8, seed=42, 
                     accept=-500.0, restart_temp_ratio=1.0e-9)

quality = sol.fun / mean_cost

print(sol)
print("Quality: ", quality)

## Setting up vectors for MCMC search
theta = sol.x
print(f"Vector theta: \n{theta}")

y = np.zeros(len(theta)) # vector y to contain the expected values of equilibrium equations and mass balance
Pstd = 1.0
for i in range(len(y)):
    if i < 19:
        if i == 14:
            y[i] = -(G[f"R{i+1}"] - ln(1.0e4 / Pstd))
        else:
            y[i] = -G[f"R{i+1}"]
    
    elif 19 <= i <= 25:
        y[i] = elements_values[(i-19)]
    elif i in (26, 27, 28):
        y[i] = 1.000
    elif i == 29:
        y[i] = 0.0

print(f"actual model: \n{y}")

lnk_err = 0.005    #blanket error on equilibrium reactions
moles_err = 0.0001 #blanket error on mole mass balance
sum_err = 0.00001  #blanket error on summing equations
P_err = 0.1        #fractional error for pressure
yerr = np.zeros(len(theta)) # vector yerr to contain blanket uncertainties on model equations
for i in range(len(yerr)):
    if i < 19:
        if i in (13, 16, 18):
            yerr[i] = abs(y[i]*(lnk_err / 5.0))
        else:
            yerr[i] = abs(y[i]*lnk_err)
    elif 19 <= i <= 25:
        yerr[i] = abs(y[i]*moles_err)
    elif i in (26, 27, 28):
        yerr[i] = abs(y[i]*sum_err)
    elif i == 29:
        yerr[i] = P_err 

data_emcee = (variable_keys, config, moles_initial, G, y, yerr)

y_model = model(theta, variable_keys, config, moles_initial, G, Pstd=Pstd)

print(f"Current model: \n{y_model}")
print(f"Errors on model: \n{yerr}")

## MCMC parameters
n_walkers = 200
thin = 10
n_iters_MCMC = 400000
n_iter_eff = int(n_iters_MCMC / thin)

walker_p0 = [(theta)+config["offset_MCMC"]*np.random.randn(len(theta)) for i in range(n_walkers)]

## Run MCMC search
def runMCMC(walker_p0, n_walkers, n, lnprob, data):
    ctx = get_context('fork')
    with Pool(6, context=ctx) as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n, lnprob, pool=pool, args=data)
        print("Initial burn in running ... ")
        pos,_,_ = sampler.run_mcmc(walker_p0, 500)

        sampler.reset()

        print("Running full MCMC search")
        pos, prob, state = sampler.run_mcmc(pos, n_iter_eff, skip_initial_state_check=False, progress=True, thin_by=thin)

    return sampler, pos, prob, state

sampler, pos, prob, state = runMCMC(walker_p0, n_walkers, len(variables), lnprob, data_emcee)
print("MCMC search finished, processing results")

samples = sampler.get_chain()
posteriors = sampler.get_log_prob()

samples_flat = sampler.flatchain
posteriors_flat = sampler.flatlnprobability

MCMCresult = samples_flat[np.argmax(sampler.flatlnprobability)] # set of params theta with greatest posterior probability
bestfit_model = model(MCMCresult, variable_keys, config, moles_initial, G)

## Calculate statistics
chi2 = sum(((y - bestfit_model)**2) / yerr**2)
reduced_chi2 = chi2 / (float(71) - float(len(variables_initial))) #taken from Young's code
mean_af = np.mean(sampler.acceptance_fraction)
GR = gelman_rubin(samples)
refinement = MCMCresult - sol.x

print(f"Reduced chi2 of fit: {reduced_chi2}")

print(f"Mean acceptance fraction: {mean_af}")

print(f"Gelman-Rubin statistic for each variable (values in the range of 0.9 - 1.1 are acceptable): \n{GR}")

print(f"Absolute difference between dual_annealing and end of MCMC best-fit parameters: \n{refinement}")

## Calculate results
result = dict(zip(variable_keys, MCMCresult))

# Check mass balance
final_element_values = [moles_in_system(element, result) for element in elements_keys]
for i in range(len(final_element_values)):
    print(f" Initial n{elements_keys[i]}: {elements_values[i]}\n Final n{elements_keys[i]}: {final_element_values[i]}")

# wt% of phase components
wtpcs = calculate_wtpc(result)

# Oxygen Fugacities
DIW_apparent = 2.0*log((result["FeO_melt"] + result["FeSiO3_melt"]) / 0.85) #DIW melt for Fe_metal = 0.85
DIW_actual = 2.0*log((result["FeO_melt"] + result["FeSiO3_melt"]) / result["Fe_metal"])

K_IW = 10.0**(-(G["R11"]/log_to_ln))
logPO2_IW = log(K_IW**2.0)
logPO2_atm = log(result["O2_gas"]*result["P_penalty"])
DIW_atm = logPO2_atm - logPO2_IW

# Planet densities
gpm_atm, gpm_melt, gpm_metal = gpm_phases(result)
moles_total = result["moles_atm"] + result["moles_melt"] + result["moles_metal"]
mass_atm = (result["moles_atm"]/moles_total) * gpm_atm #g
mass_melt = (result["moles_melt"]/moles_total) * gpm_melt #g
mass_metal = (result["moles_metal"]/moles_total) * gpm_melt #g
mass_total = mass_atm + mass_melt + mass_metal

massfrac_atm = mass_atm / mass_total
massfrac_melt = mass_melt / mass_total
massfrac_metal = mass_metal / mass_total

density_atm = gpm_atm / (10.0*R_gas*T_used*result["P_penalty"])

density_melt = sum([(result[key]*(molwts[key.replace('_melt','')]/(10.0*molvols[key.replace('_melt','')]))) for \
                       key in result if '_melt' in key and 'moles' not in key])

density_metal = sum([(result[key]*(molwts[key.replace('_metal','')]/(10.0*molvols[key.replace('_metal','')]))) for \
                     key in result if '_metal' in key and 'moles' not in key])

density_pureFe = molwts["Fe"]/(10.0*molvols["Fe"])

density_deficit_A = ((density_pureFe - density_metal)/density_pureFe)*100 # in %
density_deficit_B = (8.7*wtpcs["wt_H_metal"]+1.2*wtpcs["wt_O_metal"]+0.8*["wt_Si_metal"]) # in %; taken from Young's code, evidently 'deficit based on experimental values'

density_planet = (massfrac_metal*density_metal) + (massfrac_melt*density_melt) + (massfrac_atm*density_atm) #g/cc

# Nominal atmospheric pressure
fratio = massfrac_atm/(1.0 - massfrac_atm)
nominal_pressure = 1.2e6*fratio*(config["M_p"])**(2.0/3.0)





endtime = time.time()
print(f"Script concluded in {endtime-starttime} s")