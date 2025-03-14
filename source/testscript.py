## Package Imports
import numpy as np
import numpy.testing as npt
from copy import copy, deepcopy
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
elements_keys = ['Si', 'Mg', 'O', 'Fe', 'H', 'Na', 'C']
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

## Calculate objective function first value
P_estimate = calculate_pressure(config, config)

variables = deepcopy(variables_initial)
variables[-1] = P_estimate
F_ini = objectivefunction_initial(variables, variable_keys, config, moles_initial, G)

## Use initial values to calculate scale factor
w_gas = 1.0/np.max(np.abs(F_ini[:19])) # Weight for thermodynamic equations 
                                       # from maximum objective function from the 19 reaction values

Value = objectivefunction(variables, variable_keys, config, moles_initial, G, w_gas)
print(f"Initial value of objective function: {Value}")

# theta = deepcopy(variables_initial)
# y_model = model(theta, variable_keys, config, moles_initial, G)
# for i in range(len(y_model)):
#     print(y_model[i])

# Sample objective function over 500 random sets of variables to estimate mean cost of function
bounds = get_bounds(config)
n_iters = 5000
costs = np.zeros(n_iters)
random_variables = np.zeros(len(variables))
for i in range(n_iters):
    # np.random.seed(i+2) Uncomment to force same random variables each run
    for j in range(len(variables)):
        random_variables[j] = np.random.uniform(bounds[j,0], bounds[j,1])
    costs[i] = objectivefunction(random_variables, variable_keys, config, moles_initial, G, w_gas)
try:
    cost_smoothed = smoothTriangle(costs, 5)
except:
    print("ERROR: Number of iterations insufficient to apply smoothTriangle.")
    cost_smoothed = costs

mean_cost = np.mean(cost_smoothed)
mean_unsmoothed = np.mean(costs)
std_cost = np.std(cost_smoothed)
std_unsmoothed = np.std(costs)

#print(f"Initial objective function: {Value}")
print(f"Caculating statistics over {n_iters} iterations")
print(f"    mean cost (smoothed):   {mean_cost}\n     mean cost (unsmoothed): {mean_unsmoothed}")
print(f"    std cost (smoothed):   {std_cost}\n    std cost (unsmoothed): {std_unsmoothed}")

# ## Invoke Simulated Annealing
#     ## NOTE: the use of 'seed' in dual_annealing is legacy behaviour and will cease to work at some point
#     ## in the future. keyword 'rng' takes over the functionality, so investigate using it instead soon.

T_estimate = -std_cost / ln(0.98)
print(f"Estimated initial search T: {T_estimate}")


sol = dual_annealing(objectivefunction,bounds,maxiter=10000,args=(variable_keys, config, moles_initial, G, w_gas), 
                     initial_temp=50000, visit=2.98, maxfun=1e8, seed=user_seed, 
                     accept=-500.0, restart_temp_ratio=1.0e-9)

# sol = dual_annealing(objectivefunction,bounds,maxiter=config["niters"],args=(variable_keys, config, moles_initial, G, w_gas), 
#                      initial_temp=T_estimate, visit=2.98, maxfun=1e8, seed=user_seed, 
#                      accept=-500.0, restart_temp_ratio=1.0e-9)

quality = sol.fun / mean_cost

print(sol)
print("Quality: ", quality)

## Setting up vectors for MCMC search
theta_self = sol.x
theta = np.array([5.08532272e-02, 3.10898511e-01, 1.82280854e-01, 9.94405227e-02,
 5.28188124e-03, 2.68694569e-05, 5.60674644e-03, 5.68730380e-02,
 3.14517876e-01, 2.16406471e-04, 2.21736058e-04, 4.81660600e-01,
 8.70924369e-02, 2.80621197e-01, 2.18468108e-01, 1.41868508e-01,
 1.72113066e-03, 2.33213897e-02, 1.49574644e-03, 8.57958613e-06,
 3.66646470e-01, 3.04849063e-03, 1.69431603e-03, 2.49340375e-03,
 3.35731784e-03, 5.76584565e-01, 1.50468449e+01, 2.98727064e+03,
 2.05541488e+03, 6.44191236e+02])
print(f"Vector theta from self calculation: \n{theta_self}")

# y = np.zeros(len(theta)) # vector y to contain the expected values of equilibrium equations and mass balance
# for i in range(len(y)):
#     if i < 19:
#         if i == 14:
#             y[i] = -(G[f"R{i+1}"] - ln(1e4 / Pstd))
#         else:
#             y[i] = -G[f"R{i+1}"]
    
#     elif 19 <= i <= 25:
#         y[i] = elements_values[(i-19)]
#     elif i in (26, 27, 28):
#         y[i] = 1.000
#     elif i == 29:
#         y[i] = 0.0

# print(f"actual model: \n{y}")

# lnk_err = 0.005    #blanket error on equilibrium reactions
# moles_err = 0.0001 #blanket error on mole mass balance
# sum_err = 0.00001  #blanket error on summing equations
# P_err = 0.1        #fractional error for pressure
# yerr = np.zeros(len(theta)) # vector yerr to contain blanket uncertainties on model equations
# for i in range(len(yerr)):
#     if i < 19:
#         if i in (13, 16, 18):
#             yerr[i] = abs(y[i]*(lnk_err / 5.0))
#         else:
#             yerr[i] = abs(y[i]*lnk_err)
#     elif 19 <= i <= 25:
#         yerr[i] = abs(y[i]*moles_err)
#     elif i in (26, 27, 28):
#         yerr[i] = abs(y[i]*sum_err)
#     elif i == 29:
#         yerr[i] = P_err 

# data_emcee = (variable_keys, config, moles_initial, G, y, yerr)

# y_model = model(theta, variable_keys, config, moles_initial, G, Pstd=Pstd)

# print(f"Current model: \n{y_model}")
# print(f"Errors on model: \n{yerr}")

# print(f"Likelihood for model (with initial state as test)= \n{lnlikelihood(theta, variable_keys, config, moles_initial, G, y, yerr)}")
# print(f"ln prob for model: (with initial state as test) \n{lnprob(theta, variable_keys, config, moles_initial, G, y, yerr)}")

# exit()
# ## MCMC parameters
# n_walkers = 200
# thin = 10
# n_iters_MCMC = 1000000
# n_iter_eff = int(n_iters_MCMC / thin)

# walker_p0 = [(theta)+config["offset_MCMC"]*np.random.randn(len(theta)) for i in range(n_walkers)]

# ## Run MCMC search
# def runMCMC(walker_p0, n_walkers, n, lnprob, data):
#     ctx = get_context('fork')
#     with Pool(6, context=ctx) as pool:
#         sampler = emcee.EnsembleSampler(n_walkers, n, lnprob, pool=pool, args=data)
#         print("Initial burn in running ... ")
#         pos,_,_ = sampler.run_mcmc(walker_p0, 500)

#         sampler.reset()

#         print("Running full MCMC search")
#         pos, prob, state = sampler.run_mcmc(pos, n_iter_eff, skip_initial_state_check=False, progress=True, thin_by=thin)

#     return sampler, pos, prob, state

# sampler, pos, prob, state = runMCMC(walker_p0, n_walkers, len(variables), lnprob, data_emcee)

# ## Processing results
# print("MCMC search finished, processing results")

# samples = sampler.flatchain
# posteriors = sampler.flatlnprobability

# MCMCresult = samples[np.argmax(sampler.flatlnprobability)] # set of params theta with greatest posterior probability
# bestfit_model = model(MCMCresult, variable_keys, config, moles_initial, G)

# chi2 = sum(((y - bestfit_model)**2) / yerr**2)
# #reduced_chi2 = chi2 / float(len(variables_initial)) # Chi2 per degrees of freedom (as defined)
# reduced_chi2 = chi2 / (float(71) - float(len(variables_initial))) #taken from Young's code

# refinement = MCMCresult - sol.x

# print(f"Reduced chi2 of fit: {reduced_chi2}")
# print(f"Absolute difference between dual_annealing and end of MCMC best-fit parameters: \n{refinement}")

# ## Save a .txt file with the best-fit model for analysis; saves us having to re-run the model each time.
# result_filename = "modelresult_TEST_27_02"
# np.savetxt(result_filename, bestfit_model)

endtime = time.time()
print(f"Script concluded in {endtime-starttime} s")