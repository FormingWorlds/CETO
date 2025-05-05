#!/dataserver/users/formingworlds/lania/mscthesis/venv/bin/python

## Package Imports
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from constants import *
from scipy.optimize import dual_annealing, minimize
import time
import emcee
import corner
import logging
from multiprocessing.pool import Pool
from multiprocessing import get_start_method
from multiprocessing import get_context
import argparse

## Model Imports
from thermodynamics import *
from activity import *
from model import *
from utilities import *
from newmodel import *

parser = argparse.ArgumentParser("Argument parser for testscript.py")
parser.add_argument("-input", required=True, default="defaultconfig.txt")
parser.add_argument("--logname", nargs='?', default='lania.log')
parser.add_argument("--runID",nargs='?', default="lania")
parser.add_argument("--doplots",nargs='?',default='True')
args = parser.parse_args()

inputfile = args.input
logname = args.logname
runID = args.runID
bool_plots = args.doplots

## Creating logging instance
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logname, filemode='w', encoding='utf-8', level=logging.INFO, 
                    format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S', force=True)

logging.info("Starting model run")
starttime = time.time()


## Reading model input from config file
sourcedir = Path(__file__).parent
path_to_config = sourcedir / inputfile

logging.info(f"Attempting to read config file: {path_to_config}")

try:
    config = readconfig(path_to_config)
except FileNotFoundError:
    logging.critical(f"Config file {path_to_config} not found.")
    exit()

## Creating random number generators
rng_random = np.random.default_rng()

user_seed=config["seed"]
if user_seed == 0:
    randomseed = rng_random.integers(0, 1000)
    rng_global = np.random.default_rng(randomseed)
    logging.info(f"Config seed is zero; random seed for run: {randomseed}")
else:
    randomseed = user_seed
    rng_global = np.random.default_rng(randomseed)
    logging.info(f"Config seed = {user_seed}, seeding random generator.")

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

Value = objectivefunction(variables, variable_keys, config, moles_initial, G, w_gas)
print(f"Initial value of objective function: {Value}")

# Sample objective function over 500 random sets of variables to estimate mean cost of function
bounds = get_bounds(config)
n_iters = 500
logging.info(f"Calculating statistics over {n_iters} iterations")
costs = np.zeros(n_iters)
random_variables = np.zeros(len(variables))
for i in range(n_iters):
    for j in range(len(variables)):
        random_variables[j] = rng_random.uniform(bounds[j,0], bounds[j,1])
    costs[i] = objectivefunction(random_variables, variable_keys, config, moles_initial, G, w_gas)
try:
    cost_smoothed = smoothTriangle(costs, 5)
except:
    logging.warning("Number of iterations insufficient to apply smoothTriangle")
    print("ERROR: Number of iterations insufficient to apply smoothTriangle.")
    cost_smoothed = costs

mean_cost = np.mean(cost_smoothed)
mean_unsmoothed = np.mean(costs)
std_cost = np.std(cost_smoothed)
std_unsmoothed = np.std(costs)

logging.info(f"mean cost (smoothed): {mean_cost}\n mean cost (unsmoothed): {mean_unsmoothed}")
logging.info(f"std cost (smoothed): {std_cost}\n std cost (unsmoothed): {std_unsmoothed}")

# ## Invoke Simulated Annealing
#     ## NOTE: the use of 'seed' in dual_annealing is legacy behaviour and will cease to work at some point
#     ## in the future. keyword 'rng' takes over the functionality, so investigate using it instead soon.

T_estimate = -std_cost / ln(0.98)
logging.info(f"Estimated initial search T (by -std_cost / ln(0.98) = {T_estimate})")

time_start_dual_annealing = time.time()
sol = dual_annealing(objectivefunction,bounds,maxiter=config['niters'],args=(variable_keys, config, moles_initial, G, w_gas), 
                     initial_temp=1e18, visit=2.98, maxfun=1e8, seed=randomseed, 
                     accept=-500.0, restart_temp_ratio=1.0e-9)

quality = sol.fun / mean_cost

logging.info(f"Simulated annealing result: \n {sol}")

time_end_dual_annealing = time.time()
logging.info(f"Time needed for Simulated Annealing search: {time_end_dual_annealing - time_start_dual_annealing}")

## Setting up vectors for MCMC search
theta = sol.x
logging.info(f"Vector theta from Simulated Annealing: \n {theta}")


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

logging.info(f"Expected values of equilibrium equations and mass balance: \n{y}")

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

logging.info(f"Model before MCMC search:")
for i in range(len(y_model)):
    logging.info(f"y{i+1} = {y_model[i]} +- {yerr[i]}")
logging.info(f"")

print(f"Current model: \n{y_model}")
print(f"Errors on model: \n{yerr}")

## MCMC parameters
time_start_MCMC = time.time()

n_walkers = 200
thin = 10
n_iters_MCMC = 7500
n_iter_eff = int(n_iters_MCMC / thin)

walker_p0 = [(theta)+config["offset_MCMC"]*np.random.randn(len(theta)) for i in range(n_walkers)]

logging.info(f"Running MCMC search for {n_iters_MCMC} iterations")

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
time_end_MCMC = time.time()

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
logging.info(f"Model reduced chi2 of fit: {reduced_chi2}")

print(f"Mean acceptance fraction: {mean_af}")
logging.info(f"Model mean acceptance fraction: {mean_af}")

print(f"Gelman-Rubin statistic for each variable (values in the range of 0.9 - 1.1 are acceptable): \n{GR}")
for i in range(len(GR)):
    logging.debug(f"Gelman-Rubin statistic for {variable_keys[i]}: {GR[i]}")

print(f"Absolute difference between dual_annealing and end of MCMC best-fit parameters: \n{refinement}")

## Obtain spread around median (16, 50, 84 percentile)
percentile_spread_params = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_flat, [16, 50, 84], axis = 0)))) #Calculates uncertainties wrt 16th, 50th (medians) and 84th percentile
for i in range(len(percentile_spread_params)):
    print(f"{variable_keys[i]}: median = {percentile_spread_params[i][0]}, 16th percentile = {percentile_spread_params[i][1]}, 84th percentile = {percentile_spread_params[i][2]}")

## Obtain n next-best samples after the MAP to gauge spread around maximum posterior estimate
num_next_best = 500
partitioned_indices = np.argpartition(posteriors_flat, -num_next_best)[-num_next_best:]
top_n_indices = partitioned_indices[np.argsort(posteriors_flat[partitioned_indices])][::-1]
n_best_samples = samples_flat[top_n_indices]

for i in range(len(variable_keys)):
    key = variable_keys[i]
    logging.info(f"{key}: MAP result = {n_best_samples[0,i]}\n mean over {num_next_best} most probable samples: {np.mean(n_best_samples[:,i])}\n standard deviation over {num_next_best} most probable samples: {np.std(n_best_samples[:,i])}")

if bool_plots == 'True' or bool_plots == 'true':
    for i in range(len(variable_keys)):
        key = variable_keys[i]
        points = np.zeros(num_next_best)
        for j in range(len(n_best_samples)):
            points[j] = n_best_samples[j,i]

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        ax.hist(points, bins=30, alpha=0.3)
        ax.vlines(points[0], 0, 30, color='red', linestyle='--', label='MAP')
        ax.vlines(np.mean(points), 0, 30, color='green', linestyle='--', label='mean')
        ax.set_xlabel(f"{key}", fontsize='x-large')
        ax.set_ylabel("Counts", fontsize='x-large')
        plt.show()
        fig.savefig(f"{runID}_{key}_hist.png")
        plt.close()

## Calculate results and propagate uncertainties
errs = dict(zip(variable_keys,[np.std(n_best_samples[:,i]) for i in range(len(variables))]))

result = dict(zip(variable_keys, MCMCresult))
print("Best-fit parameters: \n")
for key in result:
    logging.info(f"result {key} = {result[key]} +- {errs[key]}")
    print(f"{key} : {result[key]} +- {errs[key]}")

# Check mass balance
final_elements_values = [moles_in_system(element, result) for element in elements_keys]
for i in range(len(final_elements_values)):
    print(f" Initial n{elements_keys[i]}: {elements_values[i]}\n Final n{elements_keys[i]}: {final_elements_values[i]}")

# wt% of phase components
wtpcs = calculate_wtpc_uncertainties(result, errs)

# Oxygen Fugacities
DIW_apparent = 2.0*log((result["FeO_melt"] + result["FeSiO3_melt"]) / 0.85) #DIW melt for Fe_metal = 0.85
DIW_actual = 2.0*log((result["FeO_melt"] + result["FeSiO3_melt"]) / result["Fe_metal"])


A = result["FeO_melt"] + result["FeSiO3_melt"]
err_A = np.sqrt(errs["FeO_melt"]**2 + errs["FeSiO3_melt"]**2)
B = A / result["Fe_metal"]
err_B = np.abs(B)*np.sqrt(((err_A)/A)**2 + (errs["Fe_metal"]/result["Fe_metal"])**2)

err_DIW_apparent = np.abs(2*(err_A)/(A*ln(10)))
err_DIW_actual = np.abs(2.0*(err_B / (B*ln(10))))


K_IW = 10.0**(-(G["R11"]/log_to_ln))
logPO2_IW = log(K_IW**2.0)
logPO2_atm = log(result["O2_gas"]*result["P_penalty"])
DIW_atm = logPO2_atm - logPO2_IW

err_logPO2_atm = np.abs((np.abs(logPO2_atm)*np.sqrt((errs["O2_gas"]/result["O2_gas"])**2 + (errs["P_penalty"]/result["P_penalty"])**2) \
                        / (ln(10)*(result["O2_gas"]*result["P_penalty"]))))
err_DIW_atm = err_logPO2_atm

## Mass Fraction
gpm_atm, gpm_melt, gpm_metal = gpm_phases_uncertainties(result, errs)
moles_total = result["moles_atm"] + result["moles_melt"] + result["moles_metal"]
err_moles_total = (errs["moles_atm"]**2 + errs["moles_melt"]**2 + errs["moles_metal"]**2)**0.5


mass_atm = (result["moles_atm"]/moles_total) * gpm_atm[0] #g
mass_melt = (result["moles_melt"]/moles_total) * gpm_melt[0] #g
mass_metal = (result["moles_metal"]/moles_total) * gpm_metal[0] #g
mass_total = mass_atm + mass_melt + mass_metal

err_mass_atm = mass_atm*np.sqrt((errs["moles_atm"]/result["moles_atm"])**2 + (err_moles_total/moles_total)**2 + \
                                (gpm_atm[1]/gpm_atm[0])**2)
err_mass_melt = mass_melt*np.sqrt((errs["moles_melt"]/result["moles_melt"])**2 + (err_moles_total/moles_total)**2 + \
                                (gpm_melt[1]/gpm_melt[0])**2)
err_mass_metal = mass_metal*np.sqrt((errs["moles_metal"]/result["moles_metal"])**2 + (err_moles_total/moles_total)**2 + \
                                (gpm_metal[1]/gpm_metal[0])**2)
err_mass_total = (err_mass_atm**2 + err_mass_melt**2 + err_mass_metal**2)**0.5


massfrac_atm = mass_atm / mass_total
massfrac_melt = mass_melt / mass_total
massfrac_metal = mass_metal / mass_total

err_mf_atm = massfrac_atm*np.sqrt((err_mass_atm / mass_atm)**2 + (err_mass_total / mass_total)**2)
err_mf_melt = massfrac_melt*np.sqrt((err_mass_melt / mass_melt)**2 + (err_mass_total / mass_total)**2)
err_mf_metal = massfrac_metal*np.sqrt((err_mass_metal / mass_metal)**2 + (err_mass_total / mass_total)**2)

## Planet Densities
density_atm = gpm_atm[0] / (10.0*R_gas*T_used*result["P_penalty"])
err_density_atm = np.abs(density_atm)*np.sqrt((gpm_atm[1]/gpm_atm[0])**2 + (errs["P_penalty"]/result["P_penalty"])**2)


density_melt = sum([(result[key]*(molwts[key.replace('_melt','')]/(10.0*molvols[key.replace('_melt','')]))) for \
                       key in result if '_melt' in key and 'moles' not in key])

err_density_melt = np.sqrt(sum([((errs[key])**2)*(molwts[key.replace('_melt','')]/(10*molvols[key.replace('_melt','')]))**2 \
                               for key in errs if '_melt' in key and 'moles' not in key]))

density_metal = sum([(result[key]*(molwts[key.replace('_metal','')]/(10.0*molvols[key.replace('_metal','')]))) for \
                     key in result if '_metal' in key and 'moles' not in key])

err_density_metal = np.sqrt(sum([((errs[key])**2)*(molwts[key.replace('_metal','')]/(10*molvols[key.replace('_metal','')]))**2 \
                               for key in errs if '_metal' in key and 'moles' not in key]))

density_pureFe = molwts["Fe"]/(10.0*molvols["Fe"])

density_deficit = ((density_pureFe - density_metal)/density_pureFe)*100 # in %
density_deficit_experimental = (8.7*wtpcs["wt_H_metal"][0]+1.2*wtpcs["wt_O_metal"][0]+0.8*wtpcs["wt_Si_metal"][0]) # in %; taken from Young's code, evidently 'deficit based on experimental values'
err_density_deficit = np.sqrt((100/density_pureFe)**2 * (err_density_metal)**2)


density_planet = (massfrac_metal*density_metal) + (massfrac_melt*density_melt) + (massfrac_atm*density_atm) #g/cc

# Nominal atmospheric pressure
fratio = massfrac_atm/(1.0 - massfrac_atm)
nominal_pressure = 1.2e6*fratio*(config["M_p"])**(2.0/3.0)
err_p_nominal = np.abs(1.2e6 * 4.0**(2.0/3.0))*err_mf_atm

## Printing to check
print("wt% values:")
for key in wtpcs:
    print(f"{key}: {wtpcs[key][0]} +- {wtpcs[key][1]}")

print("Mass fractions:")
print(f"atm: {massfrac_atm} +- {err_mf_atm}")
print(f"melt: {massfrac_melt} +- {err_mf_melt}")
print(f"metal: {massfrac_metal} +- {err_mf_metal}")

print("Densities:")
print(f"atm: {density_atm} +- {err_density_atm}")
print(f"melt: {density_melt} +- {err_density_melt}")
print(f"metal: {density_metal} +- {err_density_metal}")
print(f"deficit: {density_deficit} +- {err_density_deficit}")

print("DIWs")
print(f"DIW atm: {DIW_atm} +- {err_DIW_atm}")
print(f"DIW_apparent: {DIW_apparent} +- {err_DIW_apparent}")
print(f"DIW_actual: {DIW_actual} +- {err_DIW_actual}")

print(f"nominal pressure: {nominal_pressure} +- {err_p_nominal}")
## End of check clause

## Deviations from equilibrium, mass balance and summing constraints
## KEQ are equilibrium constants; K represents result from modelling
KEQ = np.array([exp(y[i]) for i in range(19)])
K = np.array([exp(bestfit_model[i]) for i in range(19)])

mass_balance = np.array(elements_values) - np.array(final_elements_values)

sum_melt = 1.0 - np.sum([result[key] for key in result if '_melt' in key and 'moles' not in key and 'wt' not in key])
sum_metal = 1.0  - np.sum([result[key] for key in result if '_metal' in key and 'moles' not in key and 'wt' not in key])
sum_atm = 1.0 - np.sum([result[key] for key in result if '_gas' in key and 'moles' not in key and 'wt' not in key])


## Writing results to outputfile
outputname = f'{runID}_output_summary.txt'
outfile = open(outputname, 'w')

outfile.write("%10.5f " %reduced_chi2); outfile.write(" # Reduced chi-squre of fit\n")
outfile.write(f"{user_seed} "); outfile.write("# User-provided seed\n")
outfile.write(f"{randomseed} " ); outfile.write("# Generated seed\n")
outfile.write("%10.5f "% mean_af); outfile.write("# Mean acceptance fraction of MCMC run\n")

outfile.write("%10.5e "% config["M_p"]); outfile.write(" # Planet mass in Earth masses\n")
outfile.write("%10.5e "% config["T_surface"]); outfile.write(" # Surface Temperature in K\n")
outfile.write("%10.5e "% result["P_penalty"]); outfile.write(" # Pressure in bar\n")
outfile.write("%10.5e "% massfrac_atm); outfile.write(" # Mass fraction atmosphere\n")
outfile.write("%10.5e "% massfrac_melt); outfile.write(" # Mass fraction melt\n")
outfile.write("%10.5e "% massfrac_metal); outfile.write(" # Mass fraction metal\n")

for key in result:
    if key == 'P_penalty':
        pass
    else:
        outfile.write("%10.5e " %result[key])
        outfile.write("")
        outfile.write(" # %s\n" %key)        

outfile.write("%10.5e " %DIW_actual); outfile.write(" # DIW condensed part of planet relative to IW\n")
outfile.write("%10.5e " %DIW_atm); outfile.write(" # DIW of atmosphere relative to IW\n")

for key in wtpcs:
    outfile.write("%10.5e " %wtpcs[key][0])
    outfile.write("")
    outfile.write(" # %s\n" %key)

outfile.write("%10.5e " %density_atm); outfile.write(" # Approximate density of atmosphere in g/cm^3\n")
outfile.write("%10.5e " %density_melt); outfile.write(" # Approximate density of mantle in g/cm^3\n")
outfile.write("%10.5e " %density_metal); outfile.write(" # Uncompressed density of metal in g/cm^3\n")
outfile.write("%10.5e " %density_pureFe); outfile.write(" # Uncompressed density of pure Fe in g/cm^3\n")
outfile.write("%10.5e " %density_deficit); outfile.write(" # Density deficit by comparison with pure Fe in %\n")
outfile.write("%10.5e " %density_deficit_experimental); outfile.write(" # Density deficit by experimental values in %\n")

for i in range(0,19):
    if i == 14:
        K14temp = K[i] / result["P_penalty"]
        outfile.write("%10.5e " %K14temp)
    else:
        outfile.write("%10.5e " %K[i])
    outfile.write(" # Model KEQ for reaction %d : %s \n" %((i+1),reaction_names[i+1]))

outfile.write('## Model uncertainties \n')

outfile.write("%10.5e " %errs["P_penalty"]); outfile.write(" # Uncertainty in pressure result \n")
outfile.write("%10.5e " %err_mf_atm); outfile.write(" # Uncertainty in atmosphere mass fraction \n")
outfile.write("%10.5e " %err_mf_melt); outfile.write(" # Uncertainty in melt mass fraction \n")
outfile.write("%10.5e " %err_mf_metal); outfile.write(" # Uncertainty in metal mass fraction \n")

for i in range(len(variable_keys)):
    key = variable_keys[i]
    if key == "P_penalty":
        pass
    else:
        outfile.write("%10.5e " %errs[key])
        outfile.write("")
        outfile.write(" # %s\n" %f"uncertainty in {key}")

outfile.write("%10.5e " %err_DIW_actual); outfile.write(" # uncertainty in DIW of mantle/core\n")
outfile.write("%10.5e " %err_DIW_atm); outfile.write(" # uncertainty in DIW of atmosphere\n")

for key in wtpcs:
    outfile.write("%10.5e " %wtpcs[key][1])
    outfile.write("")
    outfile.write(" # %s\n" %f"uncertainty in {key}")

outfile.write("%10.5e " %err_density_atm); outfile.write(" # Uncertainty in density of atmosphere\n")
outfile.write("%10.5e " %err_density_melt); outfile.write(" # Uncertainty in density of mantle\n")
outfile.write("%10.5e " %err_density_metal); outfile.write(" # Uncertainty in density of metal\n")
outfile.write("%10.5e " %err_density_deficit); outfile.write(" # Uncertainty in density deficit\n")

outfile.write("## Gelman-Rubin Statistics for variables \n")
for i in range(len(GR)):
    outfile.write("%10.5f " %GR[i])
    outfile.write(f" # Gelman-Rubin statistic for {variable_keys[i]}\n")

outfile.close()

## Appending config to output file
path_to_output = outputname
copy_config(path_to_config, path_to_output)

if n_iters_MCMC >= 300000:
    try:
        time_start_cornerplot = time.time()
        ## Creating Corner Plot
        plotlabels = ['MgO melt', 'SiO2 melt', 'MgSiO3 melt', 'FeO melt', 'FeSiO3 melt', 'Na2O melt', 'Na2SiO3 melt', 'H2 melt',
                'H2O melt' 'CO melt', 'CO2 melt', 'Fe metal', 'Si metal', 'O metal', 'H metal', 'H2 gas', 'CO gas', 'CO2 gas',
                'CH4 gas', 'O2 gas', 'H2O gas', 'Fe gas', 'Mg gas', 'SiO gas', 'Na gas', 'SiH4 gas', '#M atm', '#M melt',
                '#M metal', "P"]
        figure = corner.corner(samples_flat[:,:])
        plt.show()
        figure.savefig(f"{runID}_corner_all.png")
        plt.close()

        time_end_cornerplot = time.time()
    except:
        logging.warning("Corner plot could not be made due to insufficient datapoints.")
        print("Corner plot could not be made due to insufficient amount of datapoints.")
else:
    logging.info("Number of MCMC iterations insufficient to create valid contours in cornerplot.")

endtime = time.time()

runtime_total = endtime - starttime
runtime_DA = time_end_dual_annealing - time_start_dual_annealing
runtime_MCMC = time_end_MCMC - time_start_MCMC

logging.info(f"Model final runtime: {runtime_total} s")
logging.info(f"   Runtime Simulated Annealing: {runtime_DA} s")
logging.info(f"   Runtime MCMC search: {runtime_MCMC} s")


try:
    runtime_corner = time_end_cornerplot - time_start_cornerplot
    logging.info(f"   Runtime cornerplot: {runtime_corner}")
except:
    logging.warning("No runtime of cornerplot calculated")
    print("Runtime of plotting not calculated.")

with open(outputname, 'a') as file:
    file.write("\n")
    for i in range(len(variable_keys)):
        key = variable_keys[i]
        file.write(f"{percentile_spread_params[i]} # 50/16/84 percentile of {key} \n")

with open(outputname, 'a') as file:
    file.write(f'\n{runtime_total} # Total runtime in s\n')
    file.write(f'{runtime_DA} # Runtime for Dual Annealing search in s \n')
    file.write(f'{runtime_MCMC} # Runtime for MCMC search in s\n')
    try:
        file.write(f'{runtime_corner} # Runtime for creating cornerplot in s\n')
    except:
        file.write(f'{0} # Runtime for creating cornerplot in s\n')

logging.info("\n   END   \n")
print(f"\nEnd.\n")