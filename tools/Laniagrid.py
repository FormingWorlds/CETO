#!/dataserver/users/formingworlds/lania/mscthesis/venv/bin/python

# Runs our version of the model for a grid of parameters

from __future__ import annotations

import argparse
import logging
import numpy as np
import os
from pathlib import Path
import subprocess
import time

## Create argument parser
parser = argparse.ArgumentParser("Argument parser for Lania's version of Young Exoplanet model, for grid searches")
parser.add_argument('-config', required=True, type=str)
parser.add_argument('-outputdir', required=True)
parser.add_argument('--logfile',nargs='?',default='Laniagrid.log')
parser.add_argument('--loglevel',nargs='?',default='INFO')
parser.add_argument('--runcounts',nargs='+',default=None)
parser.add_argument('--doplots',nargs='?',default=True)
args = parser.parse_args()

configpath = args.config
outputdir = args.outputdir
loglevel = args.loglevel
logname = args.logfile
runcounts = args.runcounts
doplots = args.doplots

try:
    parentdir = Path(__file__).parent.parent
    abspath_model = Path(parentdir / 'source/testscript.py').resolve(strict=True)
except:
    print(f"CRITICAL: Could not establish path to model scripts")
    exit()

## Create logging instance
logger = logging.getLogger(__name__)
numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError('Invalid log level: %s' % loglevel)

logging.basicConfig(filename=logname, filemode='w', encoding='utf-8', level=numeric_level,
                    format='%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.info("Starting run with following command line input: ")
for i in range(len(vars(args))):
    key = list(vars(args).keys())[i]
    logging.info(f"   {key}:   {vars(args)[key]}")

## Obtain 'default' config over which to run model
try:
    path_to_config = Path(f"{configpath}").resolve(strict=True)
    logging.info(f"Specified config file found: {path_to_config}")
except FileNotFoundError:
    try:
        path_to_config = Path(f"{Path(__file__).parent.parent}/source/{configpath}")
        logging.info(f"Specified config file found: {path_to_config}")
    except FileNotFoundError:
        logging.critical(f"Provided config not found: {args.config}")
        print(f"CRITICAL: could not find provided config: {configpath}")
        exit()

## Establish path to specified output directory
try:
    path_to_outputdir = Path(f"{outputdir}").resolve(strict=True)
    logging.info(f"Specified output directory found: {path_to_outputdir}")
except FileNotFoundError:
    logging.critical(f"Specified output directory does not exist: {args.outputdir}")
    print(f"Specified output directory does not exist. Create it and re-run.")
    exit()

## Establish parameter grids
variables = ["T_surface", "T_eq", "moles_atm", "moles_metal"]
arr_T_surface = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
arr_dT_eq = np.array([1000.0, 2000.0, 3000.0])
arr_moles_atm = np.array([1400.0, 7000.0, 14600.0, 32900.0])
arr_moles_metal = np.array([0, 1, 2, 3])

logging.info(f"Grid search over the following parameters: \n {variables[0]} : {arr_T_surface}\
             \n {variables[1]} : {arr_dT_eq}\n {variables[2]} : {arr_moles_atm}\n {variables[3]} : {arr_moles_metal}")

## Main loop:
logging.info("Starting grid search \n")
config = open(configpath, 'r')
config_lines = config.readlines()
config.close()

T_surface_index = [i for i in range(len(config_lines)) if '# Surface Temperature' in config_lines[i]][0]
T_eq_index = [i for i in range(len(config_lines)) if '# Core-mantle equilibration temperature' in config_lines[i]][0]
moles_atm_index = [i for i in range(len(config_lines)) if '# Moles atmosphere' in config_lines[i]][0]
moles_metal_index = [i for i in range(len(config_lines)) if '# Moles metal' in config_lines[i]][0]

runtimes = [] #will track model runtimes
skips = [] #contains indices of models to be skipped
count = 0

if runcounts != None:
    runcounts = [int(val) for val in runcounts]
    logging.info(f"Running model on the following runs: \n {runcounts}")

if len(skips) != 0:
    logging.warning(f"The following runs will be skipped because they are included in this file's skip list: \n {list(skips)}")

for h in range(len(arr_moles_metal)):
    moles_metal_value = arr_moles_metal[h]
    for i in range(len(arr_moles_atm)):
        moles_atm_value = arr_moles_atm[i]
        for j in range(len(arr_dT_eq)):
            dTeq_value = arr_dT_eq[j]
            for k in range(len(arr_T_surface)):
                T_surface_value = arr_T_surface[k]

                if count >= (int(len(arr_T_surface)*len(arr_dT_eq)*len(arr_moles_atm)*len(arr_moles_metal)))+1:
                    break
                else:
                    count += 1

                    if count in skips:
                        pass
                        logging.info(f"Run {count} is skipped; present in skip list")
                    elif runcounts != None and count not in runcounts:
                        pass
                        logging.info(f"Run {count} is skipped; not listed in specified target runs")

                    else:
                        logging.info(f"Grid in loop {count}")

                        config_lines[T_surface_index] = f"{T_surface_value} # Surface Temperature in Kelvin \n"
                        config_lines[T_eq_index] = f"{T_surface_value + dTeq_value} # Core-mantle equilibration temperature in Kelvin \n"
                        config_lines[moles_atm_index] = f"{moles_atm_value} # Moles atmosphere\n"
                        config_lines[moles_metal_index] = f"{moles_metal_value} # Moles metal core\n"

                        configname = f"config_run{count}.txt"

                        newconfig = open(configname, 'w')
                        for line in config_lines:
                            newconfig.write(line)
                        newconfig.close()

                        abspath_newconfig = str(Path(configname).resolve(strict=True))

                        logging.info(f"Created {configname} with T_surface={T_surface_value}, T_eq={T_surface_value+dTeq_value}, Moles atm={moles_atm_value}, Moles metal={moles_metal_value}")
                        logging.info(f"Running Lania model over {configname}")

                        ## Run the model!
                        runID = f"L_run{count}"
                        run_logname = f"L_run{count}.log"
                        subprocess.run(["","-input",abspath_newconfig,"--logname",run_logname,"--runID",runID,"--doplots",doplots], executable=abspath_model)

                        time.sleep(1.0)
                        logging.info("Finished running model. Summary: \n")

                        ## Perform checks on model output
                        try:
                            check = open(f'{runID}_output_summary.txt', 'r')
                            resultlines = check.readlines()
                            check.close()

                            statistics = []
                            model_runtimes = []

                            for line in resultlines[:4]: #this assumes the statistics to be the first four lines in the result
                                for i in range(len(line)):
                                    if line[i] == '#':
                                        try:
                                            val = float(line[:i])
                                            statistics.append(val)
                                        except:
                                            pass

                            chi2, user_seed, gen_seed, mean_a_f = statistics
                            
                            for line in resultlines[-5:]:
                                for i in range(len(line)):
                                    if line[i] == '#':
                                        try:
                                            rtime = float(line[:i])
                                            model_runtimes.append(rtime)
                                        except:
                                            pass

                            t_total, t_da, t_mcmc, t_corner = model_runtimes 
                            runtimes.append(t_total)

                            logging.debug(f"Run user seed: {user_seed}")
                            logging.debug(f"Run generated seed: {gen_seed}")

                            logging.info(f"Model runtime: {t_total} s\n  ... DA search time: {t_da} s\n  ... MCMC search time: {t_mcmc} s\n  ... Cornerplot time: {t_corner} s")

                            if chi2 > 2.0:
                                logging.warning(f"Run {count}: reduced chi2 = {chi2} exceeds threshold. Results may be unreliable.")
                            else:
                                logging.info(f"Model reduced chi2 = {chi2}")

                            if mean_a_f < 0.01:
                                logging.warning(f"Run {count}: mean acceptance fraction = {mean_a_f} below threshold. Results may be unreliable.")
                            elif mean_a_f > 0.5:
                                logging.warning(f"Run {count}: mean acceptance fraction = {mean_a_f} above threshold. Results may be unreliable.")
                            else:
                                logging.info(f"Run {count} mean acceptance fraction: {mean_a_f}")
                        except:
                            logging.warning(f"Logger failed to extract information on model statistics and runtimes.\nMake sure to inspect model output and re-run as necessary.")
                            
                        ## Move the results to output directory

                        outputfile = f'{runID}_output_summary.txt'
                        cornerplot = f'{runID}_corner_all.png'

                        rundirectory = Path(f"{outputdir}/run{count}")
                        os.mkdir(rundirectory)
                        logging.info(f"Created directory {rundirectory}")

                        plotsdirectory = Path(f"{outputdir}/run{count}/plots")
                        os.mkdir(plotsdirectory)
                        logging.info(f"Created directory {plotsdirectory}")

                        if doplots == 'True' or doplots == 'true':
                            catchall_hists = f'{runID}_*_hist.png' #wildcard character to take all histograms
                            try:
                                subprocess.run([f'mv {catchall_hists} {plotsdirectory}'], shell=True)
                                logging.info(f"Moved histograms to {plotsdirectory}")
                            except:
                                logging.warning(f"Could not move histograms to {plotsdirectory}")
                        else:
                            logging.info("No histograms were created.")


                        try:
                            subprocess.run([f'mv {configname} {rundirectory}'], shell=True)
                            logging.info(f"Moved {configname} to {rundirectory}")
                        except:
                            logging.warning(f"Could not move {configname} to {rundirectory}")

                        try:
                            subprocess.run([f'mv {outputfile} {rundirectory}'], shell=True)
                            logging.info(f"Moved {outputfile} to {rundirectory}")
                        except:
                            logging.warning(f"Could not move {outputfile} to {rundirectory}")

                        try:
                            subprocess.run([f'mv {cornerplot} {plotsdirectory}'], shell=True)
                            logging.info(f"Moved {cornerplot} to {plotsdirectory}")
                        except:
                            logging.warning(f"Could not move {cornerplot} to {plotsdirectory}")

                        try:
                            subprocess.run([f'mv {run_logname} {rundirectory}'], shell=True)
                            logging.info(f"Moved {run_logname} to {rundirectory}\n")
                        except:
                            logging.warning(f"Could not move {run_logname} to {rundirectory}\n")

if len(runcounts) != None:
    logging.info(f"Finished runnning {len(runcounts)} models.\n")

if len(runtimes) != None:
    logging.info(f"Mean runtime: {np.mean(np.array(runtimes))}")
                                       
logging.info("End.")


