#!/usr/bin/env python3

import argparse
import numpy as np
import os
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser("Argument parser for running exoplanet model over grid of values")
parser.add_argument("-variablekey",type=str)
parser.add_argument("-values", nargs='+', required=True)
parser.add_argument("--modelversion",nargs='?',
                    default="Exoplanet_atmosphere_model_vMCMC_SiH4_2024_xB_maybe_faster.py",
                    type=str)
parser.add_argument("--outputdir",nargs='?',
                    default="./results",
                    type=str)
args=parser.parse_args()

try:
    myfile = Path(f"{args.outputdir}")
    my_abs_path = myfile.resolve(strict=True)
except FileNotFoundError:
    print("specified outputdirectory does not exist, creating dir")
    os.mkdir(myfile)
else:
    pass

for i in range(len(args.values)):
    x = args.values[i]
    print(f"CETOgrid.py: Script in loop {i+1}, running model for {args.variablekey} = {x}")
    subprocess.run([f'python3 {args.modelversion} \
                     --key {args.variablekey} --value {x}'], shell=True)
    
    
    resultdirectory = Path(f"{args.outputdir}/result_{args.variablekey}_{x}")
    os.mkdir(resultdirectory)

    source_result = f"{str(Path.cwd())}/output_summary_atm_SiH4_xB.txt"
    source_G_files = f"{str(Path.cwd())}/G*_RT.txt"
    destination_result = f"{resultdirectory}/result_{args.variablekey}_{x}.txt"
    destination_G_files = f"{resultdirectory}"
    subprocess.run([f'mv {source_result} {destination_result}'], shell=True)
    subprocess.run([f'mv {source_G_files} {destination_G_files}'], shell=True)