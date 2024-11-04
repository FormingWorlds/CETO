import argparse
import numpy as np

parser=argparse.ArgumentParser("Argument parser for running exoplanet model over grid of values")
parser.add_argument("variable",type=list)
parser.add_argument("grid",type=np.ndarray)
args=parser.parse_args()

if len(args.variables) != args.grid.ndim:
    print("number of variable parameters does not equal dimensions of grid")

