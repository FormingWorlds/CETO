import numpy as np
import numpy.testing as npt
from pathlib import Path
from constants import *

from thermodynamics import shomate, get_Gibbs, calculate_GRT
from activity import get_activity
from model import optimisationfunction_initial
from utilities import *

## Reading model input from config file
sourcedir = Path(__file__).parent
path_to_config = sourcedir / 'defaultconfig.txt'
D = readconfig(path_to_config)

Fs, Gs = optimisationfunction_initial(D)

testsdir = Path(__file__).parent.parent / 'tests'
path_to_test = testsdir / 'testvalues_optimisation.txt'
Fs_expected = np.genfromtxt(path_to_test)

faults = []
print("="*53)
print("|{:^25}|{:^25}|".format('Test','Expected'))
print("-"*53)
for i in range(len(Fs)):
    if Fs[i] != 0 and Fs_expected[i] !=0:
        if np.abs((Fs[i]-Fs_expected[i])/Fs_expected[i]) >= 0.00001:
            print("|{:^25}|{:^25}|*".format(Fs[i],Fs_expected[i]))
            faults.append(i+1)
        else:
            print("|{:^25}|{:^25}|".format(Fs[i],Fs_expected[i]))
    else:
        print("|{:^25}|{:^25}|".format(Fs[i],Fs_expected[i]))

print('-'*53)

print(f"Test does not pass on:")
for i in range(len(faults)):
    print(f"F{faults[i]}")
