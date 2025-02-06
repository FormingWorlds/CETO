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
D2 = readconfig(path_to_config)

D2["bool_nonideal_mixing"] = False

# activitylist = ["lngSi", "lngO", "lngH2", "lngH2O_melt", "lngH_metal", "xB"]
# activity_nonideal = get_activity(D, get_all=True)
# activity_nonideal_notall = get_activity(D, get_all=False)
# activity_ideal = get_activity(D2)

# print("Activities for only ideal mixing:")
# for i in range(len(activity_ideal)):
#     print(f"{activitylist[i]} = {activity_ideal[i]}")

# print("Activities for non-ideal mixing, not all equations:")
# for i in range(len(activity_ideal)):
#     print(f"{activitylist[i]} = {activity_nonideal_notall[i]}")

# print("Activities for non-ideal mixing, all equations:")
# for i in range(len(activity_ideal)):
#     print(f"{activitylist[i]} = {activity_nonideal[i]}")

Fs = optimisationfunction_initial(D)
F_penalties = sigmoidal_penalty(Fs, 0, 5, 1, 10000)

print(F_penalties)







