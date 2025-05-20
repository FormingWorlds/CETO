import numpy as np
import numpy.testing as npt
from copy import deepcopy
import pytest

from source.utilities import *
from source.activity import get_activity

projectdir = Path(__file__).parent
path_to_config = projectdir / 'defaultconfig.txt'
config_global = readconfig(path_to_config)

config_nonidealmixing = deepcopy(config_global)
config_idealmixing = deepcopy(config_global)
config_idealmixing['bool_ideal_mixing'] = True 

activities1 = [-2.0759083310201785, 1.539999998127, 0.0, 0.0, 0.0, 3.333333335555555e-10]                       # Activities for default config, not all activities returned (From Young code)
activities2 = [-2.07590833e+00, 1.54000000e+00, 1.50000000e+00, 1.49992783e+00, 7.60000022e-09, 3.33333334e-10] # Activities for default config, all activities returned (from Young code)
activities3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                                                                    # Activities for ideal mixing config (eg. all lng terms must be zero)

testdata = [
    (config_nonidealmixing, False, activities1),
    (config_nonidealmixing, True, activities2),
    (config_idealmixing, False, activities3)
]

@pytest.mark.parametrize("testconfig, bool, expected", testdata)

def test_activities(testconfig, bool, expected):
    testresult = get_activity(testconfig, testconfig, get_all=bool)
    npt.assert_array_almost_equal(testresult, expected)