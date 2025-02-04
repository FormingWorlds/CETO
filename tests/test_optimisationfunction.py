import numpy as np
import numpy.testing as npt
import pytest
from source.constants import (log_to_ln, R_gas)
from source.thermodynamics import get_Gibbs, calculate_GRT
from source.utilities import *
from source.model import optimisationfunction

## Reading model input from config file
projectdir = Path(__file__).parent
path_to_config = projectdir / 'defaultconfig.txt'
D = readconfig(path_to_config)

## Read array of computed values from Young model as test
path_to_expected = projectdir / 'testvalues_optimisation.txt'
expected = np.genfromtxt(path_to_expected)

def test_optimisationfunction(test, expected):
    npt.assert_array_almost_equal(optimisationfunction(D), expected)




