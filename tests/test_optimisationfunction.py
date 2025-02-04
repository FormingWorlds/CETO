import numpy as np
import numpy.testing as npt
from source.utilities import *
from source.model import optimisationfunction_initial

## Reading model input from config file
projectdir = Path(__file__).parent
path_to_config = projectdir / 'defaultconfig.txt'
D = readconfig(path_to_config)

## Read array of computed values from Young model as test
path_to_expected = projectdir / 'testvalues_optimisation.txt'

def test_optimisationfunction():
    test_input = optimisationfunction_initial(D)
    expected_result = np.genfromtxt(path_to_expected)
    npt.assert_array_almost_equal(test_input, expected_result, decimal=6)




