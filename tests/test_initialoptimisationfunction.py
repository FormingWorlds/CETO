import numpy as np
import numpy.testing as npt
from source.utilities import *
from source.model import optimisationfunction_initial

## Reading model input from config file
projectdir = Path(__file__).parent
path_to_config = projectdir / 'defaultconfig.txt'
D = readconfig(path_to_config)

nSi = moles_in_system('Si', D)
nMg = moles_in_system('Mg', D)
nNa = moles_in_system('Na', D)
nFe = moles_in_system('Fe', D)
nC = moles_in_system('C', D)
nH = moles_in_system('H', D)
nO = moles_in_system('O', D)

moles_initial = {'Si' : nSi,
                 'Mg' : nMg,
                 'Fe' : nFe,
                 'Na' : nNa,
                 'H'  : nH,
                 'O'  : nO,
                 'C'  : nC}

## Read array of computed values from Young model as test
path_to_expected = projectdir / 'testvalues_initialoptimisation.txt'

def test_optimisationfunction():
    test_input = optimisationfunction_initial(D, moles_initial)
    expected_result = np.genfromtxt(path_to_expected)
    npt.assert_array_almost_equal(test_input, expected_result, decimal=6)




