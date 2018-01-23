import sys
import argparse
from test_utils import (NumpyRandomTestCase, EOMCCSDComparePsi4TestCase,
                        EOMCCSDCompareFullDiagTestCase)
parser = argparse.ArgumentParser()
parser.add_argument('--EOM_full', action='store_true')
parser.add_argument('--sparse_factors',nargs='*', type=int, default=[10,100,1000])
parser.add_argument('--nrandom', nargs='?', type=int, defualt=3)
parser.add_argument('--random_size_min',nargs='?', type=int, default=1000)
parser.add_argument('--random_size_max', nargs='?', type =int, defualt=5000)

h2o_molstr = """
O
H 1 1.1
H 1 1.1 2 104
noreorient
symmetry c1
"""

Ne_molstr = """
Ne
symmetry c1
"""
if __name__ == "__main__":
    args = parser.parse_args(sys.argv)
    if args.EOM_full:
        EOMTestCase = EOMCCSDCompareFullDiagTestCase
    else:
        EOMTestCase = EOMCCSDComparePsi4TestCase

    tests = []
    random_sizes = [np.random.randint(args.random_size_min, args.random_size_max)
            for x in range(args.nrandom)]
    for r_size in random_sizes:
        for sparse_factor in args.sparse_factors:
            # look for a random number of roots between one and approx 1% of
            # matrix dim
            nroot = np.random.randint(1, r_size//100)
            tests.append(NumpyRandomTestCase(r_size, 1/sparse_factor, nroot))
    # add our EOM Test case
    nroot = np.random.randint(1,5)
    tests.append(EOMTestCase(h2o_molstr, 'water', 'cc-pvdz', nroot))
    nroot = np.random.randint(1,5)
    tests.append(EOMTestCase(Ne_molstr, 'Neon Atom', 'cc-pvdz', nroot))

    # evaluate all tests
    for tc in tests:
        tc.run()

