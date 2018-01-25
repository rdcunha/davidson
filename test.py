import sys
import os
import argparse
import numpy as np
import psi4
import json
from datetime import datetime
from pathlib import Path
from test_utils import (NumpyRandomTestCase, EOMCCSDComparePsi4TestCase,
                        EOMCCSDCompareFullDiagTestCase)
parser = argparse.ArgumentParser()
# Use the "full matrix" diagonalization to check eigenvalues (note expensive)
parser.add_argument('--EOM_full', action='store_true', default=False)
# Delete this after debugging
parser.add_argument('--justone', action='store_true', default=False)
# If you run a larger test and want to bump up the memory for psi4
parser.add_argument('--psi4_mem', nargs='?', type=int, default=2)
# If you want to set nthread tailored to your system
parser.add_argument('--psi4_nthread', nargs='?', type=int, default=2)
# Controls the interval min for RNG on the diagonal of Random Test matrix
parser.add_argument(
    '--random_dspread_min', nargs='?', type=float, default=-0.5)
# Controls the interval max for RNG on the diagonal of Random Test matrix
parser.add_argument('--random_dspread_max', nargs='?', type=float, default=0.5)
parser.add_argument('--skip_eom', action='store_true', default=False)
parser.add_argument('--out_dir', nargs='?', type=str, default='test_'+datetime.now().isoformat())

h2o_molstr = """
O
H 1 1.1
H 1 1.1 2 104
noreorient
symmetry c1
"""

h2o2_molstr = """
O
O 1 1.39
H 1 0.94 2 102.3
H 2 0.95 1 102.3 3 -50.0
noreorient
symmetry c1
"""
def main(args):
    if args.justone:
        print("Running a single small test to check framework")
        one_test = NumpyRandomTestCase(
            1000, 1000, 2, maxiter=300, e_conv=1.0e-5, max_vecs=40)
        one_test.run()
        print("Done exiting")
        sys.exit(0)

    if args.EOM_full:
        EOMTestCase = EOMCCSDCompareFullDiagTestCase
    else:
        EOMTestCase = EOMCCSDComparePsi4TestCase

    EOMTestCase.nthread = args.psi4_nthread
    EOMTestCase.memory = '{} GB'.format(args.psi4_mem)

    tests = [
        # each set is [testcasetype (class), args(tuple), kwargs(dict)]
        # EOMCC test cases a more realistic test of the solvers use in the wild
        [ EOMTestCase, (h2o_molstr, 'water', 'cc-pvdz'), dict(no_eigs=1, max_vecs=60, maxiter=300, e_conv=1.0e-6, r_conv=1.0e-5)],
        [ EOMTestCase, (h2o_molstr, 'water', 'cc-pvdz'), dict(no_eigs=3, max_vecs=60, maxiter=300, e_conv=1.0e-6, r_conv=1.0e-5)],
        [ EOMTestCase, (h2o_molstr, 'water', 'aug-cc-pvdz'), dict(no_eigs=1, max_vecs=60, maxiter=300, e_conv=1.0e-6, r_conv=1.0e-5)],
        [ EOMTestCase, (h2o_molstr, 'water', '6-311G'), dict(no_eigs=3, max_vecs=60, maxiter=300, e_conv=1.0e-6, r_conv=1.0e-5)],
        [ EOMTestCase, (h2o2_molstr, 'hydrogen-peroxide','cc-pvdz'),dict(no_eigs=2, max_vecs=100, maxiter=300, e_conv=1.0e-6)],
        [ EOMTestCase, (h2o2_molstr, 'hydrogen-peroxide','cc-pvdz'),dict(no_eigs=4, max_vecs=100, maxiter=300, e_conv=1.0e-6)],
        [ EOMTestCase, (h2o2_molstr, 'hydrogen-peroxide','6-31G*'),dict(no_eigs=2, max_vecs=100, maxiter=300, e_conv=1.0e-6)],
        [ EOMTestCase, (h2o2_molstr, 'hydrogen-peroxide','6-31G'),dict(no_eigs=4, max_vecs=100, maxiter=300, e_conv=1.0e-6)],
        # "smaller" tests 2000-5000 dimension with varying sparsity (1-5 roots)
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 1000), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=300, e_conv=1.0e-6) ],
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 100), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=300, e_conv=1.0e-6) ],
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 10), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=300, e_conv=1.0e-6) ],
        # Stress test, no sparsity factor means non-ideal problem, but given
        # enough iters it *should* work
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 1), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=500, e_conv=1.0e-6) ],
        # Stress test, very tight convergence, see how many iters needed
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 1000), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=3000, e_conv=1.0e-6) ],
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 100), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=3000, e_conv=1.0e-6) ],
        [ NumpyRandomTestCase,(np.random.randint(2000, 5000), 10), dict(no_eigs=np.random.randint(3, 5), max_vecs=100, maxiter=3000, e_conv=1.0e-6) ],
        # "larger" tests 6000-10000 dimension with varying sparsity (1-5 roots)
        [ NumpyRandomTestCase,(np.random.randint(6000, 10000), 10), dict(no_eigs=np.random.randint(1, 3), max_vecs=100, maxiter=300, e_conv=1.0e-6) ],
        [ NumpyRandomTestCase,(np.random.randint(6000, 10000), 100), dict(no_eigs=np.random.randint(1, 3), max_vecs=100, maxiter=300, e_conv=1.0e-6) ],
        [ NumpyRandomTestCase,(np.random.randint(6000, 10000), 1000), dict(no_eigs=np.random.randint(1, 3), max_vecs=100, maxiter=300, e_conv=1.0e-6) ],
        ]  # yapf: disable
    testing_results = []
    for test_class, case_args, case_kwargs in tests:
        if (test_class == EOMTestCase) and args.skip_eom:
            continue
        else:
            case = test_class(*case_args, **case_kwargs)
            case.run()
            testing_results.append(case.checks)
    with open('allresults.json','w') as f:
        json.dump(testing_results, f, indent=4)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    outpath = Path(args.out_dir)
    outpath.mkdir()
    old_dir = Path.cwd()
    os.chdir(str(outpath))
    main(args)
    os.chdir(str(old_dir))
