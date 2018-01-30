import sys
import numpy as np
import psi4
import time
import operator
from pathlib import Path
from functools import wraps
from davidson import davidson_solver
from helper_cc import HelperCCEnergy, HelperCCHbar, HelperCCEom


def redirect_stdout(dest, append=False):
    """Decorator to send print statements directed to sys.stdout to dest
    (optionally appending output) effecting every print in the decorated
    function
    """
    mode = 'w'
    if append:
        mode = 'a'

    redir_path = Path(dest)

    def redirect_decorator(function):
        @wraps(function)
        def call(*args, **kwargs):
            stdout_backup = sys.stdout
            with redir_path.open(mode) as redir_stream:
                sys.stdout = redir_stream
                result = function(*args, **kwargs)
                sys.stdout = stdout_backup
            return result

        return call

    return redirect_decorator


class Tee(object):
    """Simple class to accept write calls the same way a file-like object would,
    but redirecting them to multiple places

    """

    class writer(object):
        def __init__(self, path, mode):
            self.f = path.open(mode)
            self.stdout = sys.stdout

        def write(self, value):
            self.f.write(value)
            self.stdout.write(value)
            self.f.flush()
            self.stdout.flush()

        def close(self):
            self.f.close()

        def flush(self):
            self.stdout.flush()

    def __init__(self, dest_fn, mode):
        self.mode = mode
        self.path = Path(dest_fn)
        self.proxy = None
        self.stdout_save = sys.stdout

    def __enter__(self):
        assert (not self.proxy)
        self.stdout_save = sys.stdout
        self.proxy = Tee.writer(self.path, self.mode)
        sys.stdout = self.proxy

    def __exit__(self, *args):
        assert (self.proxy)
        self.proxy.close()
        self.proxy = None
        sys.stdout = self.stdout_save


def echo_stdout(dest, append=False):
    """Decorator to cause sys.stdout to be written to a file **and** stdout.
    Similar function to the `tee` command line utility. Note composing
    redirect_stdout and echo_stdout can be used to echo prints to two files
    simultaneously.
    """
    mode = 'w'
    if append:
        mode = 'a'

    def echo_decorator(function):
        @wraps(function)
        def call(*args, **kwargs):
            with Tee(dest, mode):
                result = function(*args, **kwargs)
            return result

        return call

    return echo_decorator


@echo_stdout('alldavidson.stdout', True)
def testing_solver_entrypoint(*args, **kwargs):
    return davidson_solver(*args, **kwargs)


class TestCase(object):
    """Generic test case base class defines interface for customized tests"""

    def __init__(self,
                 name,
                 guess,
                 no_eigs=1,
                 e_conv=1.0E-8,
                 r_conv=None,
                 max_vecs=100,
                 maxiter=100):
        self.name = name
        self.guess = guess
        self.no_eigs = no_eigs
        self.e_conv = e_conv
        self.dv_args = {
            'no_eigs': no_eigs,
            'e_conv': e_conv,
            'r_conv': r_conv,
            'max_vecs_per_root': max_vecs // no_eigs,
            'maxiter': maxiter
        }
        solver_redirect = redirect_stdout(self.name + '.solver.stdout', False)
        self.solver_func = solver_redirect(testing_solver_entrypoint)
        self.checks = {}

    def preconditioner(self, r_vector, index, approx_eigval):
        """Subclasses must define, preconditioning function"""
        raise Exception('Not Implemented')

    def ax_function(self, vector):
        """Subclasses must define, return target matrix time *vector*"""
        raise Exception('Not Implemented')

    def get_expected(self):
        """Subclasses must define, returns a tuple
            (expected eigenvalues, expected eigenvectors)

        """
        raise Exception('Not Implemented')

    def _success(self, label):
        """Generates a success message with test name and a label"""
        print("    {:.<103} ✔ PASS ✔".format(label))

    def _fail(self, label):
        """Generates a failure message with test name and a label"""
        print("    {:.<103} ✘ FAIL ✘".format(label))

    def _describe_failure(self, test, opname, expected):
        print("    {:>103}".format("{} evaluates false".format(
            "{}({},{})".format(opname, test, expected))))


    def compare_values(self, expected, computed):
        n_exp = len(expected)
        n_comp = len(computed)
        self.check("Correct # of Eigenvalues", n_comp, n_exp,
                op=operator.eq)
        for i in range(n_exp):
            delta = abs(expected[i] - computed[i])
            self.check("Eigenvalue {} matches".format(i),
                    delta, self.e_conv, op = operator.lt)

    def compare_vectors(self, expected, computed):
        tol = self.dv_args.get('r_conv')
        if tol == None:
            tol = self.dv_args.get('e_conv') * 100
        n_exp = expected.shape[-1]
        n_comp = computed.shape[-1]
        self.check("Correct # of Eigenvectors", n_comp, n_exp,
                op=operator.eq)
        for i in range(n_exp):
            expected_norm = np.linalg.norm(expected[:, i])
            computed_norm = np.linalg.norm(computed[:, i])
            delta = abs(expected_norm - computed_norm)
            self.check("Eigenvector {} matches".format(i),
                    delta, tol, op=operator.lt)

    def check(self, label, t_val, e_val, op = operator.eq):
        """
        Given a test aspect called label, and a condition, indicating pass
        (true)/ fail(false)). Print a success/failure message and save the state
        of the label in a dictionary. Also returns condition for use
        """
        test_label = "[<{}>//{}]".format(self.name, label)
        condition = op(t_val, e_val)
        if condition:
            self._success(test_label)
        else:
            self._fail(test_label)
            self._describe_failure(t_val, op.__name__, e_val)
        self.checks[label] = condition
        return condition

    @echo_stdout('test_report.stdout', append=True)
    def run(self, check_vectors=True):
        """Generic test case runner function, subclasses should not override.
            calls :py:func:`davidson.davison_solver` using subclasses defined
            :py:func:`ax_function` and :py:func:`preconditioner` and compareds
            returned values with the output of subclasses defined
            :py:func:`get_expected`. Prints checks (reporting all) then will
            cause an system exit if any fail.
        """
        print("=" * 120)
        print("{:=^120}".format(self.name))
        print("=" * 120)
        t_start = time.time()
        computed_vals, computed_vecs = self.solver_func(
            self.ax_function, self.preconditioner, self.guess, **self.dv_args)
        time_taken = time.time() - t_start
        print("    time: {:.>100}s".format(" {:5.2f}".format(time_taken)))
        self.check("Solver Returned Solutions",
                isinstance(computed_vals, np.ndarray),
                isinstance(computed_vecs, np.ndarray),
                op =operator.and_)
        expected_vals, expected_vecs = self.get_expected()
        self.compare_values(expected_vals, computed_vals)
        if check_vectors:
            self.compare_vectors(expected_vecs, computed_vecs)


class NumpyRandomTestCase(TestCase):
    """Generates a random numpy array, and allows it to pretend it is a
    complicated object that possesses ax_function and preconditioner function

    """
    spread_min = -0.5
    spread_max = 0.5

    def __init__(self, matrix_dim, sparsity_factor, no_eigs = 1, max_vecs = 100, **dv_args):
        """Creates A'(*matrix_dim* x *matrix_dim*) a square nump array of random
        numbers. Multiplies it by *sparsity_factor*, makes it symmetric, and
        sets the diagonal as some other array of *matrix_dim* random values.
        This results in a randomized diagonally dominant matrix that can be used
        to test the Davidson solver.

        """
        name = "Random_{dim}x{dim}_spf_{spf}_{nr}_roots".format(
            dim=matrix_dim, spf=sparsity_factor, nr=no_eigs)
        self.diag = (self.spread_max - self.spread_min
                     ) * np.random.rand(matrix_dim) + self.spread_min
        guess_index = self.diag.argsort()[:no_eigs * 6]
        guess = np.eye(matrix_dim)[:, guess_index]
        self.matrix = np.random.rand(matrix_dim,
                                     matrix_dim) * 1.0 / sparsity_factor
        self.matrix = (self.matrix + self.matrix.T) / 2.0
        self.matrix[np.diag_indices(matrix_dim)] = self.diag
        super(NumpyRandomTestCase, self).__init__(name, guess, no_eigs,
                                                  **dv_args)

    def preconditioner(self, r_vector, index, approx_eigval):
        """Diagonal preconditioner using the stored diagonal vals"""
        return r_vector / (approx_eigval - self.diag[index])

    def ax_function(self, vector):
        """Function handle for doting matrix * vector"""
        return np.dot(self.matrix, vector)

    def get_expected(self):
        """Explicitly diagonalize the matrix return sorted/truncated val,vec"""
        full_val, full_vec = np.linalg.eig(self.matrix)
        idx = full_val.argsort()[:self.no_eigs]
        return full_val[idx], full_vec[:, idx]


class EOMCCSDComparePsi4TestCase(TestCase):
    """More realistic Davidson test case using the EOMCC code from psi4numpy"""

    memory = '2 GB'
    nthread = 2

    def __init__(self, molecule_str, molecule_name, basis, no_eigs=1, max_vecs=100, **dv_args):
        name = "EOMCCSD_{mol}_{bas}_{nr}_roots".format(
            mol=molecule_name, bas=basis, nr=no_eigs)
        psi4.set_memory(self.memory)
        psi4.set_num_threads(self.nthread)
        psi4.core.set_output_file(name + '.psi4.output.dat', False)
        self.mol = psi4.geometry(molecule_str)
        psi4.set_options({'basis': basis, 'roots_per_irrep': [no_eigs]})
        ccsd = HelperCCEnergy(self.mol)
        ccsd.compute_energy()
        cchbar = HelperCCHbar(ccsd)
        self.cceom = HelperCCEom(ccsd, cchbar)
        self.D = np.hstack(
            (self.cceom.Dia.flatten(), self.cceom.Dijab.flatten()))
        guess = self.CIS(no_eigs * 6)
        super(EOMCCSDComparePsi4TestCase, self).__init__(
            name, guess, no_eigs = no_eigs, max_vecs = max_vecs, **dv_args)

    def preconditioner(self, r_vector, index, approx_eigval):
        return r_vector / (approx_eigval - self.D[index])

    def CIS(self, nroot):
        H_cis = np.einsum('ab,ij->iajb',
                          self.cceom.get_F('vv'), np.eye(self.cceom.ndocc))
        H_cis -= np.einsum('ij,ab->iajb',
                           self.cceom.get_F('oo'), np.eye(self.cceom.nvir))
        H_cis += self.cceom.get_L('ovov')
        H_cis.shape = (self.cceom.nsingles, self.cceom.nsingles)
        cis_vals, cis_vecs = np.linalg.eig(H_cis)
        idx = cis_vals.argsort()[:nroot]
        guesses = np.zeros((self.cceom.nsingles + self.cceom.ndoubles, nroot))
        guesses[:self.cceom.nsingles, :] += cis_vecs[:, idx]
        return guesses

    def ax_function(self, vector):
        V1 = vector[:self.cceom.nsingles].reshape(self.cceom.ia_shape)
        V2 = vector[self.cceom.nsingles:].reshape(self.cceom.ijab_shape)
        S1 = self.cceom.build_sigma1(V1, V2)
        S2 = self.cceom.build_sigma2(V1, V2)
        return np.hstack((S1.flatten(), S2.flatten()))

    def get_expected(self):
        psi4.energy('eom-ccsd', molecule=self.mol)
        psi4_ex_e = []
        # fill this up with None and ignore it in checks
        psi4_vectors = []
        for i in range(self.no_eigs):
            var_str = "CC ROOT {} CORRELATION ENERGY".format(i + 1)
            e_ex = psi4.core.get_variable(var_str)
            psi4_ex_e.append(e_ex)
            psi4_vectors.append(None)
        return psi4_ex_e, psi4_vectors

    def run(self, check_vectors=False):
        """Work around for arg based switching"""
        super(EOMCCSDComparePsi4TestCase, self).run(check_vectors)
        # need to make sure this is done, some files aren't cleaned up properly
        # without it
        psi4.core.clean()


class EOMCCSDCompareFullDiagTestCase(EOMCCSDComparePsi4TestCase):
    """**Very Expensive, using sigma vector function to build up the whole
    matrix and then explicitly diagonalize it, useful only to check vectors

    """

    def __init__(self, *args, **dv_args):
        super(EOMCCSDCompareFullDiagTestCase, self).__init__(*args, **dv_args)

    def get_expected(self):
        I_hbar_size = np.eye(self.cceom.nsingles + self.cceom.ndoubles)
        Hbar = np.zeros_like(I_hbar_size)
        for i in range(self.cceom.nsingles + self.cceom.ndoubles):
            Hbar[:, i] = self.ax_function(I_hbar_size[:, i])
        dirty_hbar_eval, dirty_hbar_evec = np.linalg.eig(Hbar)
        # full hbar built this way is over determined so we have to remove zeros
        nonzero_idx = np.where(np.abs(dirty_hbar_eval) > 1.0E-12)
        full_hbar_eval = dirty_hbar_eval[nonzero_idx]
        full_hbar_evec = dirty_hbar_evec[:, nonzero_idx]
        idx = full_hbar_eval.argsort()[:self.no_eigs]
        ret_eval = full_hbar_eval[idx].real()
        ret_evec = full_hbar_evec[:, idx].real()
        return ret_eval, ret_evec

    def run(self, check_vectors = True):
        super(EOMCCSDCompareFullDiagTestCase, self).run(check_vectors)
        # No need to run "clean" here since we call the superfunc which does
        # that for us
