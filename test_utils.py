import sys
import numpy as np
import psi4
from davidson import davidson_solver
from helper_cc import HelperCCEnergy, HelperCCHbar, HelperCCEom

class TestCase(object):
    """Generic test case base class defines interface for customized tests"""
    def __init__(self,
                 name,
                 guess,
                 no_eigs=1,
                 e_conv=1.0E-8,
                 r_conv=None,
                 max_vecs_per_root=10,
                 maxiter=100):
        self.name = name
        self.guess = guess
        self.no_eigs = no_eigs
        self.e_conv = e_conv
        self.dv_args = {
            'no_eigs': no_eigs,
            'e_conv': e_conv,
            'r_conv': r_conv,
            'max_vecs_per_root': max_vecs_per_root,
            'maxiter': maxiter
        }

    def preconditioner(self, r_vector, index, aprox_eigval):
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
        return '\t{:.<100} ✔ PASS ✔'.format("[{}/{}]".format(self.name, label))

    def _fail(self, label):
        """Generates a failure message with test name and a label"""
        return '\t{:.<100} ✘ FAIL ✘'.format("[{}/{}]".format(self.name, label))

    def compare_values(self, expected, computed, index):
        """Function to compare *expected* eigenvalue to *computed* eigenvalue
            creates a label "eigenvalue *index*" and passes it to
            :py:func:`TestCase._success` or :py:func:`TestCase._fail` as
            appropriate. Returns `True` or `False` matching success/failure

        """
        if abs(expected - computed) < self.e_conv:
            print(self._success("eigenvalue {}").format(index))
            return True
        else:
            print(self._fail("eigenvalue {}".format(index)))
            return False

    def compare_vectors(self, expected, computed, index):
        """Function to compare *expected* eigenvector to *computed* eigenvector
            creates a label "eigenvector *index*" and passes it to
            :py:func:`TestCase._success` or :py:func:`TestCase._fail` as
            appropriate. Returns `True` or `False` matching success/failure

        """
        if np.allclose(expected, computed):
            print(self._success("eigenvector {}".format(index)))
            return True
        else:
            print(self._fail("eigenvector {}".format(index)))
            return False

    def run(self, check_vectors=True):
        """Generic test case runner function, subclasses should not override.
            calls :py:func:`davidson.davison_solver` using subclasses defined
            :py:func:`ax_function` and :py:func:`preconditioner` and compareds
            returned values with the output of subclasses defined
            :py:func:`get_expected`. Prints checks (reporting all) then will
            cause an system exit if any fail.
        """
        print("Running {}...".format(self.name))
        computed_vals, computed_vecs = davidson_solver(
            self.ax_function, self.preconditioner, self.guess, **self.dv_args)
        expected_vals, expected_vecs = self.expected_returns()
        if len(computed_vals) == self.no_eigs:
            print(self._success("# eigenvalues"))
        else:
            print(self._fail("# eigenvalues"))
            sys.exit(1)
        if check_vectors:
            if computed_vecs.shape[-1] == self.no_eigs:
                print(self._success("# eigenvectors"))
            else:
                print(self._fail("# eigenvectors"))
                sys.exit(1)
        check = True
        for i,(expected_val,computed_val) in enumerate(zip(expected_vals, computed_vals)):
            check = check and self.compare_values(expected_val, computed_val, i)
            if check_vectors:
                expected_vec = expected_vecs[:, i]
                computed_vec = computed_vecs[:, i]
                check = check and self.compare_vector(expected_vec, computed_vec, i)
        if not check:
            print("test case {} Failed one or more checks ".format(self.name))
            sys.exit(1)

class NumpyRandomTestCase(TestCase):
    """Generates a random numpy array, and allows it to pretend it is a
    complicated object that possesses ax_function and preconditioner function

    """
    def __init__(self, matrix_dim, sparsity_factor, no_eigs, **dv_args):
        """Creates A'(*matrix_dim* x *matrix_dim*) a square nump array of random
        numbers. Multiplies it by *sparsity_factor*, makes it symmetric, and
        sets the diagonal as some other array of *matrix_dim* random values.
        This results in a randomized diagonally dominant matrix that can be used
        to test the Davidson solver.

        """
        name = "Random {dim}x{dim}(sparsity={spf}) {nr} roots".format(
                dim=matrix_dim, spf=sparsity_factor, nr=no_eigs)
        guess = np.eye(matrix_dim)[:,no_eigs]
        self.diag = np.random.rand(matrix_dim)
        self.matrix = np.random.rand(matrix_dim,matrix_dim) * sparsity_factor
        self.matrix = (self.matrix + self.matrix.T) / 2.0
        self.matrix[np.diag_indices(matrix_dim)] = self.diag
        super(NumpyRandomTestCase, self).__init__(name, guess, no_eigs,
                **dv_args)

    def preconditioner(self, r_vector, index, aprox_eigval):
        """Diagonal preconditioner using the stored diagonal vals"""
        return r_vector / (approx_eigval - self.diag[index])

    def ax_function(self, vector):
        """Function handle for doting matrix * vector"""
        return np.dot(self.matrix, vector)

    def get_expected(self):
        """Explicitly diagonalize the matrix return sorted/truncated val,vec"""
        full_val, full_vec = np.linalg.eig(self.matrix)
        idx = full_val.argsort()[:self.no_eigs]
        return full_vall[idx], full_vec[:,idx]

class EOMCCSDComparePsi4TestCase(TestCase):
    """More realistic Davidson test case using the EOMCC code from psi4numpy"""
    def __init__(self,molecule_str, molecule_name, basis, no_eigs, **dv_args):
        name = "{mol}/{bas}[EOMCCSD Test checking roots vs psi4] {nr} roots".format(mol=molecule_name,
                bas=basis, nr=no_eigs)
        psi4.set_memory('2 GB')
        psi4.set_num_threads(2)
        psi4.core.set_output_file('output.dat', False)
        self.mol = psi4.geometry(molecule_str)
        psi4.set_options({'basis':basis, 'roots_per_irrep':[no_eigs]})
        ccsd = HelperCCEnergy(mol)
        ccsd.compute_energy()
        cchbar = HelperCCHbar(ccsd)
        self.cceom = HelperCCEom(ccsd, cchbar)
        self.D = np.hstack((self.cceom.Dia.flatten(), self.cceom.Dijab.flatten()))
        B_idx = self.D[:self.cceom.nsingles].argsort(:no_eigs)
        guess = np.eye(self.cceom.nsingles + self.cceom.ndoubles)[:, B_idx]
        super(EOMCCSDTestCase, self).__init__(name, guess, no_eigs, **dv_args)

    def preconditioner(self, r_vector, index, approx_eigval):
        return r_vector / (approx_eigval - self.D[index])

    def ax_function(self, vector):
        V1 = vector[:self.cceom.nsingles].reshape(self.cceom.ia_shape)
        V2 = vector[self.cceom.nsingles:].reshape(self.cceom.ijab_shape)
        S1 = self.cceom.build_sigma1(V1, V2)
        S2 = self.cceom.build_sigma2(V1, V2)
        return np.hstack((S1.flatten(), S2.flatten()))

    def get_expected(self):
        psi4.energy('eom-ccsd',molecule=self.mol)
        psi4_ex_e = []
        # fill this up with None and ignore it in checks
        psi4_vectors = []
        for i in range(self.no_eigs):
            var_str = "CC ROOT {} CORRELATION ENERGY".format(i + 1)
            e_ex = psi4.core.get_variable(var_str)
            psi4_ex_e.append(e_ex)
            psi4_vectors.append(None)
        return psi4_ex_e, psi4_vectors

class EOMCCSDCompareFullDiagTestCase(TestCase):
    """**Very Expensive, using sigma vector function to build up the whole
    matrix and then explicitly diagonalize it, useful only to check vectors

    """
    def __init__(self, molecule_str, molecule_name, basis, no_eigs, **dv_args):
        name = "{mol}/{bas}[EOMCCSD Test explicit diagonalization] {nr} roots".format(mol=molecule_name,
                bas=basis, nr=no_eigs)
        psi4.set_memory('2 GB')
        psi4.set_num_threads(2)
        psi4.core.set_output_file('output.dat', False)
        self.mol = psi4.geometry(molecule_str)
        psi4.set_options({'basis':basis, 'roots_per_irrep':[no_eigs]})
        ccsd = HelperCCEnergy(mol)
        ccsd.compute_energy()
        cchbar = HelperCCHbar(ccsd)
        self.cceom = HelperCCEom(ccsd, cchbar)
        self.D = np.hstack((self.cceom.Dia.flatten(), self.cceom.Dijab.flatten()))
        B_idx = self.D[:self.cceom.nsingles].argsort(:no_eigs)
        guess = np.eye(self.cceom.nsingles + self.cceom.ndoubles)[:, B_idx]
        super(EOMCCSDTestCase, self).__init__(name, guess, no_eigs, **dv_args)

    def preconditioner(self, r_vector, index, approx_eigval):
        return r_vector / (approx_eigval - self.D[index])

    def ax_function(self, vector):
        V1 = vector[:self.cceom.nsingles].reshape(self.cceom.ia_shape)
        V2 = vector[self.cceom.nsingles:].reshape(self.cceom.ijab_shape)
        S1 = self.cceom.build_sigma1(V1, V2)
        S2 = self.cceom.build_sigma2(V1, V2)
        return np.hstack((S1.flatten(), S2.flatten()))

    def get_expected(self):
        I_hbar_size = np.eye(self.cceom.nsingles + self.cceom.ndoubles)
        Hbar = np.zeros_like(I_hbar_size)
        for i in range(self.cceom.nsingles + self.cceom.ndoubles):
              Hbar[:,i] = self.ax_function(I_hbar_size[:,i])
        dirty_hbar_eval, dirty_hbar_evec = np.linalg.eig(Hbar)
        # full hbar built this way is over determined so we have to remove zeros
        nonzero_idx = np.where(np.abs(dirty_hbar_eval) > 1.0E-12 )
        full_hbar_eval = dirty_hbar_eval[nonzero_idx]
        full_hbar_evec = dirty_hbar_evec[:, nonzero_idx]
        idx = full_hbar_eval.argsort()[:self.no_eigs]
        ret_eval = full_hbar_eval[idx].real()
        ret_evec = full_hbar_evec[:, idx].real()
        return ret_eval, ret_evec
