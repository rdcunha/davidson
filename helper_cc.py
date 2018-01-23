# A simple Psi4 script to compute CCSD linear response properties (spin-free) from
# RHF reference Scipy and numpy python modules are required
# Algorithms were taken directly from Daniel Crawford's programming website:
# http://github.com/CrawfordGroup/ProgrammingProjects
# Equations were spin-adapted using unitary group approach used in the molecular
# electronic structure theory.
# Special thanks to Dr. T Daniel Crawford for help in spin-adaptation of
# the equations and Lori Burns for integral help
#
# Created by: Ashutosh Kumar, Daniel G. A. Smith.
# Date: 5/17/2017
# License: GPL v3.0
#

import time
import numpy as np
import psi4

# N dimensional dot
# Like a mini DPD library
def ndot(input_string, op1, op2, prefactor=None):
    """
    No checks, if you get weird errors its up to you to debug.

    ndot('abcd,cdef->abef', arr1, arr2)
    """
    inp, output_ind = input_string.split('->')
    input_left, input_right = inp.split(',')

    size_dict = {}
    for s, size in zip(input_left, op1.shape):
        size_dict[s] = size
    for s, size in zip(input_right, op2.shape):
        size_dict[s] = size

    set_left = set(input_left)
    set_right = set(input_right)
    set_out = set(output_ind)

    idx_removed = (set_left | set_right) - set_out
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed

    # Tensordot axes
    left_pos, right_pos = (), ()
    for s in idx_removed:
        left_pos += (input_left.find(s),)
        right_pos += (input_right.find(s),)
    tdot_axes = (left_pos, right_pos)

    # Get result ordering
    tdot_result = input_left + input_right
    for s in idx_removed:
        tdot_result = tdot_result.replace(s, '')

    rs = len(idx_removed)
    dim_left, dim_right, dim_removed = 1, 1, 1
    for key, size in size_dict.items():
        if key in keep_left:
            dim_left *= size
        if key in keep_right:
            dim_right *= size
        if key in idx_removed:
            dim_removed *= size

    shape_result = tuple(size_dict[x] for x in tdot_result)
    used_einsum = False

    # Matrix multiply
    # No transpose needed
    if input_left[-rs:] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed),
                          op2.reshape(dim_removed, dim_right))

    # Transpose both
    elif input_left[:rs] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T,
                          op2.reshape(dim_right, dim_removed).T)

    # Transpose right
    elif input_left[-rs:] == input_right[-rs:]:
        new_view = np.dot(op1.reshape(dim_left, dim_removed),
                          op2.reshape(dim_right, dim_removed).T)

    # Tranpose left
    elif input_left[:rs] == input_right[:rs]:
        new_view = np.dot(op1.reshape(dim_removed, dim_left).T,
                          op2.reshape(dim_removed, dim_right))

    # If we have to transpose vector-matrix, einsum is faster
    elif (len(keep_left) == 0) or (len(keep_right) == 0):
        new_view = np.einsum(input_string, op1, op2)
        used_einsum = True

    else:
        new_view = np.tensordot(op1, op2, axes=tdot_axes)

    # Make sure the resulting shape is correct
    if (new_view.shape != shape_result) and not used_einsum:
        if (len(shape_result) > 0):
            new_view = new_view.reshape(shape_result)
        else:
            new_view = np.squeeze(new_view)

    # In-place mult by prefactor if requested
    if prefactor is not None:
        new_view *= prefactor

    # Do final tranpose if needed
    if used_einsum:
        return new_view
    elif tdot_result == output_ind:
        return new_view
    else:
        return np.einsum(tdot_result + '->' + output_ind, new_view)


class HelperCCEnergy(object):

    def __init__(self, mol, freeze_core=False, memory=2):

        if freeze_core:
            raise Exception("Frozen core doesnt work yet!")
        #print("\nInitalizing CCSD object...\n")

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        #print('Computing RHF reference.')
        psi4.core.set_active_molecule(mol)
        psi4.set_module_options('SCF', {'SCF_TYPE':'PK'})
        psi4.set_module_options('SCF', {'E_CONVERGENCE':10e-13})
        psi4.set_module_options('SCF', {'D_CONVERGENCE':10e-13})

        # Core is frozen by default
        if not freeze_core:
            psi4.set_module_options('CCENERGY', {'FREEZE_CORE':'FALSE'})

        self.rhf_e, self.wfn = psi4.energy('SCF', return_wfn=True)
        #print('RHF Final Energy                          % 16.10f\n' % (self.rhf_e))

        self.ccsd_corr_e = 0.0
        self.ccsd_e = 0.0

        #self.eps = np.asarray(self.wfn.epsilon_a())
        self.ndocc = self.wfn.doccpi()[0]
        self.nmo = self.wfn.nmo()
        self.memory = memory
        self.nfzc = 0

        # Freeze core
        if freeze_core:
            Zlist = np.array([mol.Z(x) for x in range(mol.natom())])
            self.nfzc = np.sum(Zlist > 2)
            self.nfzc += np.sum(Zlist > 10) * 4
            if np.any(Zlist > 18):
                raise Exception("Frozen core for Z > 18 not yet implemented")

            #print("Cutting %d core orbitals." % self.nfzc)

            # Copy C
            oldC = np.array(self.wfn.Ca(), copy=True)

            # Build new C matrix and view, set with numpy slicing
            self.C = psi.Matrix(self.nmo, self.nmo - self.nfzc)
            self.npC = np.asarray(self.C)
            self.npC[:] = oldC[:, self.nfzc:]

            self.ndocc -= self.nfzc

        else:
            self.C = self.wfn.Ca()
            self.npC = np.asarray(self.C)

        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        H = np.asarray(self.mints.ao_kinetic()) + np.asarray(self.mints.ao_potential())
        self.nmo = H.shape[0]

        # Update H, transform to MO basis
        H = np.einsum('uj,vi,uv', self.npC, self.npC, H)

        #print('Starting AO ->  MO transformation...')

        ERI_Size = self.nmo  * 128.e-9
        memory_footprint = ERI_Size * 5
        if memory_footprint > self.memory:
            psi.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB."
                                                % (memory_footprint, self.memory))

        # Integral generation from Psi4's MintsHelper
        self.MO = np.asarray(self.mints.mo_eri(self.C, self.C, self.C, self.C))
        self.MO = self.MO.swapaxes(1,2)
        #print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_Size, self.nmo))

        # Update nocc and nvirt
        self.nocc = self.ndocc
        self.nvirt = self.nmo - self.nocc - self.nfzc

        # Make slices
        self.slice_nfzc = slice(0, self.nfzc)
        self.slice_o = slice(self.nfzc, self.nocc + self.nfzc)
        self.slice_v = slice(self.nocc + self.nfzc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'f': self.slice_nfzc, 'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

        # Compute Fock matrix
        self.F = H + 2.0 * np.einsum('pmqm->pq', self.MO[:, self.slice_o, :, self.slice_o])
        self.F -= np.einsum('pmmq->pq', self.MO[:, self.slice_o, self.slice_o, :])

        ### Build D matrices
        Focc = np.diag(self.F)[self.slice_o]
        Fvir = np.diag(self.F)[self.slice_v]

        #print(Focc)
        tmp = Focc.reshape(-1,1)
        #print(tmp)
        #print(Fvir)

        self.Dia = Focc.reshape(-1, 1) - Fvir
        #print(self.Dia)
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir

        ### Construct initial guess
        #print('Building initial guess...')
        # t^a_i
        self.t1 = np.zeros((self.nocc, self.nvirt))
        # t^{ab}_{ij}
        self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v] / self.Dijab

        #print('\n..initialized CCSD in %.3f seconds.\n' % (time.time() - time_init))

    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v
    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]


    #Bulid Eqn 9: tilde{\Tau})
    def build_tilde_tau(self):
        ttau = self.t2.copy()
        tmp = 0.5 * np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        return ttau


    #Build Eqn 10: \Tau)
    def build_tau(self):
        ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        ttau += tmp
        return ttau


    #Build Eqn 3:
    def build_Fae(self):
        Fae = self.get_F('vv').copy()

        Fae -= ndot('me,ma->ae', self.get_F('ov'), self.t1, prefactor=0.5)

        Fae += ndot('mf,mafe->ae', self.t1, self.get_MO('ovvv'), prefactor=2.0)
        Fae += ndot('mf,maef->ae', self.t1, self.get_MO('ovvv'), prefactor=-1.0)

        Fae -= ndot('mnaf,mnef->ae', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=2.0)
        Fae -= ndot('mnaf,mnfe->ae', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=-1.0)


        return Fae


    #Build Eqn 4:
    def build_Fmi(self):
        Fmi = self.get_F('oo').copy()

        Fmi += ndot('ie,me->mi', self.t1, self.get_F('ov'), prefactor=0.5)

        Fmi += ndot('ne,mnie->mi', self.t1, self.get_MO('ooov'), prefactor=2.0)
        Fmi += ndot('ne,mnei->mi', self.t1, self.get_MO('oovo'), prefactor=-1.0)

        Fmi += ndot('inef,mnef->mi', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=2.0)
        Fmi += ndot('inef,mnfe->mi', self.build_tilde_tau(), self.get_MO('oovv'), prefactor=-1.0)
        return Fmi


    #Build Eqn 5:
    def build_Fme(self):
        Fme = self.get_F('ov').copy()
        Fme += ndot('nf,mnef->me', self.t1, self.get_MO('oovv'), prefactor=2.0)
        Fme += ndot('nf,mnfe->me', self.t1, self.get_MO('oovv'), prefactor=-1.0)
        return Fme


    #Build Eqn 6:
    def build_Wmnij(self):
        Wmnij = self.get_MO('oooo').copy()

        Wmnij += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'))
        Wmnij += ndot('ie,mnej->mnij', self.t1, self.get_MO('oovo'))

        Wmnij += ndot('ijef,mnef->mnij', self.build_tau(), self.get_MO('oovv'), prefactor=1.0)
        return Wmnij


    #Build Eqn 8:
    def build_Wmbej(self):
        Wmbej = self.get_MO('ovvo').copy()
        Wmbej += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        Wmbej -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))

        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1)

        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.get_MO('oovv'))

        Wmbej += ndot('njfb,mnef->mbej', self.t2, self.get_MO('oovv'), prefactor=1.0)
        Wmbej += ndot('njfb,mnfe->mbej', self.t2, self.get_MO('oovv'), prefactor=-0.5)
        return Wmbej

    def build_Wmbje(self):
        Wmbje = -1.0 * (self.get_MO('ovov').copy())
        Wmbje -= ndot('jf,mbfe->mbje', self.t1, self.get_MO('ovvv'))
        Wmbje += ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))

        tmp = (0.5 * self.t2)
        tmp += np.einsum('jf,nb->jnfb', self.t1, self.t1)

        Wmbje += ndot('jnfb,mnfe->mbje', tmp, self.get_MO('oovv'))
        return Wmbje

    def build_Zmbij(self):
        Zmbij = 0
        Zmbij += ndot('mbef,ijef->mbij', self.get_MO('ovvv'), self.build_tau())
        return Zmbij

    def update(self):
        # Updates amplitudes

        ### Build intermediates
        Fae = self.build_Fae()
        Fmi = self.build_Fmi()
        Fme = self.build_Fme()

        #### Build residual of self.t1 equations
        r_T1 = self.get_F('ov').copy()
        r_T1 += ndot('ie,ae->ia', self.t1, Fae)
        r_T1 -= ndot('ma,mi->ia', self.t1, Fmi)

        r_T1 += ndot('imae,me->ia', self.t2, Fme, prefactor=2.0)
        r_T1 += ndot('imea,me->ia', self.t2, Fme, prefactor=-1.0)

        r_T1 += ndot('nf,nafi->ia', self.t1, self.get_MO('ovvo'), prefactor=2.0)
        r_T1 += ndot('nf,naif->ia', self.t1, self.get_MO('ovov'), prefactor=-1.0)

        r_T1 += ndot('mief,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=2.0)
        r_T1 += ndot('mife,maef->ia', self.t2, self.get_MO('ovvv'), prefactor=-1.0)

        r_T1 -= ndot('mnae,nmei->ia', self.t2, self.get_MO('oovo'), prefactor=2.0)
        r_T1 -= ndot('mnae,nmie->ia', self.t2, self.get_MO('ooov'), prefactor=-1.0)

        ### Build RHS side of self.t2 equations
        r_T2 = self.get_MO('oovv').copy()

        # P^(ab)_(ij) {t_ijae Fae_be }
        tmp = ndot('ijae,be->ijab', self.t2, Fae)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0,1).swapaxes(2,3)

        # P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me }
        tmp = ndot('mb,me->be', self.t1, Fme)
        first = ndot('ijae,be->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0,1).swapaxes(2,3)

        # P^(ab)_(ij) {-t_imab Fmi_mj }
        tmp = ndot('imab,mj->ijab', self.t2, Fmi, prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0,1).swapaxes(2,3)

        # P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me }
        tmp = ndot('je,me->jm', self.t1, Fme)
        first = ndot('imab,jm->ijab', self.t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0,1).swapaxes(2,3)


        # tau_mnab Wmnij_mnij + tau_ijef <ab|ef> }
        tmp_tau = self.build_tau()
        Wmnij = self.build_Wmnij()
        Wmbej = self.build_Wmbej()
        Wmbje = self.build_Wmbje()
        Zmbij = self.build_Zmbij()


        r_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=1.0)
        r_T2 += ndot('ijef,abef->ijab', tmp_tau, self.get_MO('vvvv'), prefactor=1.0)

        # P^(ab)_(ij) {t_ie <ab|ej> }
        tmp = ndot('ie,abej->ijab', self.t1, self.get_MO('vvvo'), prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0,1).swapaxes(2,3)

        # P^(ab)_(ij) {-t_ma <mb|ij> }
        tmp = ndot('ma,mbij->ijab', self.t1, self.get_MO('ovoo'), prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0,1).swapaxes(2,3)

        # P...
        r_T2 += ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        r_T2 += ndot('imea,mbej->ijab', self.t2, Wmbej, prefactor=-1.0)

        r_T2 += ndot('imae,mbej->ijab', self.t2, Wmbej, prefactor=1.0)
        r_T2 += ndot('imae,mbje->ijab', self.t2, Wmbje, prefactor=1.0)

        r_T2 += ndot('mjae,mbie->ijab', self.t2, Wmbje, prefactor=1.0)
        r_T2 += ndot('imeb,maje->ijab', self.t2, Wmbje, prefactor=1.0)

        r_T2 += ndot('jmbe,maei->ijab', self.t2, Wmbej, prefactor=1.0)
        r_T2 += ndot('jmbe,maie->ijab', self.t2, Wmbje, prefactor=1.0)

        r_T2 += ndot('jmbe,maei->ijab', self.t2, Wmbej, prefactor=1.0)
        r_T2 += ndot('jmeb,maei->ijab', self.t2, Wmbej, prefactor=-1.0)

        # P....

        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        r_T2 -= ndot('imea,mbej->ijab', tmp, self.get_MO('ovvo'))

        tmp = ndot('ie,mb->imeb', self.t1, self.t1)
        r_T2 -= ndot('imeb,maje->ijab', tmp, self.get_MO('ovov'))

        tmp = ndot('je,ma->jmea', self.t1, self.t1)
        r_T2 -= ndot('jmea,mbie->ijab', tmp, self.get_MO('ovov'))

        tmp = ndot('je,mb->jmeb', self.t1, self.t1)
        r_T2 -= ndot('jmeb,maei->ijab', tmp, self.get_MO('ovvo'))

        r_T2 -= ndot('ma,mbij->ijab', self.t1, Zmbij)
        r_T2 -= ndot('ma,mbij->jiba', self.t1, Zmbij)


        ### Update T1 and T2 amplitudes
        self.t1 += r_T1 / self.Dia
        self.t2 += r_T2 / self.Dijab

        rms = np.einsum('ia,ia->', r_T1, r_T1)
        rms += np.einsum('ijab,ijab->', r_T2, r_T2)

        return np.sqrt(rms)


    def compute_corr_energy(self):
        CCSDcorr_E = 2.0 * np.einsum('ia,ia->', self.get_F('ov'), self.t1)
        tmp_tau = self.build_tau()
        CCSDcorr_E += 2.0 * np.einsum('ijab,ijab->', tmp_tau, self.get_MO('oovv'))
        CCSDcorr_E -= 1.0 * np.einsum('ijab,ijba->', tmp_tau, self.get_MO('oovv'))

        self.ccsd_corr_e = CCSDcorr_E
        self.ccsd_e = self.rhf_e + self.ccsd_corr_e
        return CCSDcorr_E

    def compute_energy(self, e_conv=1.e-13, maxiter=50, max_diis=8):
        ### Setup DIIS
        diis_vals_t1 = [self.t1.copy()]
        diis_vals_t2 = [self.t2.copy()]
        diis_errors = []

        ### Start Iterations
        ccsd_tstart = time.time()

        # Compute MP2 energy
        CCSDcorr_E_old = self.compute_corr_energy()
        #print("CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   MP2" % (0, CCSDcorr_E_old, -CCSDcorr_E_old))

        # Iterate!
        diis_size = 0
        for CCSD_iter in range(1, maxiter + 1):

            # Save new amplitudes
            oldt1 = self.t1.copy()
            oldt2 = self.t2.copy()

            rms = self.update()

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy()

            # Print CCSD iteration information
            #print('CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   DIIS = %d' % (CCSD_iter, CCSDcorr_E, (CCSDcorr_E - CCSDcorr_E_old), diis_size))

            # Check convergence
            #if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv):
            if (rms < e_conv):
                #print('\nCCSD has converged in %.3f seconds!' % (time.time() - ccsd_tstart))
                return CCSDcorr_E

            # Add DIIS vectors
            diis_vals_t1.append(self.t1.copy())
            diis_vals_t2.append(self.t2.copy())

            # Build new error vector
            error_t1 = (diis_vals_t1[-1] - oldt1).ravel()
            error_t2 = (diis_vals_t2[-1] - oldt2).ravel()
            diis_errors.append(np.concatenate((error_t1, error_t2)))

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            if CCSD_iter >= 1:
                # Limit size of DIIS vector
                if (len(diis_vals_t1) > max_diis):
                    del diis_vals_t1[0]
                    del diis_vals_t2[0]
                    del diis_errors[0]

                diis_size = len(diis_vals_t1) - 1

                # Build error matrix B
                B = np.ones((diis_size + 1, diis_size + 1)) * -1
                B[-1, -1] = 0

                for n1, e1 in enumerate(diis_errors):
                    B[n1, n1] = np.dot(e1, e1)
                    for n2, e2 in enumerate(diis_errors):
                        if n1 >= n2: continue
                        B[n1, n2] = np.dot(e1, e2)
                        B[n2, n1] = B[n1, n2]

                B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()

                # Build residual vector
                resid = np.zeros(diis_size + 1)
                resid[-1] = -1

                # Solve pulay equations
                ci = np.linalg.solve(B, resid)

                # Calculate new amplitudes
                self.t1[:] = 0
                self.t2[:] = 0
                for num in range(diis_size):
                    self.t1 += ci[num] * diis_vals_t1[num + 1]
                    self.t2 += ci[num] * diis_vals_t2[num + 1]

            # End DIIS amplitude update
            # End HelperCCEnergy class

class HelperCCHbar(object):

    def __init__(self, ccsd, memory=2):

        # Integral generation from Psi4's MintsHelper
        time_init = time.time()

        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nfzc = 0
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc - ccsd.nfzc

        self.slice_nfzc = slice(0, self.nfzc)
        self.slice_o = slice(self.nfzc, self.nocc + self.nfzc)
        self.slice_v = slice(self.nocc + self.nfzc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'f': self.slice_nfzc, 'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}


        self.F = ccsd.F
        self.Dia = ccsd.Dia
        self.Dijab = ccsd.Dijab
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        #print(self.t1)
        #print(self.t2)
        #print('\nBuilding appropriate pieces of similarity transformed hamiltonian ...')

        tmp = self.MO.copy()
        self.L = 2.0 * tmp
        self.L -= tmp.swapaxes(2,3)

        self.build_Hov()
        #print('\n Hov \n')
        #print(self.Hov)
        self.build_Hoo()
        #print('\n Hoo \n')
        #print(self.Hoo)
        self.build_Hvv()
        #print('\n Hvv \n')
        #print(self.Hvv)
        self.build_Hoooo()
        #print('\n Hoooo \n')
        #print(self.Hoooo)
        self.build_Hvvvv()
        #print('\n Hvvvv \n')
        #print(self.Hvvvv)
        self.build_Hvovv()
        #print('\n Hvovv \n')
        #print(self.Hvovv)
        self.build_Hooov()
        #print('\n Hoovv \n')
        #print(self.Hooov)
        self.build_Hovvo()
        #print('\n Hovvo \n')
        #print(self.Hovvo)
        self.build_Hovov()
        #print('\n Hovov \n')
        #print(self.Hovov)
        self.build_Hvvvo()
        #print('\n Hvvvo \n')
        #print(self.Hvvvo)
        self.build_Hovoo()
        #print('\n Hovoo \n')
        #print(self.Hovoo)
        #print('\n..CCHBAR completed !!')

    # occ orbitals i, j, k, l, m, n
    # virt orbitals a, b, c, d, e, f
    # all oribitals p, q, r, s, t, u, v

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                     self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_L(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_L: string %s must have 4 elements.' % string)
        return (self.L[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]])


    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def build_tau(self):
        self.ttau = self.t2.copy()
        tmp = np.einsum('ia,jb->ijab', self.t1, self.t1)
        self.ttau += tmp
        return self.ttau

    def build_Hov(self):
        self.Hov = self.get_F('ov').copy()
        self.Hov += ndot('nf,mnef->me', self.t1, self.get_L('oovv'))
        return self.Hov

    def build_Hoo(self):
        self.Hoo = self.get_F('oo').copy()
        self.Hoo += ndot('ie,me->mi', self.t1, self.get_F('ov'))
        self.Hoo += ndot('ne,mnie->mi', self.t1, self.get_L('ooov'))
        self.Hoo += ndot('mnef,inef->mi', self.get_L('oovv'), self.build_tau())
        return self.Hoo

    def build_Hvv(self):
        self.Hvv = self.get_F('vv').copy()
        self.Hvv -= ndot('ma,me->ae', self.t1, self.get_F('ov'))
        self.Hvv += ndot('amef,mf->ae', self.get_L('vovv'), self.t1)
        self.Hvv -= ndot('mnfa,mnfe->ae', self.build_tau(), self.get_L('oovv'))
        return self.Hvv

    def build_Hoooo(self):
        self.Hoooo = self.get_MO('oooo').copy()
        self.Hoooo += ndot('je,mnie->mnij', self.t1, self.get_MO('ooov'), prefactor=2.0)
        self.Hoooo += ndot('mnef,ijef->mnij', self.get_MO('oovv'), self.build_tau())
        return self.Hoooo

    def build_Hvvvv(self):
        self.Hvvvv = self.get_MO('vvvv').copy()
        self.Hvvvv -= ndot('mb,amef->abef', self.t1, self.get_MO('vovv'), prefactor=2.0)
        self.Hvvvv += ndot('mnab,mnef->abef', self.build_tau(), self.get_MO('oovv'))
        return self.Hvvvv

    def build_Hvovv(self):
        self.Hvovv = self.get_MO('vovv').copy()
        self.Hvovv -= ndot('na,nmef->amef', self.t1, self.get_MO('oovv'))
        return self.Hvovv

    def build_Hooov(self):
        self.Hooov = self.get_MO('ooov').copy()
        self.Hooov += ndot('if,nmef->mnie', self.t1, self.get_MO('oovv'))
        return self.Hooov

    def build_Hovvo(self):
        self.Hovvo = self.get_MO('ovvo').copy()
        self.Hovvo += ndot('jf,mbef->mbej', self.t1, self.get_MO('ovvv'))
        self.Hovvo -= ndot('nb,mnej->mbej', self.t1, self.get_MO('oovo'))
        self.Hovvo -= ndot('njbf,mnef->mbej', self.build_tau(), self.get_MO('oovv'))
        self.Hovvo += ndot('njfb,mnef->mbej', self.t2, self.get_L('oovv'))
        return self.Hovvo

    def build_Hovov(self):
        self.Hovov = self.get_MO('ovov').copy()
        self.Hovov += ndot('jf,bmef->mbje', self.t1, self.get_MO('vovv'))
        self.Hovov -= ndot('nb,mnje->mbje', self.t1, self.get_MO('ooov'))
        self.Hovov -= ndot('jnfb,nmef->mbje', self.build_tau(), self.get_MO('oovv'))
        return self.Hovov

    def build_Hvvvo(self):
        self.Hvvvo =  self.get_MO('vvvo').copy()
        self.Hvvvo += ndot('if,abef->abei', self.t1, self.get_MO('vvvv'))
        self.Hvvvo -= ndot('mb,amei->abei', self.t1, self.get_MO('vovo'))
        self.Hvvvo -= ndot('ma,bmie->abei', self.t1, self.get_MO('voov'))
        self.Hvvvo -= ndot('imfa,mbef->abei', self.build_tau(), self.get_MO('ovvv'))
        self.Hvvvo -= ndot('imfb,amef->abei', self.build_tau(), self.get_MO('vovv'))
        self.Hvvvo += ndot('mnab,mnei->abei', self.build_tau(), self.get_MO('oovo'))
        self.Hvvvo -= ndot('me,miab->abei', self.get_F('ov'), self.t2)
        self.Hvvvo += ndot('mifb,amef->abei', self.t2, self.get_L('vovv'))
        tmp =    ndot('mnef,if->mnei', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('mnab,mnei->abei', self.t2, tmp)
        tmp =    ndot('mnef,ma->anef', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('infb,anef->abei', self.t2, tmp)
        tmp =    ndot('mnef,nb->mefb', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('miaf,mefb->abei', self.t2, tmp)
        tmp =    ndot('mnfe,mf->ne', self.get_L('oovv'), self.t1)
        self.Hvvvo -= ndot('niab,ne->abei', self.t2, tmp)
        tmp =    ndot('mnfe,na->mafe', self.get_L('oovv'), self.t1)
        self.Hvvvo -= ndot('mifb,mafe->abei', self.t2, tmp)
        tmp1 =   ndot('if,ma->imfa', self.t1, self.t1)
        tmp2 =   ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
        self.Hvvvo += ndot('imfa,mbef->abei', tmp1, tmp2)
        return self.Hvvvo

    def build_Hovoo(self):
        self.Hovoo =  self.get_MO('ovoo').copy()
        self.Hovoo += ndot('mbie,je->mbij', self.get_MO('ovov'), self.t1)
        self.Hovoo += ndot('ie,bmje->mbij', self.t1, self.get_MO('voov'))
        self.Hovoo -= ndot('nb,mnij->mbij', self.t1, self.get_MO('oooo'))
        self.Hovoo -= ndot('ineb,nmje->mbij', self.build_tau(), self.get_MO('ooov'))
        self.Hovoo -= ndot('jneb,mnie->mbij', self.build_tau(), self.get_MO('ooov'))
        self.Hovoo += ndot('ijef,mbef->mbij', self.build_tau(), self.get_MO('ovvv'))
        self.Hovoo += ndot('me,ijeb->mbij', self.get_F('ov'), self.t2)
        self.Hovoo += ndot('njeb,mnie->mbij', self.t2, self.get_L('ooov'))
        tmp =    ndot('mnef,jf->mnej', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('ineb,mnej->mbij', self.t2, tmp)
        tmp =    ndot('mnef,ie->mnif', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('jnfb,mnif->mbij', self.t2, tmp)
        tmp =    ndot('mnef,nb->mefb', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('ijef,mefb->mbij', self.t2, tmp)
        tmp =    ndot('mnef,njfb->mejb', self.get_L('oovv'), self.t2)
        self.Hovoo += ndot('mejb,ie->mbij', tmp, self.t1)
        tmp =    ndot('mnef,nf->me', self.get_L('oovv'), self.t1)
        self.Hovoo += ndot('me,ijeb->mbij', tmp, self.t2)
        tmp1 =   ndot('ie,jf->ijef', self.t1, self.t1)
        tmp2 =   ndot('mnef,nb->mbef', self.get_MO('oovv'), self.t1)
        self.Hovoo -= ndot('mbef,ijef->mbij', tmp2, tmp1)
        return self.Hovoo

class HelperCCEom(object):
    """
    EOMCCSD helper class for spin adapted EOMCCSD

    """

    def __init__(self, ccsd, cchbar):
        """
        Initializes the HelperCCEom object

        Parameters
        ----------
        ccsd: HelperCCSd object
            Energy should already be computed

        cchbar: HelperCCHbar object


        Returns
        -------
        ret : HelperCCEom
            An initialized HelperCCEom object

        Notes
        -----
        Spin orbital sigma equations for EOMCCSDT can be found in:
            I. Shavitt and R. J. Bartlett, "Many-Body Methods in Chemistry and
            Physics: MBPT and Coupled-Cluster Theory", Cambridge University
            Press, 2009.
        The relevant contributions for EOMCCSD were extracted and the equations
        spin adapted to arrive at the equations implemented in this class.

        Special thanks to Ashutosh Kumar for Hbar components and help with spin
        adaptation.

        """
        # Steal dimensions
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nfzc = 0
        self.nocc = ccsd.ndocc
        self.nvir = ccsd.nmo - ccsd.nocc - ccsd.nfzc
        self.nsingles = self.ndocc * self.nvir
        self.ndoubles = self.ndocc * self.ndocc * self.nvir * self.nvir

        # Steal integrals/amps from ccsd
        self.MO = ccsd.MO
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2

        # Steal "ova" translation
        self.slice_nfzc = cchbar.slice_nfzc
        self.slice_o = cchbar.slice_o
        self.slice_v = cchbar.slice_v
        self.slice_a = cchbar.slice_a
        self.slice_dict = cchbar.slice_dict

        # Steal Hbar blocks
        self.Hov = cchbar.Hov
        self.Hoo = cchbar.Hoo
        self.Hvv = cchbar.Hvv
        self.Hoooo = cchbar.Hoooo
        self.Hvvvv = cchbar.Hvvvv
        self.Hvovv = cchbar.Hvovv
        self.Hooov = cchbar.Hooov
        self.Hovvo = cchbar.Hovvo
        self.Hovov = cchbar.Hovov
        self.Hvvvo = cchbar.Hvvvo
        self.Hovoo = cchbar.Hovoo

        # Steal L integrals (L[pqrs] = 2*MO[pqrs] - MO[pqsr])
        self.L = cchbar.L

        # Build Approximate Diagonal of Hbar
        self.Dia = self.Hoo.diagonal().reshape(-1, 1) - self.Hvv.diagonal()
        self.Dijab = self.Hoo.diagonal().reshape(
            -1, 1, 1, 1) + self.Hoo.diagonal().reshape(
                -1, 1, 1) - self.Hvv.diagonal().reshape(
                    -1, 1) - self.Hvv.diagonal()

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def get_L(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_L: string %s must have 4 elements.' % string)
        return self.L[self.slice_dict[string[0]], self.slice_dict[string[1]],
                      self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def build_sigma1(self, B1, B2):
        """
        Compute the contributions to <ia|Hbar*B|0>

        Parameters
        ----------
        B1: array like, shape(ndocc, nvir)
          The first nsingles elements of a guess vector reshaped to (o,v)

        B2: array like, shape(ndocc,ndocc,nvir,nvir)
          The last ndoubles elements of a guess vector reshaped to (o,o,v,v)

        Returns
        -------
        S1: ndarray shape(ndocc, nvir)

        Examples
        --------

        >>> # Get some vectors as cols of a 2D numpy array and orthogonalize them
        >>> c  = np.random.rand(eom.nsingles + eom.ndoubles, 2)
        >>> c,  = np.linalg.qr(c)

        >>> # Slice out the singles, doubles blocks of the first vector and reshape
        >>> B1 = c[:,:nsingles].reshape(eom.ndocc, eom.nvir)
        >>> B2 = c[:,nsingles:].reshape(eom.ndocc, eom.ndocc, eom.nvir, eom.nvir)
        >>> S1 = eom.build_sigma1(B1, B2)

        """
        S1 = ndot('ie,ae->ia', B1, self.Hvv)
        S1 -= ndot('mi,ma->ia', self.Hoo, B1)
        S1 += ndot('maei,me->ia', self.Hovvo, B1, prefactor=2.0)
        S1 += ndot('maie,me->ia', self.Hovov, B1, prefactor=-1.0)
        S1 += ndot('miea,me->ia', B2, self.Hov, prefactor=2.0)
        S1 += ndot('imea,me->ia', B2, self.Hov, prefactor=-1.0)
        S1 += ndot('imef,amef->ia', B2, self.Hvovv, prefactor=2.0)
        S1 += ndot('imef,amfe->ia', B2, self.Hvovv, prefactor=-1.0)
        S1 -= ndot('mnie,mnae->ia', self.Hooov, B2, prefactor=2.0)
        S1 -= ndot('nmie,mnae->ia', self.Hooov, B2, prefactor=-1.0)
        return S1

    def build_sigma2(self, B1, B2):
        """
        Compute the contributions to <ijab|Hbar*B|0>:

        Parameters
        ----------
        B1: array like, shape(ndocc, nvir)
          The first nsingles elements of a guess vector reshaped to (o,v)

        B2: array like, shape(ndocc,ndocc,nvir,nvir)
          The last ndoubles elements of a guess vector reshaped to (o,o,v,v)

        Returns
        -------
        S2: ndarray shape(ndocc, ndocc, nvir, nvir)

        Examples
        --------

        >>> # Get some vectors as cols of a 2D numpy array and orthogonalize them
        >>> c  = np.random.rand(eom.nsingles + eom.ndoubles, 2)
        >>> c,  = np.linalg.qr(c)

        >>> # Slice out the singles, doubles blocks of the first vector and reshape
        >>> B1 = c[:,:nsingles].reshape(eom.ndocc, eom.nvir)
        >>> B2 = c[:,nsingles:].reshape(eom.ndocc, eom.ndocc, eom.nvir, eom.nvir)
        >>> S2 = eom.build_sigma2(B1, B2)

        """
        S_2 = ndot('ie,abej->ijab', B1, self.Hvvvo)
        S_2 -= ndot('mbij,ma->ijab', self.Hovoo, B1)

        Zvv = ndot("amef,mf->ae", self.Hvovv, B1, prefactor=2.0)
        Zvv += ndot("amfe,mf->ae", self.Hvovv, B1, prefactor=-1.0)
        Zvv -= ndot('nmaf,nmef->ae', B2, self.get_L('oovv'))
        S_2 += ndot('ijeb,ae->ijab', self.t2, Zvv)

        Zoo = ndot('mnie,ne->mi', self.Hooov, B1, prefactor=-2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, B1, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.get_L('oovv'), B2)
        S_2 += ndot('mi,mjab->ijab', Zoo, self.t2)

        S_2 += ndot('ijeb,ae->ijab', B2, self.Hvv)
        S_2 -= ndot('mi,mjab->ijab', self.Hoo, B2)

        S_2 += ndot('mnij,mnab->ijab', self.Hoooo, B2, prefactor=0.5)
        S_2 += ndot('ijef,abef->ijab', B2, self.Hvvvv, prefactor=0.5)

        S_2 -= ndot('imeb,maje->ijab', B2, self.Hovov)
        S_2 -= ndot('imea,mbej->ijab', B2, self.Hovvo)

        S_2 += ndot('miea,mbej->ijab', B2, self.Hovvo, prefactor=2.0)
        S_2 += ndot('miea,mbje->ijab', B2, self.Hovov, prefactor=-1.0)
        return S_2 + S_2.swapaxes(0, 1).swapaxes(2, 3)

if __name__ == "__main__":
    arr4 = np.random.rand(4, 4, 4, 4)
    arr2 = np.random.rand(4, 4)

    def test_ndot(string, op1, op2):
        ein_ret = np.einsum(string, op1, op2)
        ndot_ret = ndot(string, op1, op2)
        assert np.allclose(ein_ret, ndot_ret)

    test_ndot('abcd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abef', arr4, arr4)
    test_ndot('acbd,cdef->abfe', arr4, arr4)
    test_ndot('mnab,mnij->ijab', arr4, arr4)

    test_ndot('cd,cdef->ef', arr2, arr4)
    test_ndot('ce,cdef->df', arr2, arr4)
    test_ndot('nf,naif->ia', arr2, arr4)
