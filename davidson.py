# Generalized Davidson solver

import numpy as np

np.set_printoptions(precision=12, suppress=True)

def davidson_solver(ax_function, preconditioner, guess, e_conv=1.0E-8,
        r_conv=None, no_eigs=1, max_vecs_per_root=20, maxiter=100):
    """
    Solves for the lowest few eigenvalues and eigenvectors of the given real symmetric matrix

    Parameters
    -----------
    ax_function : function
        Takes in a guess vector and returns the Matrix-vector product.
    preconditioner : function
        Takes in a list of :py:class:`~psi4.core.Matrix` objects and a mask of active indices. Returns the preconditioned value.
    guess : list of :py:class:`~psi4.core.Matrix`
        Starting vectors, required
    e_conv : float
        Convergence tolerance for eigenvalues
    r_conv : float
        Convergence tolerance for residual vectors
    no_eigs : int
        Number of eigenvalues needed
    max_vecs_per_root : int
        Maximum number of vectors per root before subspace collapse.
    maxiter : int
        The maximum number of iterations this function will take.

    Returns
    -----------

    Notes
    -----------

    Examples
    -----------

    """

    print("Starting Davidson algo:")
    print("options:")
    for k,v in dict(e_conv=e_conv, r_conv=r_conv, no_eigs=no_eigs,
            max_vecs_per_root = max_vecs_per_root, maxiter=maxiter).items():
        if v == None:
            v = 'None'
        print("{:<25}{:.>100}".format(k,v))
    if r_conv == None:
        r_conv = e_conv * 100
    d_tol = 1.0E-8

    # using the shape of the guess vectors to set the dimension of the matrix
    N = guess.shape[0]

    #sanity check, guess subspace must be at least equal to number of eigenvalues
    nli = guess.shape[1]
    if nli < no_eigs:
        raise ValueError("Not enough guess vectors provided!")

    nl = nli
    converged=False
    count = 0
    A_w_old = np.ones(no_eigs)
    max_ss_size = no_eigs * max_vecs_per_root
    B = guess

    ### begin loop
    conv_roots = [False] * no_eigs
    while count < maxiter:
        # active_mask = [True for x in range(nl)] # never used later
        # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
        ## orthogonalize preconditioned residuals against all other vectors in the search subspace
        #B, r = np.linalg.qr(B)
        nl = B.shape[1]

        print("Davidson: Iter={:<3} nl = {:<4}".format(count, nl))
        # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
        sigma = np.zeros_like(B)
        for i in range(nl):
            sigma[:,i] = ax_function(B[:,i])

        # compute subspace matrix A_b = Btranspose sigma
        A_b = np.dot(B.T, sigma)

        # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
        A_w, A_v = np.linalg.eig(A_b)

        # sorting eigenvalues and corresponding eigenvectors
        idx = A_w.argsort()[:no_eigs]
        A_v_imag = A_v[:, idx].imag
        A_w_imag = A_w[idx].imag
        A_v = A_v[:, idx].real
        A_w = A_w[idx].real

        # Print warning if complex parts are too large
        for i,w_i in enumerate(A_w_imag):
            if abs(w_i) > 1.0e-10:
                print("*** WARNING***")
                print("    Root {:>5}: |Imag[A_w[{}]]| > 1.0E-10!".format(i))
                print("              : |Imag[A_v[{}]]| = {:.2E}".format(
                    i, np.linalg.norm(A_v_imag[:, i])))


        # here, check if no residuals > max no residuals, if so, collapse subspace
        if nl >= max_ss_size:
            print("Subspace too big. Collapsing.\n")
            B = np.dot(B, A_v)
            continue
        # else, build residual vectors
        ## residual_i = sum(j) sigma_j* eigvec_i - eigval_i * B_j * eigvec_i
        norm = np.zeros(no_eigs)
        new_Bs = []
        # Only need a residual for each desired root, not one for each guess
        force_collapse = False
        for i in range(no_eigs):
            if conv_roots[i]:
                print("    ROOT {:<3}: CONVERGED!".format(i))
                continue
            residual = np.dot(sigma, A_v[:, i]) - A_w[i] * np.dot(B, A_v[:,i])

            # check for convergence by norm of residuals
            norm[i] = np.linalg.norm(residual)
            # apply the preconditioner
            precon_resid = preconditioner(residual, i, A_w[i])
            de = abs(A_w_old[i] - A_w[i])
            print("    ROOT {:<3} de = {:<10.6f} |r| = {:<10.6f}".format(i, de,
                norm[i]))
            conv_roots[i] = (de < e_conv) and (norm[i] < r_conv)
            if conv_roots[i]:
                force_collapse = True
            else:
                new_Bs.append(precon_resid)

        # check for convergence by diff of eigvals and residual norms
        # r_norm = np.linalg.norm(norm)
        # eig_norm = np.linalg.norm(A_w - A_w_old)
        A_w_old = A_w
        #if( r_norm < r_conv) and (eig_norm < e_conv):
        if all(conv_roots):
            converged = True
            print("Davidson converged at iteration number {}".format(count))
            print("{:<3}|{:^20}|".format('count','eigenvalues'))
            print("{:<3}|{:^20}|".format('='*3, '='*20))
            retvecs = np.dot(B,A_v)
            for i, val in enumerate(A_w):
                print("{:<3}|{:<20.12f}".format(i, val))
            return A_w, retvecs
        else:
            if force_collapse:
                B = np.dot(B, A_v)
            n_left_to_converge = np.count_nonzero(np.logical_not(conv_roots))
            n_converged = np.count_nonzero(conv_roots)
            max_ss_size = n_converged + (n_left_to_converge * max_vecs_per_root)
            #B = np.column_stack(tuple(B[:, i] for i in range(B.shape[-1])) + tuple(new_Bs))
            for i in range(new_Bs.shape[1])
                B.transpose_this()
                B.add_and_orthogonalize_row(new_Bs[:,i])
                B.transpose_this()

        count += 1

    if not converged:
        print("Davidson did not converge. Max iterations exceeded.")
        return None, None
