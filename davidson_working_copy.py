# Generalized Davidson solver

import numpy as np

from psi4 import core

np.set_printoptions(precision=12, suppress=True)

N = 1000
spf = 0.0001
A = np.zeros((N,N))
randA = np.random.RandomState(13)
for i in range(0,N):
    A[i,i] = i + 1.0
A = A + randA.randn(N,N)* spf
A = (A + A.T)/2

def ax_function(vec):
    ax = np.dot(A,vec)
    return ax

def preconditioner(residual, x, A, A_w):
    precon_resid = np.zeros_like(residual)
    diag = np.diagonal(A)
    precon_resid = residual / (A_w[x] - diag[x]) 
    return precon_resid

def davidson_solver(ax_function, preconditioner, guess, e_conv=1.0E-8, r_conv=None, no_eigs=1, max_vecs_per_root=10, maxiter=100):
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
    maxiter : int
        The maximum number of iterations this function will take.

    Returns
    -----------
    
    Notes
    -----------

    Examples
    -----------

    """

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
    sub_count = nli
    A_w_old = np.ones(nli)
    max_ss_size = nli * max_vecs_per_root
    B = np.zeros((N,N))
    B[:,:nli] = guess

    ### begin loop
    while count < maxiter:
        active_mask = [True for x in range(nl)]
        # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
        ## orthogonalize preconditioned residuals against all other vectors in the search subspace
        B, r = np.linalg.qr(B)

        # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
        sigma = np.zeros((N,nl))
        for i in range(nl):
            bvec = B[:,i]
            sigma[:,i] = ax_function(B[:,i])

        # compute subspace matrix A_b = Btranspose sigma
        A_b = np.dot(B[:,:nl].T, sigma)

        # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
        A_w, A_v = np.linalg.eig(A_b)

        # sorting eigenvalues and corresponding eigenvectors
        A_v = A_v[:, A_w.argsort()]
        A_w = A_w[A_w.argsort()]

        # here, check if no residuals > max no residuals, if so, collapse subspace
        sub_count = A_v.shape[0]
        if sub_count >= max_ss_size:
            print("Subspace too big. Collapsing.\n")
            Bnew = np.zeros((N,N))
            Bnew[:,:nli] = np.dot(B[:,:nl], A_v[:,:nli])
            B = Bnew
            nl = nli
            continue
        # else, build residual matrix
        ## residual_i = sigma * eigvec - eigval * B * eigvec
        norm = np.zeros(nli)
        for i in range(0, nli):
            mat = A - A_w[i] * np.identity(N) 
            residual = np.dot(mat, np.dot(B[:,:sub_count], A_v[:,i]))

            ## check for convergence by norm of residuals
            norm[i] = np.linalg.norm(residual)
        ##apply the preconditioner (A_ii - A_v_i)^-1
            precon_resid = preconditioner(residual, i, A, A_w)

        ## normalize and add to search subspace if they're larger than a threshold
            if np.linalg.norm(precon_resid) > d_tol:
                B[:,nl+1] = precon_resid
                nl += 1

        # check for convergence by diff of eigvals and residual norms
        check = norm < r_conv
        eig_norm = np.linalg.norm(A_w[:no_eigs] - A_w_old[:no_eigs])
        A_w_old = A_w
        if(check.all() == True and eig_norm < e_conv):
            converged = True
            break
        count += 1 

    if converged:
        print("Davidson converged at iteration number {}. \n Eigenvalues: {} \n Eigenvectors: {}".format(count, A_w[:no_eigs], A_v[:,:no_eigs]))
    else:
        print("Davidson did not converge. Max iterations exceeded.")

if __name__ == "__main__":
    N = 1000
    n_g = 10
    n_e = 5
    g_vec = np.eye(N)[:,:n_g]
    davidson_solver(ax_function, preconditioner, g_vec, max_vecs_per_root=20, no_eigs=n_e)
