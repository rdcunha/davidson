# Generalized Davidson solver

import numpy as np

from psi4 import core

np.set_printoptions(precision=12, suppress=True)

def preconditioner(residual, x, A, A_w):
    precon_resid = np.zeros_like(residual)
    diag = np.diagonal(A)
    precon_resid = residual / (A_w[x] - diag[x]) 
    return precon_resid

def davidson_solver(ax_function, preconditioner, guess=None, no_eigs=5, maxiter=100):
    """
    Solves for the lowest few eigenvalues and eigenvectors of the given real symmetric matrix

    Parameters
    -----------
    ax_function : function
        Takes in a guess vector and returns the Matrix-vector product.
    preconditioner : function
        Takes in a list of :py:class:`~psi4.core.Matrix` objects and a mask of active indices. Returns the preconditioned value.
    guess : list of :py:class:`~psi4.core.Matrix`
        Starting vectors, if None a unit vector guess
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

N = 1000
no_eigs = 5
nli = 10
nl = nli
r_conv = 1.0E-6
e_conv = 1.0E-8
d_tol = 1.0E-8
maxiter = 100
spf = 0.0001

A = np.zeros((N,N))
randA = np.random.RandomState(13)
for i in range(0,N):
    A[i,i] = i + 1.0
A = A + randA.randn(N,N)* spf
A = (A + A.T)/2

# build guess vector matrix B
## sort diagonal elements in ascending order, generate corresponding unit vectors
x = A.diagonal()
j = 0
B = np.zeros_like(A)
for i in x.argsort():
    B[i, j] = 1
    j += 1

converged=False
count = 0
sub_count = nli
A_w_old = np.ones(nli)
### begin loop
while count < maxiter:
    active_mask = [True for x in range(nl)]
    # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
    ## orthogonalize preconditioned residuals against all other vectors in the search subspace
    B, r = np.linalg.qr(B)

    # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
    sigma = np.dot(A,B[:,:nl])

    # compute subspace matrix A_b = Btranspose sigma
    A_b = np.dot(B[:,:nl].T, sigma)

    # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
    A_w, A_v = np.linalg.eig(A_b)

    # sorting eigenvalues and corresponding eigenvectors
    A_v = A_v[:, A_w.argsort()]
    A_w = A_w[A_w.argsort()]

    # here, check if no residuals > max no residuals, if so, collapse subspace
    sub_count = A_v.shape[0]
    if sub_count >= N-(2*no_eigs):
        print("Subspace too big. Collapsing.\n")
        B = np.zeros_like(A)
        B[:,:nli] = np.dot(B[:,:nl], A_v)
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
