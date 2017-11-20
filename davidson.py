# Generalized Davidson solver

import numpy as np

from psi4 import core

def preconditioner(residual, active_mask, A, A_w):
    precon_resid = np.zeros_like(residual)
    for x in range(len(active_mask)):
        precon_resid[:,x] = 1.0/(A[x,x] - A_w[x]) * residual[:,x]
    return precon_resid

def davidson_solver(mat, ax_function, preconditioner, guess=None, no_eigs=5, maxiter=100):
    """
    Solves for the lowest few eigenvalues and eigenvectors of the given real symmetric matrix

    Parameters
    -----------
    mat : psi4.core.Matrix object
        The matrix whose eigenvalues are sought
    ax_function : function
        Takes in ?? Returns the Matrix-vector product.
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

N = 40
no_eigs = 5
nl = 5
conv = 1.0E-12
active_mask = [True for x in range(no_eigs)]
A = np.random.random_integers(-2000,2000,size=(N,N))
A = (A + A.T)/2

# build guess vector matrix B
## sort diagonal elements in ascending order, generate corresponding unit vectors
x = A.diagonal()
j = 0
B = np.zeros_like(A)

# print(x.argsort())
for i in x.argsort():
    B[i, j] = 1
    j += 1
B = B[::, :no_eigs]
print("B")
print(B)

converged=False
count = 0
### begin loop
while count<10:
    # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
    ## orthogonalize preconditioned residuals against all other vectors in the search subspace
    B, r = np.linalg.qr(B)
    print("B")
    print(B)
    # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
    sigma = np.zeros((nl, N))

    sigma = np.dot(A, B)
    sigma = sigma[::, :nl]

    print("sigma")
    print(sigma)
    # compute subspace matrix A_v = Vtranspose sigma
    A_b = np.dot(B.T, sigma)

    print(A_b)
    # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
    A_w, A_v = np.linalg.eig(A_b)

    print("A_w")
    print(A_w)
    print("A_v")
    print(A_v)
    A_v = A_v[:, A_w.argsort()]
    A_w = A_w[A_w.argsort()]
    print("A_w (sorted)")
    print(A_w)
    print("A_v (sorted)")
    print(A_v)
    # here, check if no residuals > max no residuals, if so, collapse subspace

    #if len(B[0] > 20):
    #    B = np.dot(B, A_v)
    #    B = B[::, :no_eigs]
    # else, build residual matrix
    ## residual_i = sigma * eigvec - eigval * B * eigvec
    residual = np.zeros((N, no_eigs))
    for i in range(no_eigs):
        residual[:,i] = np.dot(sigma, A_v[:,i]) - A_w[i] * np.dot(B, A_v[:, i]) 

    print("residual matrix:")
    print(residual)
    ## check for convergence by norm of residuals
    norm = np.mean(residual, axis=0)**2
    print("Norm\n", norm)
    if(norm.all() < conv):
        converged = True
        break
    ## else apply the preconditioner (A_ii - A_v_i)^-1
    else:
        precon_resid = preconditioner(residual, active_mask, A, A_w)
        print("precon_resid matrix")
        print(precon_resid)
    ## normalize and add to search subspace if they're larger than a threshold
        for i in range(no_eigs):
            to_append = precon_resid[:,i]
            print("Preconditioned residual ",i, to_append)
            if np.linalg.norm(to_append) > conv:
                print("Norm larger than conv. Appending. B: ")
                B = np.append(B, precon_resid[:,i])
                nl += 1
                B = B.reshape(nl,N).T
                print(B)
    print("nl")
    print(nl)
    print("new B matrix")
    print(B)
        
    print(converged)
    count += 1 
    
