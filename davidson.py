# Generalized Davidson solver

import numpy as np

from psi4 import core

def preconditioner(residual, active_mask, A, A_w):
    precon_resid = np.zeros_like(residual)
    for x in range(len(active_mask)):
        if (A[x,x] - A_w[x]) < 1.0E-8:
            precon_resid[:,x] = residual[:,x]
        else:
            precon_resid[:,x] = (1.0/(A[x,x] - A_w[x])) * residual[:,x]
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

N = 30
no_eigs = 5
nl = 5
conv = 1.0E-8
active_mask = [True for x in range(no_eigs)]
# A = np.random.random_integers(-200,200,size=(N,N))
randA = np.random.RandomState(13)
A = randA.random_integers(-200,200,size=(N,N))
A = (A + A.T)/2

print("A\n",A)
print("Eigvals of A: \n",np.linalg.eig(A)[0])

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
# print("B\n", B)

converged=False
count = 0
### begin loop
while count<30:
    # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
    ## orthogonalize preconditioned residuals against all other vectors in the search subspace
    B, r = np.linalg.qr(B)
    #print("B\n", B)
    # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
    sigma = np.zeros((nl, N))

    sigma = np.dot(A, B)
    sigma = sigma[::, :nl]

    #print("sigma\n", sigma)
    # compute subspace matrix A_v = Vtranspose sigma
    A_b = np.dot(B.T, sigma)

    #print("Subspace matrix \n", A_b)
    # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
    A_w, A_v = np.linalg.eig(A_b)

    #print("A_w\n", A_w, "\nA_v\n", A_v)
    A_v = A_v[:, A_w.argsort()]
    A_w = A_w[A_w.argsort()]
    # print("A_w (sorted)\n", A_w, "\nA_v (sorted)\n", A_v)
    # here, check if no residuals > max no residuals, if so, collapse subspace

    if len(B[1]) >= N-(2*no_eigs):
        B = np.dot(B, A_v)
        B = B[::, :no_eigs]
        nl = no_eigs
        continue
    # else, build residual matrix
    ## residual_i = sigma * eigvec - eigval * B * eigvec
    residual = np.zeros((N, no_eigs))
    for i in range(no_eigs):
        residual[:,i] = np.dot(sigma, A_v[:,i]) - A_w[i] * np.dot(B, A_v[:, i]) 
    
    for i in range(no_eigs):
        if np.linalg.norm(residual[:,i]) < conv:
            residual = np.delete(residual, i, 1)

    #print("residual matrix:\n", residual)
    ## check for convergence by norm of residuals
    norm = np.mean(residual, axis=0)**2
  #  print("Norm\n", norm)
    if(norm.all() < conv):
        converged = True
        break
    ## else apply the preconditioner (A_ii - A_v_i)^-1
    else:
        precon_resid = preconditioner(residual, active_mask, A, A_w)
   #     print("precon_resid matrix\n", precon_resid)
    ## normalize and add to search subspace if they're larger than a threshold
        for i in range(no_eigs):
            to_append = precon_resid[:,i]
            #print("Preconditioned residual ",i, to_append)
            if np.linalg.norm(to_append) > conv:
                #print("Norm larger than conv. Appending. B: ")
                precon_resid[:,i] = precon_resid[:,i]/np.linalg.norm(to_append)
                B = np.append(B, precon_resid[:,i])
                nl += 1
                B = B.reshape(nl,N).T
                #print(B)
    #print("nl\n", nl)
    #print("new B matrix\n", B)
        
    print("converged = ", converged)
    count += 1 
   
