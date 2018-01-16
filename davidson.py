# Generalized Davidson solver

import numpy as np

from psi4 import core

np.set_printoptions(precision=4, suppress=True)

def preconditioner(residual, active_mask, A, A_w):
    precon_resid = np.zeros_like(residual)
    diag = np.diagonal(A)
    for x in range(len(active_mask)):
        for i in range(N):
            if np.absolute(diag[i] - A_w[x]) < 1.0E-8:
                precon_resid[i,x] = residual[i,x]
            else:
                precon_resid[i,x] = residual[i,x]* (1.0/(diag[i] - A_w[x])) 
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

N = 100
no_eigs = 10
nl = 10
conv = 1.0E-5
# A = np.random.random_integers(-200,200,size=(N,N))
randA = np.random.RandomState(13)
A = randA.randn(N,N)
A = (A + A.T)/2

# print("A\n",A)

# np.savetxt("matrix.txt", A, fmt='%8.5f')

ans = np.linalg.eig(A)[0]
ans = np.sort(ans)
np.savetxt("ans.txt", ans, fmt='%8.5f')
#print("Eigvals of A: \n",np.linalg.eig(A)[0])
print("Eigvals of A(sorted): \n", ans)

# build guess vector matrix B
## sort diagonal elements in ascending order, generate corresponding unit vectors
x = A.diagonal()
j = 0
B = np.zeros_like(A)

print(x.argsort())
for i in x.argsort():
    B[i, j] = 1
    j += 1

#for i in range(nl):
#    B[i,j] = 1
#    j += 1

B = B[::, :no_eigs]
print("B\n", B)

converged=False
count = 0
### begin loop
while count<3:
    active_mask = [True for x in range(no_eigs)]
    print("Iteration number: ", count, "\n")
    # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
    ## orthogonalize preconditioned residuals against all other vectors in the search subspace
    B, r = np.linalg.qr(B)
    #Blen = B.shape
    # print("B\n",Blen,"\n", B)

    # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
    sigma = np.zeros((nl, N))

    sigma = np.dot(A,B)

    #print("sigma\n", sigma)
    # compute subspace matrix A_b = Btranspose sigma
    A_b = np.dot(B.T, sigma)
    sigma = sigma[::, :nl]

    #print("Subspace matrix \n", A_b)
    # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
    A_w, A_v = np.linalg.eig(A_b)

    #print("A_w\n", A_w, "\nA_v\n", A_v)
    A_v = A_v[:, A_w.argsort()]
    A_v = A_v[::,:no_eigs]
    A_w = A_w[A_w.argsort()]
    A_w = A_w[:no_eigs]
    print("A_w (sorted)\n", A_w, "\nA_v (sorted)\n", A_v)
    # here, check if no residuals > max no residuals, if so, collapse subspace

    if B.shape[1] >= N-(2*no_eigs):
        print("Subspace too big. Collapsing.\n")
        B = np.dot(B, A_v)
        nl = no_eigs
        continue
    # else, build residual matrix
    ## residual_i = sigma * eigvec - eigval * B * eigvec
    residual = np.zeros((N, no_eigs))
    for i in range(no_eigs):
        #mat = A - A_w[i] * np.identity(N) 
        #residual[:,i] = np.dot(np.dot(mat, B), A_v[:,i]) 
        residual[:,i] = np.dot(sigma, A_v[:,i]) - (A_w[i] * np.dot(B, A_v[:,i])) 

    #res_s = residual.shape
    #print("Shape of residual matrix:",res_s, "\n")
    print("residual matrix:\n", residual)
    for i in range(no_eigs):
        if np.linalg.norm(residual[:,i]) < 0.1:
            print("Inside this LOOP\n")
            residual = np.delete(residual, i, axis=1)
            active_mask = np.delete(active_mask, i)
    #res_s = residual.shape
    #print("Shape of residual matrix:",res_s, "\n")

    ## check for convergence by norm of residuals
    # norm = np.mean(residual, axis=0)**2
    norm = np.zeros(no_eigs)
    for i in range(no_eigs):
        norm[i] = np.linalg.norm(residual[:,i])
    #if(np.allclose(norm,norm1)):
    #    print("These two are the same.\n")
    #print("Norm\n", norm)
    print("Norm\n", norm)
    if(norm.all() < conv):
        converged = True
        break
    ## else apply the preconditioner (A_ii - A_v_i)^-1
    else:
        precon_resid = preconditioner(residual, active_mask, A, A_w)
        #print("precon_resid matrix\n", precon_resid)
    ## normalize and add to search subspace if they're larger than a threshold
        for i in range(no_eigs):
            to_append = precon_resid[:,i]
            #print("Preconditioned residual ",i, precon_resid[:,i])
            if np.linalg.norm(to_append) > 0.1:
                #print("Norm larger than conv. Appending. B before: ")
                #print(B)
                #print("norm of precon_resid to append: \n", np.linalg.norm(to_append))
                to_append = precon_resid[:,i] * (1.0/np.linalg.norm(to_append))
                #print("precon_resid to append: \n", to_append)
                B = np.append(B, to_append)
                nl += 1
                B = B.reshape(N,nl)
                #print("B after:\n",B)
    print("nl\n", nl)
    #print("new B matrix\n", B)

        
    print("converged = ", converged)
    count += 1 
   
