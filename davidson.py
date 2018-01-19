# Generalized Davidson solver

import numpy as np

from psi4 import core

np.set_printoptions(precision=12, suppress=True)

def preconditioner(residual, x, A, A_w):
    precon_resid = np.zeros_like(residual)
    diag = np.diagonal(A)
    if np.absolute(A_w[x] - diag[x]) < 1E-6:
        precon_resid = residual
    else:
        precon_resid = residual / (A_w[x] - diag[x]) 
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

N = 500
no_eigs = 5
nli = 5
nl = nli
conv = 1.0E-10
d_tol = 1.0E-11
maxiter = 100
spf = 0.0001
# A = np.random.random_integers(-200,200,size=(N,N))
A = np.zeros((N,N))
randA = np.random.RandomState(13)
for i in range(0,N):
    A[i,i] = i + 1.0
A = A + randA.randn(N,N)* spf
A = (A + A.T)/2

# print("A\n",A)

# np.savetxt("matrix.txt", A, fmt='%8.5f')

ans = np.linalg.eig(A)[0]
ans = np.sort(ans)
np.savetxt("ans.txt", ans, fmt='%8.5f')
#print("Eigvals of A: \n",np.linalg.eig(A)[0])
#print("Eigvals of A(sorted): \n", ans)

# build guess vector matrix B
## sort diagonal elements in ascending order, generate corresponding unit vectors
x = A.diagonal()
j = 0
B = np.zeros_like(A)

#print(x.argsort())
for i in x.argsort():
    B[i, j] = 1
    j += 1

#for i in range(nl):
#    B[i,j] = 1
#    j += 1

#B = B[::, :no_eigs]
#print("B\n", B)

converged=False
count = 0
sub_count = nli
### begin loop
while count < maxiter:
    active_mask = [True for x in range(sub_count)]
    print("Iteration number: ", count, "\n")
    # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
    ## orthogonalize preconditioned residuals against all other vectors in the search subspace
    B, r = np.linalg.qr(B)
    Blen = B.shape
    print("B\n",Blen,"\n")

    # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
    sigma = np.zeros((N, nli))

    sigma = np.dot(A,B[:,:nl])
    slen = sigma.shape

    print("sigma shape\n", slen)
    # compute subspace matrix A_b = Btranspose sigma
    A_b = np.dot(B[:,:nl].T, sigma)
    ablen = A_b.shape
    #sigma = sigma[::, :nl]

    print("Subspace matrix shape\n", ablen)
    # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
    A_w, A_v = np.linalg.eig(A_b)

    #print("A_w\n", A_w, "\nA_v\n", A_v)
    A_v = A_v[:, A_w.argsort()]
    #A_v = A_v[::,:no_eigs]
    A_w = A_w[A_w.argsort()]
    A_w = A_w[:no_eigs]
    print("A_w (sorted)\n", A_w, "\nA_v (sorted)\n", A_v)
    # here, check if no residuals > max no residuals, if so, collapse subspace
    sub_count = A_v.shape[0]

    if sub_count >= N-(2*no_eigs):
        print("Subspace too big. Collapsing.\n")
        B = np.dot(B[:,:nl], A_v)
        nl = nli
        continue
    # else, build residual matrix
    ## residual_i = sigma * eigvec - eigval * B * eigvec
    #residual = np.zeros((N, no_eigs))
    norm = np.zeros(nli)
    for i in range(0, nli):
        mat = A - A_w[i] * np.identity(N) 
        residual = np.dot(mat, np.dot(B[:,:sub_count], A_v[:,i]))
        #residual[:,i] = np.dot(sigma, A_v[:,i]) - (A_w[i] * np.dot(B, A_v[:,i])) 

    ##apply the preconditioner (A_ii - A_v_i)^-1
        precon_resid = preconditioner(residual, i, A, A_w)
        #print("precon_resid matrix\n", precon_resid)

    ## normalize and add to search subspace if they're larger than a threshold
        #print("Preconditioned residual ",i, precon_resid[:,i])
        if np.linalg.norm(precon_resid) > d_tol:
            #print("Norm larger than conv. Appending. B before: ")
            #print(B)
            #print("norm of precon_resid to append: \n", np.linalg.norm(to_append))
            B[:,nl+1] = precon_resid/np.linalg.norm(precon_resid)
            nl += 1
        #print("B after:\n",B)
        ## check for convergence by norm of residuals
            print("precon_resid to append: \n", precon_resid)
        norm[i] = np.linalg.norm(precon_resid)
    check = norm < conv
    print("Norm\n", norm)
    if(check.all() == True):
        converged = True
        print("converged.")
        break
    print("nl\n", nl)
    #print("new B matrix\n", B)
    count += 1 

        
print("converged = ", converged)
   
