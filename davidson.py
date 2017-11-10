# Generalized Davidson solver

def davidson_solver(mat, hx_function, preconditioner, guess=None, printer=None, printlvl=1, maxiter=100):
    """
    Solves for the lowest few eigenvalues and eigenvectors of the given real symmetric matrix

    Parameters
    -----------
    mat : psi4.core.Matrix object
        The matrix whose eigenvalues are sought
    hx_function : function
        Takes in a list of :py:class:`~psi4.core.Matrix` objects and a mask of active indices. Returns the Hessian-vector product.
    preconditioner : function
        Takes in a list of :py:class:`~psi4.core.Matrix` objects and a mask of active indices. Returns the preconditioned value.
    guess : list of :py:class:`~psi4.core.Matrix`
        Starting vectors, if None use a unit vector guess
    printlvl : int
        The level of printing provided by this function. 
    maxiter : int
        The maximum number of iterations this function will take.

    Returns
    -----------
    
    Notes
    -----------

    Examples
    -----------

    """

