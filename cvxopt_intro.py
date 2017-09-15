# cvxopt_intro.py
"""Volume 2B: CVXOPT
<Benjamin Jafek>
<MATH 322>
<1/9/17>
"""
from __future__ import division
from cvxopt import matrix, solvers
import numpy as np
from scipy import linalg as la


def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x + 2y          >= 3
                    2x + 10y + 3z   >= 10
                    x               >= 0
                    y               >= 0
                    z               >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """

    #Define matrices
    c = matrix([2.,1.,3.])
    G = matrix([[-1.,-2.,-1.,0.,0.],[-2.,-10.,0.,-1.,0.],[0.,-3.,0.,0.,-1.]])
    h = matrix([-3.,-10.,0.,0.,0.])


    #suppress iteration output (opt)
    solvers.options['show_progress'] = False


    #Find the solution through convex optimization
    sol = solvers.lp(c,G,h)
    x = sol['x']
    objective = sol['primal objective']
    return np.ravel(x), np.ravel(objective)


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray), without any slack variables u
        The optimal value (sol['primal objective'])
    """
    #Make sure A and b use floats
    m,n=np.shape(A)
    A=np.array(A,dtype=float)
    b=np.array(b,dtype=float)

    #Define matrices we'll use
    minimizer = np.hstack((np.ones_like(b),np.zeros_like(b))).astype(float)

    subjector_one = np.hstack((-np.eye(n),np.eye(n)))
    subjector_two = np.hstack((-np.eye(n),-np.eye(n)))
    subjector = np.vstack((subjector_one,subjector_two)).astype(float)

    equality = np.hstack((np.zeros_like(A),A)).astype(float)

    #Define elements of the cvxopt
    c=matrix(minimizer)

    G=matrix(subjector)
    h=matrix(np.zeros(len(b)*2))

    A=matrix(equality)
    b=matrix(b)

    #Do it!
    sol = solvers.lp(c,G,h,A,b)

    #Return what we want
    best_x = sol['x'][n:]
    optimal = sol['primal objective']

    return np.ravel(best_x), np.ravel(optimal)


def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    """
    #Define some np.arrays
    Garray = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1],[1,0,1,0,1,0],[0,1,0,1,0,1]])
    harray = np.array([7,2,4,5,8])
    Gtot = np.vstack((Garray,-Garray))
    htot = np.hstack((harray,harray))

    #Define our cvx matrices
    c = matrix([4.,7.,6.,8.,8.,9.])
    G = matrix(Gtot)
    h = matrix(htot)
    """
    #Example
    c = matrix([4., 7., 6., 8., 8., 9])
    G = matrix(np.vstack((-1*np.eye(6),np.array([[0,1,0,1,0,1],[0,-1,0,-1,0,-1]]))))
    h = matrix([0.,0.,0.,0.,0.,0.,8.,8.])
    A = matrix(np.array([[1.,1.,0.,0.,0.,0.],[0.,0.,1.,1.,0.,0.],[0.,0.,0.,0.,1.,1.],[1.,0.,1.,0.,1.,0.]]))
    b = matrix([7., 2., 4., 5.])
    sol = solvers.lp(c, G, h, A, b)

    return np.ravel(sol['x']),np.ravel(sol['primal objective'])

def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    P = matrix(np.array([[3.,2.,1.],[2.,4.,2.],[1.,2.,3.]]))
    
    q = matrix(np.array([3.,0.,1.]))

    sol = solvers.qp(P,q)

    return np.ravel(sol['x']), np.ravel(sol['primal objective'])


# Problem 5
def l2Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_2
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (sol['primal objective'])
    """
    m, n = np.shape(A)

    #Lab spec logic
    P = matrix(2.*np.eye(n).astype(float))
    q = matrix(np.zeros(n).astype(float))


    #Convert A and b to cvx matrices
    A = matrix(A.astype(float))
    b = matrix(b.astype(float))


    #suppress iteration output (opt)
    solvers.options['show_progress'] = False

    sol = solvers.qp(P, q, A=A, b=b)
    return np.ravel(sol['x']),np.ravel(sol['primal objective'])



def prob6():
    """Solve the allocation model problem in 'ForestData.npy'.
    Note that the first three rows of the data correspond to the first
    analysis area, the second group of three rows correspond to the second
    analysis area, and so on.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (sol['primal objective']*-1000)
    """
    forest =  np.load("ForestData.npy")
    acres = forest[:,1]
    npv = forest[:,3] #We will maximize this
    timber = forest[:,4] #>=40,000
    grazing = forest[:,5] #>=5
    wilderness = forest[:,6] #/788 >=70
    
    #Minimize:
    c = -matrix(npv) 

    #Subject to:
    positivity_matrix = np.eye(21)
    
    G=matrix(np.vstack((-timber,-wilderness, -grazing, -positivity_matrix)))
    
    h=matrix(-1*np.array([40000.,70.*788,5.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]))
    
    A=matrix(np.array([[1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,0.,0.,0.],
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.]]))

    b=matrix([acres[0],acres[3],acres[6],acres[9],acres[12],acres[15],acres[18]])

    #Suspend output
    solvers.options['show_progress'] = False

    #Convex optimization
    sol = solvers.lp(c, G, h, A, b)
    allocation = sol['x']
    best_npv = sol['primal objective']*-1000

    return np.ravel(allocation), np.ravel(best_npv)
    

