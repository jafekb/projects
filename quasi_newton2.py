# quasi_newton.py
"""Volume 2: Quasi-Newton Methods
<Benjamin Jafek
<MATH 323> 
<03/02/17>
"""

import numpy as np
import time
import scipy.optimize as opt
from matplotlib import pyplot as plt
from scipy.optimize import leastsq

#Problem 1 -- DONE
def newton_ND(J, H, x0, niter=10, tol=1e-5):
    """
    Perform Newton's method in N dimensions.

    Inputs:
        J (function): Jacobian of the function f for which we are finding roots.
        H (function): Hessian of f.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criterion for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
    x = x0
    for i in xrange(niter):
        xnext = x - np.linalg.solve(H(x),J(x))
        if np.linalg.norm(xnext - x) < tol:
            return xnext, i+1
        x = xnext
    return "Try more iterations to reach the minimum"

#Problem 2 -- DONE
def broyden_ND(J, H, x0, niter=20, tol=1e-5):
    """
    Perform Broyden's method in N dimensions.

    Inputs:
        J (function): Jacobian of the function f for which we are finding roots.
        H (function): Hessian of f.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
    A = H(x0)
    x = x0
    for i in xrange(niter):
        xnext = x - np.linalg.solve(A,J(x))
        s = xnext - x
        y = J(xnext).T - J(x).T
        Anext = A + np.outer((y-np.dot(A,s))/np.linalg.norm(s)**2, s.T)
        if np.linalg.norm(xnext-x) < tol:
            return xnext, i+1
        A = Anext
        x = xnext
    return "Needs more iterations to reach the minimum"

#Problem 3 -- DONE
def BFGS(J, H, x0, niter=10, tol=1e-6):
    """
    Perform BFGS in N dimensions.

    Inputs:
        J (function): Jacobian of objective function.
        H (function): Hessian of objective function.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.
        tol (float): Stopping criteria for iterations.

    Returns:
        The approximated root and the number of iterations it took.
    """
    x = x0
    A = H(x0)
    for i in xrange(niter):
        xnext = x - np.dot(np.linalg.inv(H(x)), J(x).T)
        s = xnext - x
        y = J(xnext).T - J(x).T
        #AssTA = np.outer(np.dot(A,s),np.dot(s.T,A))
        #sTAs = float(np.dot(s.T, np.dot(A, s)))
        #Anext = A + np.dot(y,y.T)/float(np.dot(y.T,s)) - AssTA/sTAs
        #below is the condensed formula. It's broken up above
        Anext = A + np.outer(y,y.T)/float(np.dot(y.T,s)) - np.outer(np.dot(A,s),np.dot(s.T,A))/float(np.dot(s.T, np.dot(A, s)))
        if np.linalg.norm(xnext-x) < tol:
            return xnext, i+1
        A = Anext
        x = xnext
    return "You need more iterations to find the minimum"

#Problem 4 -- DONE
def prob4():
    """
    Compare the performance of Newton's, Broyden's, and modified Broyden's
    methods on the following functions:
        f(x,y) = 0.26(x^2 + y^2) - 0.48xy
        f(x,y) = sin(x + y) + (x - y)^2 - 1.5x + 2.5y + 1
    """
    f = lambda x: .26(x[0]**2 + x[1]**2) - .48*x[0]*x[1]
    print "\nf(x,y) = 0.26(x^2 + y^2) - 0.48xy"
    J1 = lambda x: np.array([.52*x[0]-.48*x[1] , .52*x[1]-.48*x[0]])
    H1 = lambda x: np.array([[.52, -.48],[-.48,.52]])
    x0 = np.array([1.,1.])
    time1 = time.clock()
    x, niter = newton_ND(J1,H1,x0)
    time2 = time.clock()
    x, biter = broyden_ND(J1,H1,x0)
    time3 = time.clock()
    x, bfiter = BFGS(J1,H1,x0)
    time4 = time.clock()
    print "Newton:\t\t" + str(round(time2-time1,6)) + " seconds\n\t\t" + str(niter) + " iterations"
    print "Broyden:\t" + str(round(time3-time2,6)) + " seconds\n\t\t" + str(biter) + " iterations"
    print "BFGS:\t\t" + str(round(time4-time3,6)) + " seconds\n\t\t" + str(bfiter) + " iterations\n"

    g = lambda x: np.sin(x[0]+x[1]) + (x[0]-x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1
    print "f(x,y) = sin(x+y) + (x-y)^2 - 1.5x + 2.5y + 1"
    J2 = lambda x: np.array([np.cos(x[0]+x[1])+2*x[0]-2*x[1]-1.5 , np.cos(x[0]+x[1])-2*x[0]+2*x[1]+2.5])
    H2 = lambda x: np.array([ [2-np.sin(x[0]+x[1]), -1.*np.sin(x[0]+x[1])-2] , [-1.*np.sin(x[0]+x[1])-2, 2-np.sin(x[0]+x[1])] ])
    x0 = np.array([1.,1.])
    time1 = time.clock()
    x, niter = newton_ND(J2,H2,x0)
    time2 = time.clock()
    x, biter = broyden_ND(J2,H2,x0)
    time3 = time.clock()
    x, bfiter = BFGS(J2,H2,x0)
    time4 = time.clock()
    print "Newton:\t\t" + str(round(time2-time1,6)) + " seconds\n\t\t" + str(niter) + " iterations"
    print "\t\tTime per iteration: " + str(round((time2-time1)/niter,6)) + " seconds"
    print "Broyden:\t" + str(round(time3-time2,6)) + " seconds\n\t\t" + str(biter) + " iterations"
    print "\t\tTime per iteration: " + str(round((time3-time2)/biter,6)) + " seconds"
    print "BFGS:\t\t" + str(round(time4-time3,6)) + " seconds\n\t\t" + str(bfiter) + " iterations"
    print "\t\tTime per iteration: " + str(round((time4-time3)/bfiter,6)) + " seconds"

#Problem 5 -- DONE
def gauss_newton(J, r, x0, niter=10):
    """
    Solve a nonlinear least squares problem with Gauss-Newton method.

    Inputs:
        J (function): Jacobian of the objective function.
        r (function): Residual vector.
        x0 (float): The initial guess.
        niter (int): Number of iterations to compute.

    Returns:
        The approximated root.
    """
    x = x0
    for i in xrange(niter):
        xnext = x - np.linalg.solve(np.dot(J(x).T,J(x)), np.dot(J(x).T, r(x)))
    return xnext

def test_prob5():
    t = np.arange(10)
    y = 3*np.sin(.5*t)+.5*np.random.randn(10)

    def model(x,t):
        return x[0]*np.sin(x[1]*t)
    def residual(x):
        return model(x,t)-y
    def jac(x):
        ans = np.empty((10,2))
        ans[:,0]=np.sin(x[1]*t)
        ans[:,1]=x[0]*t*np.cos(x[1]*t)
        return ans
    def objective(x):
        return .5*(residual(x)**2).sum()
    def grad(x):
        return jac(x).T.dot(residual(x))
    x0 = np.array([2.5,.6])
    x = gauss_newton(jac,residual,x0)

    dom = np.linspace(0,10,100)
    plt.plot(t,y,'*', label='Perturbed points')
    plt.plot(dom, 3*np.sin(.5*dom), '--', label='Sine curve')
    plt.plot(dom, x[0]*np.sin(x[1]*dom), label='Gauss-Newton approximation')

    plt.legend(loc='lower left')
    plt.show()

def prob6():
    """
    Compare the least squares regression with 8 years of population data and 16
    years of population data.
    """
    #First 8 decades
    years1 = np.arange(8)
    pop1 = np.array([3.929, 5.308, 7.24, 9.638, 12.866, 17.069, 23.192, 31.443])

    #For the first 8 decades
    x0 = np.array([150,.4,2.5])
    t = years1
    def model1(x, t):
        return x[0]*np.exp(x[1]*(t+x[2]))
    def residual1(x):
        return model1(x,t)-pop1
    x1 = leastsq(residual1, x0)[0]

    plt.plot(years1, pop1, 'r*', label="Original")
    plt.plot(years1, model1(x1,t), 'b-', label="Fitted")
    plt.legend(loc="upper left")
    plt.xlabel("Decade (from 1790)")
    plt.ylabel("Population (millions)")
    plt.title("Population of the United States")
    plt.show()

    #first 16 decades
    years2 = np.arange(16)
    pop2 = np.array([3.929, 5.308, 7.240, 9.638, 12.866, 17.069, 23.192, 31.443, 38.558, 50.156, 62.948, 75.996, 91.972, 105.711, 122.775, 131.669])

    t = years2
    def model2(x, t):
        return x[0]/(1+np.exp(-1.*x[1]*(t+x[2])))
    def residual2(x):
        return model2(x,t)-pop2
    x2 = leastsq(residual2, x0)[0]

    plt.plot(years2, pop2, 'r*', label="Original")
    plt.plot(years2, model2(x2,t), 'b-', label="Fitted")
    plt.legend(loc="upper left")
    plt.xlabel("Decade (from 1790)")
    plt.ylabel("Population (millions)")
    plt.title("Population of the United States")
    plt.show()

if __name__ == "__main__":
    pass #Functions can be uncommented as necessary.
    """Problem 1"""
    # Jrosen = opt.rosen_der
    # Hrosen = opt.rosen_hess
    # xstart1 = np.array([-2.,2.])
    # xstart2 = np.array([10.,-10.])
    # print newton_ND(Jrosen,Hrosen,xstart1)
    # print newton_ND(Jrosen,Hrosen,xstart2)

    """Problem 2"""
    # f = lambda x: np.exp(x[0]-1)+np.exp(1-x[1])+(x[0]-x[1])**2
    # df = lambda x: np.array([2*x[0]+np.exp(x[0]-1)-2*x[1] , -2*x[0]-np.exp(1-x[1])+2*x[1]])
    # ddf = lambda x: np.array([ [np.exp(x[0]-1)+2, -2] , [-2, np.exp(1-x[1])+2] ])
    # xstart3 = np.array([2.,3.])
    # xstart4 = np.array([3.,2.])
    # print broyden_ND(df,ddf,xstart3)
    # print broyden_ND(df,ddf,xstart4)

    """Problem 3 (requires Problem 2)"""
    # print BFGS(df,ddf,xstart3)
    # print BFGS(df,ddf,xstart4)

    """Problem 4"""
    # prob4()

    """Problem 5"""
    # test_prob5()

    """Problem 6"""
    # prob6()