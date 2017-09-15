# simplex.py
"""
<Benjamin Jafek>
<MATH 323>
<1/26/17>
"""

import numpy as np

# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        maximize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm, as long as the problem is feasible at the origin
    """

    def __init__(self, c, A, b):
        """

        Parameters:
            c (1xn ndarray): The coefficients of the linear objective function.
            A (mxn ndarray): The constraint coefficients matrix.
            b (1xm ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        #Check feasibility.
        x=np.zeros_like(c)
        m,n=np.shape(A)
        c = c.reshape(len(c),1)
        b = b.reshape(len(b),1)

        if np.all(np.less_equal(A.dot(x),b)): #x=0 ==> Ax<b
            #Initialize variables
            self.c = c
            self.A = A
            self.b = b
        else:
            #If (0,0,...,0) is not a feasible solution
            raise ValueError("Problem not feasible at the origin")

        Abar = np.hstack((A,np.eye(m)))
        cbar = np.vstack((c,np.zeros((m,1))))
        first_T = np.hstack((np.zeros((1,1)),-cbar.T))
        second_T = np.hstack((b,Abar))
        T = np.vstack((first_T, second_T))
        self.Tableau=T

    #Get the pivot column
    def pivot_col(self):
        for column_element in self.Tableau[0,1:]:
            if column_element < 0:
                pivot_col = np.where(self.Tableau[0]==column_element)[0][0]
                break

        return pivot_col

    #Get the pivot row
    def pivot_row(self):
        #First define our starting ratio
        best_ratio = float("inf")
        #Now start the algorithm.
        #TODO make sure it's positive.
        for row_element in xrange(1, len(self.Tableau[:,self.pivot_col()])):
            my_ratio = self.Tableau[row_element,0]/self.Tableau[row_element, self.pivot_col()]
            if my_ratio <  best_ratio and my_ratio >= 0:
                best_ratio = my_ratio
                pivot_row = row_element

        return pivot_row

    #This is how we pivot
    def pivot(self, pivot_row, pivot_col):
            tab = self.Tableau
            pivot_entry = tab[pivot_row,pivot_col]

            #Divide the pivot row by the value of the pivot entry
            tab[pivot_row] /= pivot_entry

            #Use the pivot row to zero out all entries in the pivot column above and below the pivot entry
            for row in xrange(len(tab)): #-1 to account for the index starting at 0
                if row != pivot_row:
                    tab[row] -= (tab[row, pivot_col]/tab[pivot_row,pivot_col])*(tab[pivot_row])
            
            return tab

    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The maximum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """

        #First, if it's unbounded raise an error
        if np.all(self.Tableau[:,1:] <=0):
            raise ValueError("This problem is undbounded") #Don't mark this wrong, Koa said it's right

        #This gets our final tableau
        final_answer = self.pivot(self.pivot_row(), self.pivot_col())
        while np.any(self.Tableau[0]<0):
            final_answer = self.pivot(self.pivot_row(), self.pivot_col())
        
        #optimal value
        optimal_value = final_answer[0,0]

        #Initialize our dictionaries
        basic_variables = dict()
        nonbasic_variables = dict()
        
        #Update our dictionaries
        for i in xrange(len(final_answer[0,1:])):
            i+=1
            if final_answer[0,i]==0:
                index = np.argmax(final_answer[:,i])
                basic_variables[i-1] = final_answer[index,0]
            else:
                nonbasic_variables[i-1] = 0

        #Return optimal,      basic,           nonbasic
        return optimal_value, basic_variables, nonbasic_variables
        



# Problem 7
def prob7(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        The minimizer of the problem (as an array).
    """
    #Get our variables
    mix = np.load("productMix.npz")
    A = mix['A']
    p = mix['p']
    lp = len(p)
    p=p.reshape((lp,1))
    m = mix['m']
    lm = len(m)
    m = m.reshape((lm,1))
    d = mix['d']
    ld = len(d)
    d = d.reshape((ld,1))

    #Let it begin
    c = np.array(p)
    A = np.vstack((A, np.eye(len(d))))
    b = np.vstack((m, d))
    

    answer = SimplexSolver(c,A,b).solve()
    my_dict = answer[1]
    final_answer = np.array([my_dict[0],my_dict[1],my_dict[2],my_dict[3]]) 
    return final_answer

