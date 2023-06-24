#FUNCTIONS
import math
import matplotlib.pyplot as plt
import pandas as pd
from cpsplines.mosek_functions.interval_constraints import IntConstraints
import numpy as np
from scipy.optimize import minimize

from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.fittings.fit_cpsplines import CPsplines
import mosek.fusion as mf


#Función para calcular los polinomios ortogonales de la misma forma que se hace en R
def poly(x, degree):
    xbar = np.mean(x)
    x = x - xbar

    # R: outer(x, 0L:degree, "^")
    X = x[:, None] ** np.arange(0, degree+1)

    #R: qr(X)$qr
    q, r = np.linalg.qr(X)

    #R: r * (row(r) == col(r))
    z = np.diag((np.diagonal(r)))  

    # R: Z = qr.qy(QR, z)
    Zq, Zr = np.linalg.qr(q)
    Z = np.matmul(Zq, z)

    # R: colSums(Z^2)
    norm1 = (Z**2).sum(0)

    #R: (colSums(x * Z^2)/norm2 + xbar)[1L:degree]
    alpha = ((x[:, None] * (Z**2)).sum(0) / norm1 +xbar)[0:degree]

    # R: c(1, norm2)
    norm2 = np.append(1, norm1)

    # R: Z/rep(sqrt(norm1), each = length(x))
    Z = Z / np.reshape(np.repeat(norm1**(1/2.0), repeats = x.size), (x.size, -1), order='F')

    #R: Z[, -1]
    Z = np.delete(Z, 0, axis=1)
    return [Z, alpha, norm2]



## ----------------------------------------------------------------------------------------------------------

# GCV 1
def simpleGCV(lambda_, phi, P, y):
    return np.prod(y.shape) * np.square(np.linalg.norm(y - phi[0] @ np.linalg.solve(phi[0].T @ phi[0] + lambda_* P[0],  phi[0].T @ y )))/np.square(
        np.prod(y.shape)
        - np.trace(np.linalg.solve(phi[0].T @ phi[0] + lambda_* P[0], phi[0].T @ phi[0] ))
        )

## ----------------------------------------------------------------------------------------------------------

# Choose lambda 1

def choose_lambda( obj_matrices, ftol) :
    lambda_0 = 1

    res_SLSQP = minimize(simpleGCV, lambda_0, args=(obj_matrices["B"], obj_matrices["D_mul"], obj_matrices["y"]), method='SLSQP', bounds = [(1e-10, 1e16)],  options={'disp': False, "ftol": ftol})
    lambda_ = res_SLSQP.x
    # print(lambda_)
    return lambda_

## ----------------------------------------------------------------------------------------------------------

##Integral calculations, only to check things are done well


def simpson_spline_full(X, beta , k,  initial_grid,  newgrid_size, lambda_ = 1):

    N = X.shape[0]
    #Obtain the spline matrices
    
    bspline = BsplineBasis(deg=3, xsample=initial_grid, n_int=k)
    bspline.get_matrix_B()
    phi = bspline.matrixB
    bspline.get_matrices_S()
    S = bspline.matrices_S


    pen = PenaltyMatrix(bspline=bspline)
    D = pen.get_diff_matrix()
    P = pen.get_penalty_matrix()
    assert np.allclose(D.T @ D, P)
    ##Evaluate phi in a sharper grid
    T2 = np.linspace(initial_grid.min(), initial_grid.max(), num=newgrid_size)

    phistar=bspline.bspline_basis(T2)
    #SIMSPON WEIGHTS
    w = [1/3] + [4/3 if i%2==0 else 2/3 for i in range(2,newgrid_size)] + [1/3]
    W = 1/newgrid_size * np.diag(w)


    #ESTIMATE THE COEFFICIENTS FOR X
    a_matrix= np.empty((N, k+3))
    for i in range(N):
        with mf.Model() as model:
            # Create the variables
            a = model.variable('a', k+3, mf.Domain.unbounded())
            u = model.variable('u',  mf.Domain.greaterThan(0))
            v = model.variable('v',  mf.Domain.greaterThan(0))

            exp = mf.Expr.vstack(1/2, u, mf.Expr.mul(phi, a))
            model.constraint(exp, mf.Domain.inRotatedQCone())


            #SQRT(LAMBDA)??
            exp2 = mf.Expr.vstack(1/2, v, mf.Expr.mul(math.sqrt(lambda_)*D, a))
            model.constraint(exp2, mf.Domain.inRotatedQCone())
            #  # Set up the objective function
            f = mf.Matrix.dense(-2 * X[[i],:] @ phi)
            obj = mf.Expr.add(mf.Expr.add(u, v), mf.Expr.mul(f,a))

            model.objective(mf.ObjectiveSense.Minimize, obj)

           # Add constraints (if any)
           # ...#

           # Solve the optimization problem
            model.solve()#

           # Get the solution
            a_sol = a.level()
            a_sol_reshaped = np.reshape(a_sol, [k+3])
            a_matrix[i,:]=a_sol

    ####################################
    #Estimation of coefficients for beta
    with mf.Model() as model:#
        # Create the variables
        a = model.variable('theta', k+3, mf.Domain.unbounded())
        u = model.variable('u',  mf.Domain.greaterThan(0))
        v = model.variable('v',  mf.Domain.greaterThan(0))

        exp = mf.Expr.vstack(1/2, u, mf.Expr.mul(phi, a))
        model.constraint(exp, mf.Domain.inRotatedQCone())


        #SQRT(LAMBDA)??
        exp2 = mf.Expr.vstack(1/2, v, mf.Expr.mul(math.sqrt(lambda_)*D, a))
        model.constraint(exp2, mf.Domain.inRotatedQCone())
        #  # Set up the objective function
        f = mf.Matrix.dense(-2 * np.matrix(beta) @ phi)
        obj = mf.Expr.add(mf.Expr.add(u, v), mf.Expr.mul(f,a))

        model.objective(mf.ObjectiveSense.Minimize, obj)

        #     Add constraints (if any)
        #     ...#

        #     Solve the optimization problem
        model.solve()#

        #     Get the solution
        a_solbeta = a.level()
        a_solbeta_reshaped = np.reshape(a_solbeta, [k+3,1])

        M = a_matrix @ phistar.T @ W
    return [M, M @ phistar @ a_solbeta_reshaped]





## ----------------------------------------------------------------------------------------------------------

#Obtain the matrices we are going to need for the problem

def simpson_spline_matrices(X, k,  initial_grid,  newgrid_size):
 

    N = X.shape[0]
    #Obtain the spline matrices
    
    bspline = BsplineBasis(deg=3, xsample=initial_grid, n_int=k)
    bspline.get_matrix_B()
    phi = bspline.matrixB
    
    pen = PenaltyMatrix(bspline=bspline)
    D=pen.get_diff_matrix()

    ##Evaluate phi in a sharper grid
    T2 = np.linspace(initial_grid.min(), initial_grid.max(), num=newgrid_size)

    phistar=bspline.bspline_basis(T2)

    bspline.get_matrices_S()
    S = bspline.matrices_S
    #SIMSPON WEIGHTS
    w = [1/3] + [4/3 if i%2==0 else 2/3 for i in range(2, newgrid_size)] + [1/3]
    W = 1/newgrid_size * np.diag(w)



    a_matrix= np.empty((N, k+3))
    for i in range(N):
        obj_matricess = dict( { "B": [phi.copy()],
                       "y": X[i, :],
                       "D_mul": [D.T.copy() @ D.copy()]})
        lambda_ = choose_lambda(obj_matricess, 1e-12)   
        with mf.Model() as model:
            # Create the variables
            a = model.variable('a', k+3, mf.Domain.unbounded())
            u = model.variable('u',  mf.Domain.greaterThan(0))
            v = model.variable('v',  mf.Domain.greaterThan(0))

            exp = mf.Expr.vstack(1/2, u, mf.Expr.mul(phi, a))
            model.constraint(exp, mf.Domain.inRotatedQCone())


            exp2 = mf.Expr.vstack(1/2, v, mf.Expr.mul(math.sqrt(lambda_)*D, a))
            model.constraint(exp2, mf.Domain.inRotatedQCone())
            #  # Set up the objective function
            f = mf.Matrix.dense(-2 * X[[i],:] @ phi)
            obj = mf.Expr.add(mf.Expr.add(u, v), mf.Expr.mul(f,a))

            model.objective(mf.ObjectiveSense.Minimize, obj)

           # Add constraints (if any)
           # ...#

           # Solve the optimization problem
            model.solve()#

           # Get the solution
            a_sol = a.level()
            a_sol_reshaped = np.reshape(a_sol, [k+3])
            
            a_matrix[i,:]=a_sol_reshaped

            actual= X[[i],:]
            actual= np.reshape(actual, [len(initial_grid)])
            # plt.figure()
            # plt.plot(initial_grid, actual, label=f'actual {i+1}')
            # plt.plot(T2, a_sol_reshaped @ phistar.T, label=f'approx {i+1}')
            # plt.legend()
            # plt.show()
            

            


    # M 

    M = a_matrix @ phistar.T @ W
    return M, phistar, D, S

## ----------------------------------------------------------------------------------------------------------

#GCV, adapted for the integrated problem

def intGCV(lambda_, Bstar, M,  D, y):
    MB = M @ Bstar[0]

    MB = np.hstack((np.ones((MB.shape[0], 1)), MB))
    P= D.T @ D
    P = np.hstack((np.zeros((P.shape[0], 1)), P))
    P = np.vstack((np.zeros((1, P.shape[1])), P))

    return np.prod(y.shape) * np.square(np.linalg.norm(y - MB @ np.linalg.solve(MB.T @ MB + lambda_* P,  MB.T @ y )))/np.square(
        np.prod(y.shape)
        - np.trace(np.linalg.solve(MB.T @ MB + lambda_* P, MB.T @ MB ))
        )

    # return np.prod(y.shape) * np.square(np.linalg.norm(y - papillarygorro(lambda_, Bstar[0], M, y, D )))/np.square(
    #     np.prod(y.shape)
    #     - np.trace(np.linalg.solve(MB.T @ MB + lambda_* P, MB.T @ MB ))
    #     )


## ----------------------------------------------------------------------------------------------------------

#Create objective matrices dictionary
def create_obj_mat(Bstar, M ,y, D ):
    return  dict({"Bstar": [Bstar.copy()],
                       "M": M.copy(),
                       "y": y.copy(),
                       "D": D.copy()})

## ----------------------------------------------------------------------------------------------------------

#Choose lambda, adapted for the integrated problem 

def choose_lambda_int(obj_matrices, ftol) :
    list_lambda_0 =  list( range(100,300, 10))
    min=100
    for lambda_0 in list_lambda_0:
        res_SLSQP = minimize(intGCV, lambda_0, args=(obj_matrices["Bstar"], obj_matrices["M"], obj_matrices["D"], obj_matrices["y"]), method='SLSQP', bounds = [(1e-10, 1e16)],  options={'disp': False, "ftol": ftol})
        if res_SLSQP.fun<min:
            lambda_ = res_SLSQP.x
            min = res_SLSQP.fun
            # print('min:', min)
            # print('lambda', lambda_)
    return lambda_
    
    # list_lambda_0 =[0.1, 1, 3, 5, 7, 10, 20, 30, 40,  75] + list( range(50,5000, 50))
    # lambdaindex = np.argmin([intGCV(lambda_0, obj_matrices["Bstar"], obj_matrices["M"], obj_matrices["D"], obj_matrices["y"]) for lambda_0 in list_lambda_0])
    # lambda_ = list_lambda_0[lambdaindex]
    # print('lambda', lambda_)
    # return lambda_











## ----------------------------------------------------------------------------------------------------------


#Perform the unconstrained fit

def unconstrained_sof_fit(X, y, k,  initial_grid,  newgrid_size, plot=True):

    M, Bstar, D, S= simpson_spline_matrices(X = X, k = k, initial_grid = initial_grid, newgrid_size= newgrid_size)
    
    obj_matrices1 = create_obj_mat(Bstar, M, y, D )
    #It is here because choose_lambda_int needs the previous D to extend it inside int_GCV
    lambda_ = choose_lambda_int(obj_matrices1, 1e-12)
    # print('El GCV con el lambda óptimo MÁS UNO es:', intGCV(lambda_+5, [Bstar], M, D.T @ D, y))

    D=np.hstack((np.zeros((D.shape[0], 1)), D))

    # print('Optimal smoothing parameter:', lambda_)

    yy = np.matrix(obj_matrices1["y"])

    C = obj_matrices1["M"] @ obj_matrices1["Bstar"][0]
    C= np.hstack((np.ones((C.shape[0], 1)), C))

    # Define the optimization problem
    with mf.Model() as model:#
        # Create the variables
        a = model.variable('theta', k+3+1, mf.Domain.unbounded())
        u = model.variable('u',  mf.Domain.greaterThan(0))
        v = model.variable('v',  mf.Domain.greaterThan(0))

        exp = mf.Expr.vstack(1/2, u, mf.Expr.mul(C, a))
        model.constraint(exp, mf.Domain.inRotatedQCone())

        exp2 = mf.Expr.vstack(1/2, v, mf.Expr.mul(math.sqrt(lambda_)*D, a))
        model.constraint(exp2, mf.Domain.inRotatedQCone())
        #  # Set up the objective function
        f = mf.Matrix.dense(-2 * yy @ C)
        obj = mf.Expr.add(mf.Expr.add(u, v), mf.Expr.mul(f,a))

        model.objective(mf.ObjectiveSense.Minimize, obj)

       # Add constraints (if any)
       # ...#

       # Solve the optimization problem
        model.solve()
       # Get the solution
        theta = a.level()


    theta_reshaped = np.reshape(theta, [k+3+1, 1])
    
    # np.allclose((C @ theta_reshaped).T,  y, atol=2)

    ##Create plots to compare what we have obtained
    actual= y
    actual= np.reshape(actual, [len(y)])

    approx= C @ theta_reshaped
    approx= np.reshape(approx, [len(y)])
    # print(sum(abs(actual-approx)))
    betagorro = obj_matrices1["Bstar"][0] @ theta[1:]
    if (plot==True):
        T= np.linspace(0,1,len(y))
        # create the plot
        plt.scatter(T, actual, label='actual')
        plt.scatter(T, approx, label='approx')
        plt.plot(T, approx, label='approx')

        #add labels and title
        plt.xlabel('T')
        plt.ylabel('Values')
        plt.title('Comparison of actual and approximated')
        plt.legend()
        plt.plot(T, y)
        #  display the plot
        plt.show()
    return approx, betagorro


## ----------------------------------------------------------------------------------------------------------


#Perform the constrained fit

def papillarygorro(lambda_, Bstar, M, y, D, k=30):

    obj_matrices1 = create_obj_mat(Bstar, M, y, D )

    # print('El GCV con el lambda óptimo MÁS UNO es:', intGCV(lambda_+5, [Bstar], M, D.T @ D, y))

    D=np.hstack((np.zeros((D.shape[0], 1)), D))

    print('Optimal smoothing parameter:', lambda_)

    yy = np.matrix(obj_matrices1["y"])

    C = obj_matrices1["M"] @ obj_matrices1["Bstar"][0]
    C= np.hstack((np.ones((C.shape[0], 1)), C))


    # Define the optimization problem
    with mf.Model() as model:#
        # Create the variables
        a = model.variable('theta', k+3+1, mf.Domain.unbounded())
        u = model.variable('u',  mf.Domain.greaterThan(0))
        v = model.variable('v',  mf.Domain.greaterThan(0))

        exp = mf.Expr.vstack(1/2, u, mf.Expr.mul(C, a))
        model.constraint(exp, mf.Domain.inRotatedQCone())

        exp2 = mf.Expr.vstack(1/2, v, mf.Expr.mul(math.sqrt(lambda_)*D, a))
        model.constraint(exp2, mf.Domain.inRotatedQCone())
        #  # Set up the objective function
        f = mf.Matrix.dense(-2 * yy @ C)
        obj = mf.Expr.add(mf.Expr.add(u, v), mf.Expr.mul(f,a))

        model.objective(mf.ObjectiveSense.Minimize, obj)

       # Add constraints (if any)
       # ...#

       # Solve the optimization problem
        model.solve()
       # Get the solution
        theta = a.level()


    theta_reshaped = np.reshape(theta, [k+3+1, 1])
    
    np.allclose((C @ theta_reshaped).T,  y, atol=2)

    ##Create plots to compare what we have obtained
    actual= y
    actual= np.reshape(actual, [len(y)])

    approx= C @ theta_reshaped
    approx= np.reshape(approx, [len(y)])
    # print(sum(abs(actual-approx)))
    theta_reshaped2 = np.reshape(theta[0:-1], [k+3, 1])
    betagorro = obj_matrices1["Bstar"][0] @ theta[1:]
    return approx


#derivative_order requires an integer (order 0, 1, 2...)
#sign requires a string "+" (the function (or some derivative) needs to be above some threshold)
## or "-" (the function (or some derivative) needs to be below some
#threshold), and the values are the thresholds.
#sign_threshold: the number to impose the restriction: greater than 0, 1...

def cp_sof_fit(X, y, k,  initial_grid,  newgrid_size, derivative_order, sign, sign_threshold, plot=True):

    bspline = BsplineBasis(deg=3, xsample=initial_grid, n_int=k)
    bspline.get_matrix_B()
    
    M, Bstar, D, S= simpson_spline_matrices(X = X, k = k, initial_grid = initial_grid, newgrid_size= newgrid_size)

    obj_matrices = create_obj_mat(Bstar, M, y, D )

    lambda_ = choose_lambda_int(obj_matrices, 1e-12)

    D=np.hstack((np.zeros((D.shape[0], 1)), D))

    # print('Optimal smoothing parameter:', lambda_)

    yy = np.matrix(obj_matrices["y"])

    C = obj_matrices["M"] @ obj_matrices["Bstar"][0]
    C= np.hstack((np.ones((C.shape[0], 1)), C))


    # Define the optimization problem
    with mf.Model() as model:#
        # Create the variables
        a = model.variable('theta', k+3+1, mf.Domain.unbounded())
        u = model.variable('u',  mf.Domain.greaterThan(0))
        v = model.variable('v',  mf.Domain.greaterThan(0))

        exp = mf.Expr.vstack(1/2, u, mf.Expr.mul(C, a))
        model.constraint(exp, mf.Domain.inRotatedQCone())


        #SQRT(LAMBDA)??
        exp2 = mf.Expr.vstack(1/2, v, mf.Expr.mul(math.sqrt(lambda_)*D, a))
        model.constraint(exp2, mf.Domain.inRotatedQCone())
        #  # Set up the objective function
        f = mf.Matrix.dense(-2 * yy @ C)
        obj = mf.Expr.add(mf.Expr.add(u, v), mf.Expr.mul(f,a))

        model.objective(mf.ObjectiveSense.Minimize, obj)

       # Add constraints 

        
        int_cons = IntConstraints(
            bspline=[bspline], var_name=0, derivative=derivative_order, constraints={sign: sign_threshold}
        )
        int_cons.interval_cons(
            model=model,
            matrices_S={0: S},
            var_dict={"theta": model.getVariable("theta").slice(1, k+3+1)},
        )
        
       # Solve the optimization problem
        model.solve()
       # Get the solution
        theta = a.level()
        approx_theta = np.dot(Bstar, theta[1:])

    theta_reshaped = np.reshape(theta, [k+3+1, 1])
    
    np.allclose((C @ theta_reshaped).T,  y, atol=2)

    ##Create plots to compare what we have obtained
    actual= y
    actual= np.reshape(actual, [len(y)])

    approx= C @ theta_reshaped
    approx= np.reshape(approx, [len(y)])

    theta_reshaped2 = np.reshape(theta[0:-1], [k+3, 1])
    betagorro = obj_matrices["Bstar"][0] @ theta[1:]

    if (plot==True):
        T= np.linspace(0,1,len(y))
        # create the plot
        plt.scatter(T, actual, label='actual')
        plt.scatter(T, approx, label='approx')

        #add labels and title
        plt.xlabel('T')
        plt.ylabel('Values')
        plt.title('Comparison of actual and approximated')
        plt.legend()
        #  display the plot
        plt.show()
    return approx, theta, betagorro
