import numpy as np
import numpy.linalg.linalg
from scipy.optimize import line_search, minimize


def create_polynomial(c, *args):
    def vector(x):
        return np.array([x, x ** 2, x ** 3, x ** 4, x ** 5])

    def polynomial(x):
        out = c
        for i in range(len(args)):
            out += np.matmul(np.array(args[i]), vector(x[i]))
        return out

    return polynomial

def create_gradient(c, *args):
    def d_vector(x):
        return np.array([1, 2 * x, 3 * x ** 2, 4 * x ** 3, 5 * x ** 4])

    def gradient(x):
        out = []
        for i in range(len(args)):
            out.append(np.matmul(np.array(args[i]), d_vector(x[i])))
        return out

    return gradient


def create_2derivative(a):
    def d_vector(x):
        return np.array([0, 2, 6 * x, 12 * x ** 2, 20 * x ** 3])

    def derivative(x):
        return np.matmul(np.array(a), d_vector(x))

    return derivative


def gessian(x, *a):
    dim = len(a)
    df = [create_2derivative(a[i]) for i in range(dim)]
    matrix = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            if i == j:
                matrix[i][j] = df[i](x[i])

    return matrix

def appriximated_inv_gessian(func, x0):
    result = minimize(func, x0, method="BFGS")

    return result.hess_inv

def newton_method(grad, gesse_matrix, start_point):
    matrix = np.array(gesse_matrix) * -1
    inversed = np.linalg.inv(matrix)
    dir = np.matmul(inversed, grad(start_point))

    return dir

def quasi_newton_method(func, grad, start_point):
    inversed = appriximated_inv_gessian(func, start_point)

    inversed = inversed * -1

    direction = np.matmul(inversed, grad(start_point))
    return direction



