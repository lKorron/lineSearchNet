import numpy as np
import numpy.linalg.linalg
from scipy.optimize import line_search, minimize
from numpy.linalg import det

dim = 3

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



matrix = gessian([1, 1, 5], [0, 2, 2, 0, 0], [1, 2, 0, 0, 0], [2, 2, 1, 0, 0])

matrix = np.array(matrix) * -1

inversed = np.linalg.inv(matrix)

func = create_polynomial(1, [0, 2, 2, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0])
grad = create_gradient(1, [0, 2, 2, 0, 0], [1, 2, 0, 0, 0], [2, 2, 0, 0, 0])
start_point = [0, 0, 0]

search_grad = np.array(grad(start_point)) * -1

dir = np.matmul(inversed, grad(start_point))
print(dir)


inv = appriximated_inv_gessian(func, start_point)

dir2 = quasi_newton_method(func, grad, start_point)

print(dir2)

print(type(dir2[0]))

res = line_search(func, grad, start_point, dir)




x_1 = start_point + res[0] * search_grad



def is_wolfe1(alpha, func, grad, start_point, dir):
    c1 = 1e-3

    x_new = start_point + alpha * dir

    if func(x_new) <= func(start_point) + c1 * alpha * np.dot(grad(start_point), dir):
        return True
    else:
        return False

def is_wolfe2(alpha, func, grad, start_point, dir):
    c2 = 0.8

    x_new = start_point + alpha * dir


    if np.abs(np.dot(grad(x_new), dir)) <= np.abs(c2 * np.dot(grad(start_point), dir)):
        return True
    else:
        return False






print(is_wolfe1(res[0], func, grad, start_point, dir))
print(is_wolfe2(res[0], func, grad, start_point, dir))

f = create_polynomial(0, [2, 2, 0, 0, 0], [2, 2, 0, 0, 0])
g = create_gradient(0, [2, 2, 0, 0, 0], [2, 2, 0, 0, 0])
point = [0, 0]
d = np.array(g(point)) * -1

# print(d)
#
# print(is_wolfe1(3, f, g, point, d))
# print(is_wolfe2(0.08, f, g, point, d))
