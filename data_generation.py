import random
import numpy as np
from numpy.linalg import det
from scipy.optimize import line_search
from math_structures import create_polynomial, create_gradient, hessian, newton_method, quasi_newton_method, appriximated_inv_hessian
from data_transformation import flatten_array

# генерация слуайных коэффициентов для многочлена n-й степени, n <= 5
def generate_coefficients(n, dim):
    c = random.randint(-100, 100)
    result = [c]

    def generate_a(n):
        limit = 5
        coefficients = []

        for i in range(limit):
            if i < n:
                coefficients.append(random.randint(-100, 100))
            else:
                coefficients.append(0)

        return coefficients

    for i in range(dim):
        result.append(generate_a(n))

    return result

# обертка для фильтрации ошибок сходимости
def linear_search(*args):
    alpha = line_search(*args)[0]

    if alpha is None:
        return -1

    if alpha >= 1024:
        alpha = -1

    return alpha


def load_data(dim, A, B, method):
    n_train = 50000
    n = 60000
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i in range(n):
        coefficients = generate_coefficients(random.randint(1, 5), dim)

        def generate_alpha():

            obj_func = create_polynomial(*coefficients)
            obj_grad = create_gradient(*coefficients)

            global start_point
            start_point = [random.randint(A, B) / 10 + random.randint(A, B) for _ in range(dim)]

            if method == "gradient":
                dir = np.array(obj_grad(start_point)) * -1
            elif method == "newton":
                matrix = hessian(start_point, *coefficients[1:])

                if det(matrix) == 0:
                    return -1

                dir = newton_method(obj_grad, matrix, start_point)

            elif method == "quasi_newton":
                # matrix = hessian(start_point, *coefficients[1:])
                matrix = appriximated_inv_hessian(obj_func, start_point)
                matrix = matrix * -1

                if det(matrix) == 0:
                    return -1

                dir = np.matmul(matrix, obj_grad(start_point))

            elif method == "mixed":
                if i < n_train / 2 or n_train < i <= (n_train + n) / 2:
                    dir = np.array(obj_grad(start_point)) * -1
                elif i >= n_train / 2 or n_train <= i > (n_train + n) / 2:
                    matrix = hessian(start_point, *coefficients[1:])
                    if det(matrix) == 0:
                        return -1
                    dir = newton_method(obj_grad, matrix, start_point)


            a = linear_search(obj_func, obj_grad, start_point, dir)
            return a

        alpha = generate_alpha()

        # фильтрация выборки от ошибочных значений
        while alpha == -1:
            coefficients = generate_coefficients(random.randint(1, 5), dim)
            alpha = generate_alpha()
        # фильтрация выборки от повторяющихся значений
        while flatten_array(coefficients) in x_train or flatten_array(coefficients) in x_test:
            coefficients = generate_coefficients(random.randint(1, 5), dim)
            alpha = generate_alpha()

        coefficients.append(start_point)

        if (i >= n_train):
            x_test.append(flatten_array(coefficients))
            y_test.append(alpha)
        else:
            x_train.append(flatten_array(coefficients))
            y_train.append(alpha)

    return (x_train, y_train), (x_test, y_test)
