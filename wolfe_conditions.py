import numpy as np


def is_wolfe1(alpha, func, grad, start_point, dir, c1):

    x_new = start_point + alpha * dir

    if func(x_new) <= func(start_point) + c1 * alpha * np.dot(grad(start_point), dir):
        return True
    else:
        return False


def is_wolfe2(alpha, grad, start_point, dir, c2):

    x_new = start_point + alpha * dir

    if np.abs(np.dot(grad(x_new), dir)) <= np.abs(c2 * np.dot(grad(start_point), dir)):
        return True
    else:
        return False


def is_wolfe(alpha, func, grad, start_point, dir, c1=1e-4, c2=0.9):
    if is_wolfe1(alpha, func, grad, start_point, dir, c1) and is_wolfe2(alpha, grad, start_point, dir, c2):
        return True
    else:
        return False
