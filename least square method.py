import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def real_func(x_point):
    return np.sin(2 * np.pi * x_point)


def fit_func(p, x_point):
    f = np.poly1d(p)
    return f(x_point)


def residuals_func(p, x_point, y_point):
    ret = fit_func(p, x_point) - y_point
    return ret


x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
y_ = real_func(x)
y = y_ + np.random.normal(0, 0.1, 10)  # add noise


def fitting(m=0):
    p_init = np.random.rand(m + 1)
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bx', label='noise')
    plt.legend()
    plt.show()
    return p_lsq


p_lsq_0 = fitting(m=0)
p_lsq_1 = fitting(m=1)
p_lsq_3 = fitting(m=3)
p_lsq_9 = fitting(m=9)

# via regularization in order to omit the influence in last case
"""
L_1: regularization * abs(p)
L_2: 0.5 * regularization * np.square(p)
"""
regularization = 0.0001


def residuals_func_regularization(p, x_point, y_point):
    ret = fit_func(p, x_point) - y_point
    ret = np.append(ret, np.sqrt(0.5 * regularization * np.square(p)))
    return ret


def fitting_regularization(m=0):
    p_init = np.random.rand(m+1)
    p_lsq = leastsq(residuals_func_regularization, p_init, args=(x, y))
    return p_lsq


p_lsq_p_regularization = fitting_regularization(9)
plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x, y, 'bo', label='noise')
plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
plt.plot(x_points, fit_func(p_lsq_p_regularization[0], x_points), label='regularization')
plt.legend()
plt.show()
