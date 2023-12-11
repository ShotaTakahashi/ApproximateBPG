import numpy as np
import numpy.random as rnd
import numpy.linalg as LA


def soft_thresholding(x, tau):
    return np.maximum(np.abs(x) - tau, 0.0)*np.sign(x)


def power_method(A, AT, b, d):
    v = rnd.randn(d)
    v = v / LA.norm(v)
    for _ in range(50):
        u = AT(b * A(v))
        v = u / LA.norm(u)
    return v


def cardano(p, q):
    disc = np.sqrt(q**2*0.25 + p**3/27)
    ans = np.cbrt(-q*0.5 + disc) + np.cbrt(-q*0.5 - disc)
    return ans


def cubic_eq(u):
    norm = LA.norm(u) ** 2
    if norm == 0:
        return 1
    return cardano(1 / norm, -1 / norm)


def constrained_linear_system(inv, a, b):
    if len(inv.shape) > 1:
        inv_a = inv.dot(a)
        inv_b = inv.dot(b)
        beta = -a.dot(inv_a)
        return - inv_b - a.dot(inv_b)/beta * inv_a
    else:
        inv_a = inv*a
        inv_b = inv*b
        beta = -a.dot(inv_a)
        return - inv_b - a.dot(inv_b) / beta * inv_a


class Bregman:
    def __init__(self, kernel, grad):
        print('Define the Bregman distance.')
        self.kernel = kernel
        self.grad = grad

    def dist(self, x, y):
        return self.kernel(x) - self.kernel(y) - np.dot(self.grad(y), x - y)
