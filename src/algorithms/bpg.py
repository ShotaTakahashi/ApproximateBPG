import numpy as np
from src.algorithms.iteration import Algorithm


class BPG(Algorithm):
    def __init__(self, x0, lsmad, grad, subprob_sol, bregman_dist, stop='diff', write=False):
        super().__init__(x0, stop, write)
        self.lsmad = lsmad
        self.grad = grad
        self.subprob_sol = subprob_sol
        self.rho = 0.99
        self.bregman_dist = bregman_dist

    def update(self, x):
        grad = self.grad(x)
        xk = self.subprob_sol(x, grad, self.lsmad)
        return xk

    def adaptive_scheme(self):
        return self.bregman_dist(self.xk, self.yk) \
            > self.rho * self.bregman_dist(self.xk_1, self.xk)


class ApproxBPG(BPG):
    def __init__(self, x0, lsmad, grad, subprob_sol, bregman_dist, obj, reg, alpha=0.99, eta=0.9, stop='diff', write=False):
        super().__init__(x0, lsmad, grad, subprob_sol, bregman_dist, stop, write)
        self.obj = obj
        self.reg = reg
        self.t_init = 1.0
        self.alpha = alpha
        self.eta = eta

    def update(self, x):
        grad = self.grad(x)
        dk = self.subprob_sol(x, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x)
        tk = self.t_init
        while self.obj(x + tk*dk) > obj_xk + self.alpha*tk*const:
            tk *= self.eta
        return x + tk*dk
