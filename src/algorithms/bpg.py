import numpy as np
from src.algorithms.iteration import FOM


class BPG(FOM):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, stop, csv_path)
        self.lsmad = lsmad
        self.subprob = subprob
        self.rho = 0.99
        self.bregman_dist = bregman_dist

    def update(self, x):
        grad = self.grad(x)
        xk = self.subprob(x, grad, self.lsmad)
        return xk

    def adaptive_scheme(self):
        return self.bregman_dist(self.xk, self.yk) \
            > self.rho * self.bregman_dist(self.xk_old, self.xk)


class ApproxBPG(BPG):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, alpha=0.99, eta=0.9, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, lsmad, subprob, bregman_dist, stop, csv_path)
        self.reg = reg
        self.t_init = 1.0
        self.alpha = alpha
        self.eta = eta

    def update(self, x):
        grad = self.grad(x)
        dk = self.subprob(x, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x)
        tk = self.t_init
        while self.obj(x + tk*dk) > obj_xk + self.alpha*tk*const:
            tk *= self.eta
        return x + tk*dk
