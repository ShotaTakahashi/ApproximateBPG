import numpy as np
from src.algorithms.iteration import FOM


class BPG(FOM):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, stop, csv_path)
        self.lsmad = lsmad
        self.subprob = subprob
        self.rho = 0.99
        self.bregman_dist = bregman_dist

    def update(self, x, y):
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

    def update(self, x, y):
        grad = self.grad(y)
        dk = self.subprob(y, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x)
        tk = self.t_init
        while self.obj(x + tk*dk) > obj_xk + self.alpha*tk*const:
            tk *= self.eta
        return x + tk*dk

class ABPGVMAW(ApproxBPG):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, dist, grad_reg, c1=0.99, c2=0.999, eta=0.9, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, c1, eta, stop, csv_path)
        self.t_init = 1.0
        self.dist = dist
        self.grad_reg = grad_reg
        self.c2 = c2

    def update(self, x, y):
        grad = self.grad(y)
        dk = self.subprob(y, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        grad_reg = self.grad_reg(x)

        dist = self.dist(x, dk)*self.lsmad
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x)
        alpha_new = 1.0
        alpha_old = 0.0
        if self.obj(x + alpha_new*dk)  >= obj_xk + self.alpha*alpha_new*(const + dist):
            while self.obj(x + alpha_new*dk)  >= obj_xk + self.alpha*alpha_new*(const + dist):
                alpha_old = alpha_new
                alpha_new = 0.9*alpha_new
        else:
            while self.obj(x + alpha_new*dk)  < obj_xk + self.alpha*alpha_new*(const + dist):
                alpha_old = alpha_new
                alpha_new = 2*alpha_new

        alpha = min(alpha_old,alpha_new)
        beta = max(alpha_new,alpha_old)
        tk = (alpha + beta)/2
        max_iter = 1000
        for i in range(max_iter):
            if self.obj(x + tk*dk)  >= obj_xk + self.alpha*tk*(const + dist):
                beta = tk
            elif (self.grad(x + tk*dk) + grad_reg).dot(dk) <= self.c2*(grad + grad_reg).dot(dk):
                alpha = tk
            else:
                break
            tk = (alpha + beta)*0.5
        if self.obj(x + dk) < self.obj(x + tk*dk):
            return x + dk
        else:
            return x + tk*dk
        
