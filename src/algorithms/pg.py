import numpy.linalg as LA
from src.algorithms.iteration import Algorithm


class PG(Algorithm):
    def __init__(self, x0, lsmooth, loss, grad, subprob_sol, linesearch=False, stop='diff', write=False):
        super().__init__(x0, stop, write)
        self.lsmooth = lsmooth
        self.loss = loss
        self.grad = grad
        self.subprob_sol = subprob_sol
        self.linesearch = linesearch
        self.rho = 0.99
        self.eta = 2

    def update(self, x):
        grad = self.grad(x)
        loss_xk = self.loss(x)
        l = self.lsmooth
        xk = self.subprob_sol(x, grad, l)
        while self.linesearch and self.loss(xk) > loss_xk + grad.dot(xk - x) + l*LA.norm(xk - x)**2/2:
            l *= self.eta
            xk = self.subprob_sol(x, grad, l)
        self.lsmooth = l
        return xk
