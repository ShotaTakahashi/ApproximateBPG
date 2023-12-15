import numpy.linalg as LA
from src.algorithms.iteration import FOM


class PG(FOM):
    def __init__(self, x0, obj, grad, opt, lsmooth, subprob, loss, linesearch=False, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, stop, csv_path)
        self.lsmooth = lsmooth
        self.loss = loss
        self.grad = grad
        self.subprob = subprob
        self.linesearch = linesearch
        self.rho = 0.99
        self.eta = 2

    def update(self, x):
        grad = self.grad(x)
        loss_xk = self.loss(x)
        l = self.lsmooth
        xk = self.subprob(x, grad, l)
        while self.linesearch and self.loss(xk) > loss_xk + grad.dot(xk - x) + l*LA.norm(xk - x)**2/2:
            l *= self.eta
            xk = self.subprob(x, grad, l)
        self.lsmooth = l
        return xk
