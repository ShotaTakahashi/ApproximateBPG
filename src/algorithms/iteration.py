import csv
import numpy as np
import numpy.linalg as LA
import time
from abc import ABCMeta, abstractmethod


class Iteration(metaclass=ABCMeta):
    def __init__(self, x0):
        self.xk = x0
        self.xk_old = x0
        self.iter = 0


class Extrapolation(Iteration):
    def __init__(self, x0, restart_iter=200):
        Iteration.__init__(self, x0)
        self.yk = x0
        self.beta, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
        self.restart_iter = restart_iter
        self.__restart = self.restart_iter

    def adaptive_scheme(self):
        return np.dot(self.yk - self.xk, self.xk - self.xk_old) > 0

    def beta_update(self):
        self.beta = (self.__theta_old - 1) / self.__theta
        self.yk = self.xk + self.beta * (self.xk - self.xk_old)
        self.__theta_update()
        self.__theta_restart()

    def __theta_update(self):
        self.__theta_old, self.__theta = self.__theta, (1 + (1 + 4 * self.__theta ** 2) ** 0.5) * 0.5

    def __theta_restart(self):
        if self.adaptive_scheme():
            self.beta, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
        if self.iter == self.restart_iter:
            self.beta, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
            self.restart_iter += self.__restart


class FOM(Extrapolation):
    def __init__(self, x0, obj, grad, opt, stop='diff', csv_path=''):
        super().__init__(x0)
        self.obj = obj
        self.grad = grad
        self.opt = opt
        self.TOL = 1e-6
        self.MAX_ITER = 1000
        self.csv_path = csv_path
        if self.csv_path:
            self.write = True
            self.file = open(self.csv_path, 'w')
            self.writer = csv.writer(self.file, lineterminator='\n')
            self.writer.writerow(['iter', 'obj', 'acc', 'grad', 'diff'])
            self.writer.writerow(
                [self.iter, self.obj(self.xk), LA.norm(self.xk - self.opt), LA.norm(self.grad(self.xk)), LA.norm(self.xk - self.xk_old)])
        self.time = 0.0
        self.stop = stop

    @abstractmethod
    def update(self, x):
        pass

    def run(self, extrapolation=False):
        start = time.time()
        for _ in range(self.MAX_ITER):
            self.iter += 1

            if extrapolation:
                self.beta_update()
            else:
                self.beta = 0.0

            self.yk = self.xk + self.beta*(self.xk - self.xk_old)
            self.xk_old, self.xk = self.xk, self.update(self.yk)
            if self.write:
                self.writer.writerow(
                    [self.iter, self.obj(self.xk), LA.norm(self.xk - self.opt), LA.norm(self.grad(self.xk)), LA.norm(self.xk - self.xk_old)])

            if self.stop_criteria():
                break
        end = time.time()
        self.time = end - start
        print('Time: {} sec'.format(self.time))
        self.file.close()
        return self.xk

    def stop_criteria(self):
        if self.stop == 'grad':
            return LA.norm(self.grad(self.xk)) < self.TOL
        else:
            return LA.norm(self.xk - self.xk_old) < self.TOL

