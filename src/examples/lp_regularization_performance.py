import random
import math
import csv
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
from src.algorithms.bpg import ApproxBPG
from src.algorithms.pg import PG
from src.tools.operators import Bregman, power_method


def obj(x):
    return loss(x) + reg(x)


def loss(x):
    return LA.norm(A.dot(x) - b)**2/2 + theta*np.sum(np.power(np.abs(x), p))/p


def reg(x):
    return 0


def grad_f(x):
    return AA.dot(x) - Ab + theta*np.power(np.abs(x), p-1)*np.sign(x)


def hess_f(x):
    return AA + theta*(p-1)*np.diag(np.power(np.abs(x), p-2))


def kernel(x):
    return LA.norm(x)**2/2 + np.sum(np.abs(np.power(x, p)))/p


def grad_kernel(x):
    return x + np.power(np.abs(x), p-1)*np.sign(x)


def subprob_sol(x, grad, L):
    return x - grad/L/(1 + theta*(p-1)*np.power(np.abs(x), p-2))


def subprob_sol_bpg(x, grad, L):
    return 0


def subprob_sol_pg(x, grad, L):
    v = x - grad/L
    return v


def subprob_sol_rnewton(x, grad, L):
    hess = hess_f(x)
    lam = np.sqrt(LA.norm(grad)*h)
    inv = LA.inv(hess + lam)
    d = -inv.dot(grad)
    return x + d


def setup(n, m, sparsity):
    A = rnd.randn(m, n)
    A = np.apply_along_axis(lambda x: x / LA.norm(x), 0, A)
    sparse_index = math.ceil(n*sparsity)
    entry = rnd.randn(sparse_index)

    opt = np.zeros(n)
    for i, j in enumerate(random.sample(range(n), sparse_index)):
        opt[j] = entry[i]
    opt = opt / LA.norm(opt)

    b = A.dot(opt)
    return A, b, opt


if __name__ == '__main__':
    sparsity = .05
    eps = 1e-6
    p = 1.1
    h = 1e-5

    m_list = []
    n_list = []

    theta = 0.05
    instance = 1

    for m in m_list:
        for n in n_list:
            dir_path = '../../results/'
            nprob = 'lp_regularization_terminal_perform_p={}'.format(p)
            path = dir_path + '{}-{}-{}-{}-{}'.format(nprob, n, m, sparsity, theta)
            csv_path = path + '.csv'
            file = open(csv_path, 'a')
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(['Instance', 'Algorithm', 'Iter', 'obj', 'acc', 'grad', 'cpu'])  # Python 3.7 以降推奨
            for i in range(instance):
                A, b, xopt = setup(n, m, sparsity)
                Ab = A.T.dot(b)
                AA = A.T.dot(A)
                x0 = rnd.randn(n)
                u = power_method(AA.dot, AA.T.dot, 1, n)
                lsmooth = u.T.dot(AA.dot(u)) / LA.norm(u) ** 2
                lsmad = lsmooth + theta
                bregman = Bregman(kernel, grad_kernel)
                bregman_dist = bregman.dist

                abpg = ApproxBPG(x0, lsmad, grad_f, subprob_sol, bregman_dist, obj, reg, stop='diff')
                abpg.run()

                pg = PG(x0, lsmooth, loss, grad_f, subprob_sol_pg, stop='diff')
                pg.run()

                pgl = PG(x0, lsmooth, loss, grad_f, subprob_sol_pg, linesearch=True, stop='diff')
                pgl.run()

                newton = PG(x0, 1, loss, grad_f, subprob_sol_rnewton, stop='diff')
                newton.run()

                writer.writerow([i, 'ABPG', abpg.iter, obj(abpg.xk), LA.norm(abpg.xk - xopt), LA.norm(grad_f(abpg.xk)),
                                 abpg.cpu_time])
                writer.writerow(
                    [i, 'PG', pg.iter, obj(pg.xk), LA.norm(pg.xk - xopt), LA.norm(grad_f(pg.xk)), pg.cpu_time])
                writer.writerow(
                    [i, 'PG_linesearch', pgl.iter, obj(pgl.xk), LA.norm(pgl.xk - xopt), LA.norm(grad_f(pgl.xk)),
                     pgl.cpu_time])
                writer.writerow(
                    [i, 'RN', newton.iter, obj(newton.xk), LA.norm(newton.xk - xopt), LA.norm(grad_f(newton.xk)),
                     newton.cpu_time])
            file.close()
