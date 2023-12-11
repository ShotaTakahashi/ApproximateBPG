import random
import math
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
from src.algorithms.bpg import ApproxBPG
from src.algorithms.pg import PG
from src.tools.operators import Bregman, power_method
from src.tools.plot_results import plot_csv_results


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
    m = 800
    n = 500
    sparsity = .05
    eps = 1e-6
    p = 1.1
    h = 1e-5

    rnd.seed(42)
    random.seed(42)

    A, b, xopt = setup(n, m, sparsity)
    Ab = A.T.dot(b)
    AA = A.T.dot(A)
    theta = 0.05
    x0 = rnd.randn(n)
    u = power_method(AA.dot, AA.T.dot, 1, n)
    lsmooth = u.T.dot(AA.dot(u))/LA.norm(u)**2
    lsmad = lsmooth + theta
    bregman = Bregman(kernel, grad_kernel)
    bregman_dist = bregman.dist

    dir_path = '../../results/'
    nprob = 'lp_regularization_p={}'.format(p)
    nalg = 'abpg'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmad)
    csv_path = path + '.csv'
    csv_paths = [csv_path]
    funcs = {'obj': obj, 'acc': lambda x: LA.norm(x - xopt), 'grad': lambda x: LA.norm(grad_f(x))}

    abpg = ApproxBPG(x0, lsmad, grad_f, subprob_sol, bregman_dist, obj, reg, stop='diff', write=True)
    abpg.run(result_dir=csv_path, funcs=funcs)
    np.save(path, abpg.xk)
    np.save(dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, n, m, sparsity, theta, lsmad)+'-gt', xopt)

    nalg = 'pg'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmooth)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    pg = PG(x0, lsmooth, loss, grad_f, subprob_sol_pg, stop='diff', write=True)
    pg.run(result_dir=csv_path, funcs=funcs)
    np.save(path, pg.xk)

    nalg = 'pgl'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmooth)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    pg = PG(x0, lsmooth, loss, grad_f, subprob_sol_pg, linesearch=True, stop='diff', write=True)
    pg.run(result_dir=csv_path, funcs=funcs)
    np.save(path, pg.xk)

    nalg = 'rn'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, h)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    newton = PG(x0, 1, loss, grad_f, subprob_sol_rnewton, stop='diff', write=True)
    newton.run(result_dir=csv_path, funcs=funcs)
    np.save(path, newton.xk)

    titles = {'obj': 'objective function values',
              'reg': 'regularization function values',
              'acc': 'accuracy',
              'grad': 'gradient values'}
    labels = {'Iter': 'k', 'obj': r'$\log_{10}\Psi(x^k)$',
              'reg': r'$\log_{10}g(x^k)$',
              'acc': r'$\log_{10}||x^k - x^*||$',
              'grad': r'$\log_{10}||\nabla f(x^k)||$'}
    logs = {'obj': True, 'reg': True, 'acc': True, 'grad': True}
    plot_csv_results(csv_paths, titles=titles, labels=labels, logs=logs)
