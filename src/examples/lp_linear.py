import random
import math
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
import scipy.sparse as sp
from src.algorithms.bpg import ApproxBPG
from src.algorithms.pg import PG
from src.tools.operators import Bregman, power_method
from src.tools.plot_results import plot_csv_results


def obj(x):
    return loss(x)


def reg(x):
    return 0


def loss(x):
    return np.sum(np.power(np.abs(A.dot(x) - b), p))/p


def grad_f(x):
    Axb = A.dot(x) - b
    return A.T.dot(np.power(np.abs(Axb), p-1) * np.sign(Axb))


def hess_f(x):
    return AA + theta*p*(p-1)*np.diag(np.power(np.abs(x), p-2))


def kernel(x):
    return LA.norm(x)**2/2 + np.sum(np.power(np.abs(A.dot(x)), p))/p


def grad_kernel(x):
    Ax = A.dot(x)
    return x + A.T.dot(np.power(np.abs(Ax), p-1) * np.sign(Ax))


def subprob_sol(x, grad, L):
    Ax = A.dot(x)
    Axp = np.power(np.abs(Ax - b), p-2)
    hess = np.eye(n) + (p-1)*np.dot(A.T, np.diag(Axp)).dot(A)
    inv = LA.inv(hess)
    return x - inv.dot(grad)/L


def subprob_sol_pg(x, grad, L):
    return x - grad/L


def subprob_sol_rnewton(x, grad, L):
    hess = hess_f(x)
    lam = np.sqrt(LA.norm(grad)*h)
    inv = LA.inv(hess + lam)
    d = -inv.dot(grad)
    return x + d


def setup(n, m, sparsity):
    A = sp.random(m, n, density=1, data_rvs=rnd.randn)
    A = A.toarray()
    A = np.apply_along_axis(lambda x: x / LA.norm(x), 0, A)
    sparse_index = math.ceil(n*sparsity)
    entry = rnd.randn(sparse_index)

    opt = np.zeros(n)
    for i, j in enumerate(random.sample(range(n), sparse_index)):
        opt[j] = entry[i]
    opt = opt / LA.norm(opt)
    b = A.dot(opt)
    return A, b, opt


def l_smad(A, b, p):
    Ap = np.sum(np.power(np.abs(A), p), axis=1)
    A2 = np.sum(A**2, axis=1)
    A2b = A2 * np.power(np.abs(b), p-2)
    if p >= 3:
        return np.sum(Ap + A2b)*2**(p-3)
    else:
        return np.max(np.power(np.abs(b), p - 2))


if __name__ == '__main__':
    m = 500
    n = 200
    sparsity = .1
    eps = 1e-6
    p = 1.1

    h = 1e-5

    rnd.seed(42)
    random.seed(42)

    A, b, xopt = setup(n, m, sparsity)
    Ab = A.T.dot(b)
    AA = A.T.dot(A)
    AAT = A.dot(A.T)
    norm_A = LA.norm(A)
    theta = 1
    x0 = np.abs(power_method(A.dot, A.T.dot, b, n))
    x0 = x0 / LA.norm(x0)
    lsmad = 1
    lsmooth = lsmad
    bregman = Bregman(kernel, grad_kernel)
    bregman_dist = bregman.dist

    dir_path = '../../results/'
    nprob = 'lp_linear_p={}'.format(p)
    nalg = 'abpg'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmad)
    csv_path = path + '.csv'
    csv_paths = [csv_path]
    funcs = {'obj': obj, 'acc': lambda x: LA.norm(x - xopt), 'grad': lambda x: LA.norm(grad_f(x))}

    abpg = ApproxBPG(x0, obj, grad_f, xopt, lsmad, subprob_sol, bregman_dist, reg, stop='diff', csv_path=csv_path)
    abpg.run()
    np.save(path, abpg.xk)
    np.save(dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, n, m, sparsity, theta, lsmad)+'-gt', xopt)

    nalg = 'pg'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmooth)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    pg = PG(x0, obj, grad_f, xopt, lsmooth, subprob_sol_pg, loss, linesearch=False, stop='diff', csv_path=csv_path)
    pg.run()
    np.save(path, pg.xk)

    nalg = 'pgl'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmooth)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    pg = PG(x0, obj, grad_f, xopt, lsmooth, subprob_sol_pg, loss, linesearch=True, stop='diff', csv_path=csv_path)
    pg.run()
    np.save(path, pg.xk)

    titles = {'obj': 'objective function values',
              'diff': 'difference of iteration',
              'acc': 'accuracy',
              'grad': 'gradient values'}
    labels = {'iter': 'k', 'obj': r'$\log_{10}\Psi(x^k)$',
              'diff': r'$\log_{10}||x^k - x^{k-1}||$',
              'acc': r'$\log_{10}||x^k - x^*||$',
              'grad': r'$\log_{10}||\nabla f(x^k)||$'}
    logs = {'obj': True, 'reg': True, 'acc': True, 'grad': True}
    plot_csv_results(csv_paths, titles=titles, labels=labels, logs=logs)
