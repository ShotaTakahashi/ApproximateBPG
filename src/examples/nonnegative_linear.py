import random
import math
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
from src.algorithms.bpg import ApproxBPG, BPG
from src.algorithms.pg import PG
from src.tools.operators import soft_thresholding, Bregman
from src.tools.plot_results import plot_csv_results


def kl_divergence(x, y):
    return np.sum((x+eps)*(np.log(x + eps) - np.log(y + eps)) - x + y)


def shannon_entropy(x):
    return np.sum(x*np.log(x)) + LA.norm(x)**2/2


def grad_shannon_entropy(x):
    return 1 + np.log(x) + x


def obj(x):
    return loss(x) + reg(x)


def loss(x):
    return kl_divergence(A.dot(x), b)


def reg(x):
    return theta*np.sum(np.abs(x))


def grad_f(x):
    Ax = A.dot(x)
    res = (np.log(Ax + eps) - np.log(b)).dot(A)
    return res


def subprob_sol(x, grad, L):
    p = L*x - x*grad
    v = soft_thresholding(p, L*theta*x)
    res = np.maximum(v, 0)/L
    return res


def subprob_sol_bpg(x, grad, L):
    Ax = A.dot(x)
    p = np.prod(np.power(np.tile(Ax/b, (n, 1)).T, A/2), axis=0)
    return x*np.exp(-theta/2)/p


def subprob_sol_pg(x, grad, L):
    res = soft_thresholding(x - grad/L, theta/L)
    res = np.maximum(res, 0.0)
    return res


def setup(n, m, sparsity):
    A = rnd.randn(m, n)
    A = np.abs(A)
    A = np.apply_along_axis(lambda x: x/np.sum(x), 0, A)
    sparse_index = math.ceil(n*sparsity)
    entry = rnd.uniform(0, 1, sparse_index)

    opt = np.zeros(n)
    for i, j in enumerate(random.sample(range(n), sparse_index)):
        opt[j] = entry[i]
    opt = np.abs(opt) / np.sum(opt)
    b = A.dot(opt)
    return A, b, opt


if __name__ == '__main__':
    m = 500
    n = 200
    sparsity = .05
    eps = 1e-6

    rnd.seed(42)
    random.seed(42)

    A, b, xopt = setup(n, m, sparsity)
    e = np.ones(n)
    theta = .05
    print(theta)
    x0 = np.abs(rnd.randn(n))
    x0 = x0 / np.sum(x0)
    lsmad = np.max(np.sum(A, axis=1))
    bregman = Bregman(shannon_entropy, grad_shannon_entropy)
    bregman_dist = bregman.dist

    dir_path = '../../results/'
    nprob = 'nonnegative_linear'
    nalg = 'abpg'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmad)
    csv_path = path + '.csv'
    csv_paths = [csv_path]
    funcs = {'obj': obj, 'acc': lambda x: LA.norm(x - xopt), 'grad': lambda x: LA.norm(grad_f(x))}

    abpg = ApproxBPG(x0, obj, grad_f, xopt, lsmad, subprob_sol, bregman_dist, reg, stop='diff', csv_path=csv_path)
    abpg.run()
    np.save(path, abpg.xk)
    np.save(dir_path + '{}-{}-{}-{}-{}-{}'.format(nprob, n, m, sparsity, theta, lsmad)+'-gt', xopt)

    nalg = 'bpg'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmad)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    bpg = BPG(x0, obj, grad_f, xopt, lsmad, subprob_sol_bpg, bregman_dist, stop='diff', csv_path=csv_path)
    bpg.run()
    np.save(path, bpg.xk)

    nalg = 'pgl'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, lsmad)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    pg = PG(x0, obj, grad_f, xopt, lsmad, subprob_sol_pg, loss, linesearch=True, stop='diff', csv_path=csv_path)
    pg.run()
    np.save(path, pg.xk)

    titles = {'obj': 'objective function values',
              'acc': 'accuracy',
              'diff': 'difference of iteration',
              'grad': 'norm of gradient'}
    labels = {'iter': 'k', 'obj': r'$\log_{10}\Psi(x^k)$',
              'acc': r'$\log_{10}||x^k - x^*||$',
              'diff': r'$\log_{10}||x^k - x^{k-1}||$',
              'grad': r'$\|\nabla f(x^k)\|$'}
    logs = {'obj': True, 'acc': True, 'diff': True, 'grad': True}
    plot_csv_results(csv_paths, titles=titles, labels=labels, logs=logs)
