import random
import math
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
from src.algorithms.bpg import ApproxBPG
from src.algorithms.pg import PG
from src.tools.operators import Bregman, power_method, constrained_linear_system
from src.tools.plot_results import plot_csv_results


def obj(x):
    return loss(x) + reg(x)


def loss(x):
    return LA.norm(A.dot(x) - b)**2/2 + theta*np.sum(np.power(np.abs(x), p))


def reg(x):
    return 0


def grad_f(x):
    return AA.dot(x) - Ab + theta*p*np.power(np.abs(x), p-1)*np.sign(x)


def hess_f(x):
    return AA + theta*p*(p-1)*np.diag(np.power(np.abs(x), p-2))


def kernel(x):
    return LA.norm(x)**2/2 + np.sum(np.abs(np.power(x, p)))


def grad_kernel(x):
    return x + p*np.power(np.abs(x), p-1)*np.sign(x)


def subprob_sol(x, grad, L):
    phi_inv = 1/(1 + theta * p * (p - 1) * np.power(np.abs(x), p - 2))
    res = x + constrained_linear_system(phi_inv, e, grad/L)
    return res


def subprob_sol_pg(x, grad, L):
    v = x - grad/L - e*np.sum(grad)/n/L
    return v


def subprob_sol_rnewton(x, grad, L):
    hess = hess_f(x)
    lam = np.sqrt(LA.norm(grad)*h)
    inv = LA.inv(hess + lam)
    v = inv.dot(grad)
    beta_inv = -1/e.dot(inv).dot(e)
    return x - v - np.sum(v)*inv.dot(e)*beta_inv


def setup(n, m, sparsity):
    A = rnd.randn(m, n)
    A = np.apply_along_axis(lambda x: x / LA.norm(x), 0, A)
    sparse_index = math.ceil(n*sparsity)
    entry = rnd.randn(sparse_index)

    opt = np.zeros(n)
    for i, j in enumerate(random.sample(range(n), sparse_index)):
        opt[j] = entry[i]
    opt = opt / np.sum(opt)
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
    e = np.ones(n)
    Ab = A.T.dot(b)
    AA = A.T.dot(A)
    e = np.ones(n)
    theta = 0.05
    x0 = rnd.randn(n)
    x0 = x0 / np.sum(x0)
    u = power_method(AA.dot, AA.T.dot, 1, n)
    lsmooth = u.T.dot(AA.dot(u))/LA.norm(u)**2
    lsmad = max(lsmooth, theta*p*(p-1.0))
    bregman = Bregman(kernel, grad_kernel)
    bregman_dist = bregman.dist

    dir_path = '../../results/'
    nprob = 'lp_regularization_p={}_simplex'.format(p)
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

    nalg = 'rn'
    path = dir_path + '{}-{}-{}-{}-{}-{}-{}'.format(nprob, nalg, n, m, sparsity, theta, h)
    csv_path = path + '.csv'
    csv_paths.append(csv_path)
    newton = PG(x0, obj, grad_f, xopt, 1, subprob_sol_rnewton, loss, stop='diff', csv_path=csv_path)
    newton.run()
    np.save(path, newton.xk)

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
