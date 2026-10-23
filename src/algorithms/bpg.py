import numpy as np
import pandas as pd
from src.algorithms.iteration import FOM


class BPG(FOM):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, stop='diff', csv_path='', dist_fn=None):
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
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, alpha=0.99, eta=0.9, stop='diff', csv_path='', dist_fn=None):
        super().__init__(x0, obj, grad, opt, lsmad, subprob, bregman_dist, stop, csv_path, dist_fn)
        self.reg = reg
        self.t_init = 1.0
        self.alpha = alpha
        self.eta = eta
        self.tk_history = []

    def update(self, x, y):
        grad = self.grad(y)
        dk = self.subprob(y, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x)
        tk = self.t_init
        while self.obj(x + tk*dk) > obj_xk + self.alpha*tk*const:
            tk *= self.eta

        self.tk_history.append(tk)
        return x + tk*dk
    
    def save_details_csv(self):
        if not self.csv_path:
            return None
        
        # 元のファイル名と区別するために _details を付与
        detail_path = self.csv_path.replace('.csv', '_approxbpg_details.csv')
        
        df_hist = pd.DataFrame({
            'tk': self.tk_history
        })
        df_hist.index.name = 'iteration'
        df_hist.index = df_hist.index + 1  # 1始まりにする
        
        df_hist.to_csv(detail_path)
        return detail_path


class MABPG(ApproxBPG):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, dist, is_bregman, alpha=0.99, eta=0.9, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, alpha, eta, stop, csv_path)
        self.t_init = 1.0
        self.dist = dist
        self.is_bregman = is_bregman

    def update(self, x, y):
        grad = self.grad(y)
        dk = self.subprob(y, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        if self.is_bregman:
            dist = self.bregman_dist(x+dk, x)*self.lsmad
        else:
            dist = self.dist(x, dk)*self.lsmad
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x) + dist
        tk = self.t_init
        while self.obj(x + tk*dk) > obj_xk + self.alpha*tk*const:
            tk *= self.eta
        if self.obj(x + dk) < self.obj(x + tk*dk):
            return x + dk
        else:
            return x + tk*dk

class ABPGVM(ApproxBPG):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, loss, reg, dist, grad_reg, alpha=0.99, eta=2, mu=0.9, t_max=2**20, stop='diff', csv_path=''):
        super().__init__(x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, alpha, eta, stop, csv_path)
        self.t_init = 1.0
        self.dist = dist
        self.grad_reg = grad_reg
        self.loss = loss
        self.mu = mu
        self.t_max = t_max
        self.tk_history = []
        self.step_choice_history = []
        self.backtrack_history = []

    def update(self, x, y):
        grad = self.grad(y)
        dk = self.subprob(y, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        dist = self.dist(x, dk)*self.lsmad
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x)

        def A(t):
            return self.obj(x + t*dk) - obj_xk - self.alpha*t*(const + dist)

        backtrack_count = 0
        if A(1) < 0:
            tk = 1
            while A(self.eta*tk) < 0 and self.eta*tk <= self.t_max:
                tk = self.eta*tk
                backtrack_count += 1
        else:
            j = 1
            while A(self.mu**j) >= 0:
                j += 1
                backtrack_count += 1
            tk = self.mu**j

        self.tk_history.append(tk)
        self.backtrack_history.append(backtrack_count)

        if self.obj(x + dk) < self.obj(x + tk*dk):
            self.step_choice_history.append(1)
            return x + dk
        else:
            self.step_choice_history.append(0)
            return x + tk*dk


    def save_details_csv(self):
        if not self.csv_path:
            return

        detail_path = self.csv_path.replace('.csv', '_details.csv')

        df_hist = pd.DataFrame({
            'tk': self.tk_history,
            'is_full_step': self.step_choice_history,
            'backtrack_count': self.backtrack_history
        })
        df_hist.index.name = 'iteration'

        df_hist.index = df_hist.index + 1

        df_hist.to_csv(detail_path)
        return detail_path

    def step_choice_summary(self):
        n = len(self.step_choice_history)
        if n == 0:
            return {}
        yk_count = sum(self.step_choice_history)
        tkd_count = n - yk_count
        return {
            'total': n,
            'yk_count': yk_count,
            'yk_ratio': yk_count / n,
            'tkd_count': tkd_count,
            'tkd_ratio': tkd_count / n,
        }


class ABPGVMA(ApproxBPG):
    def __init__(self, x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, dist, in_domain,
                 c1=0.99, eta=0.9, delta=2, t_max=2**20, stop='diff', csv_path='', dist_fn=None):
        super().__init__(x0, obj, grad, opt, lsmad, subprob, bregman_dist, reg, c1, delta, stop, csv_path, dist_fn)
        self.t_init = 1.0
        self.dist = dist
        self.in_domain = in_domain
        self.c1 = c1
        self.eta = eta
        self.delta = delta
        self.t_max = t_max

        self.tk_history = []
        self.phase_history = []
        self.eval_count_history = []
        self.step_choice_history = []
        self.hit_tmax_history = []
        self.stopped_by_domain_history = []

        self.invariant_violations = []

    def update(self, x, y):
        grad = self.grad(y)
        dk = self.subprob(y, grad, self.lsmad) - x
        obj_xk = self.obj(x)
        dist = self.dist(x, dk) * self.lsmad
        const = np.dot(grad, dk) + self.reg(x + dk) - self.reg(x) + dist

        eval_count = 0

        def A(t):
            nonlocal eval_count
            eval_count += 1
            point = x + t * dk
            if not self.in_domain(point):
                return np.inf
            return self.obj(point) - obj_xk - self.c1 * t * const

        hit_tmax = False
        stopped_by_domain = False

        a1 = A(1.0)
        if a1 < 0:
            phase = 'expand'
            tk = 1.0
            while tk < self.t_max:
                t_next = min(self.delta * tk, self.t_max)
                if not self.in_domain(x + t_next * dk):
                    stopped_by_domain = True
                    break
                if A(t_next) >= 0:
                    break
                tk = t_next
            else:
                hit_tmax = True
        else:
            phase = 'backtrack'
            tk = self.eta
            tk_floor = 1e-300
            while A(tk) >= 0 and tk > tk_floor:
                tk = self.eta * tk
            if tk <= tk_floor:
                self.invariant_violations.append(
                    (self.iter, 'backtrack underflow: tk hit floor without A(tk)<0', tk))
        a_tk = A(tk)
        if not (tk > 0):
            self.invariant_violations.append((self.iter, 't_k>0 violated', tk))
        if not (a_tk < 0):
            self.invariant_violations.append((self.iter, 'A_k(t_k)<0 violated', (tk, a_tk)))

        self.tk_history.append(tk)
        self.phase_history.append(phase)
        self.eval_count_history.append(eval_count)
        self.hit_tmax_history.append(hit_tmax)
        self.stopped_by_domain_history.append(stopped_by_domain)

        y_point = x + dk
        td_point = x + tk * dk
        y_in = self.in_domain(y_point)
        obj_y = self.obj(y_point) if y_in else np.inf
        obj_td = self.obj(td_point)

        if obj_y < obj_td:
            self.step_choice_history.append(1)
            chosen = y_point
        else:
            self.step_choice_history.append(0)
            chosen = td_point
        if not (self.obj(chosen) <= obj_xk + 1e-12):
            self.invariant_violations.append((self.iter, 'monotone decrease violated', (self.obj(chosen), obj_xk)))

        return chosen

    def save_details_csv(self):
        if not self.csv_path:
            return
        detail_path = self.csv_path.replace('.csv', '_details.csv')
        df_hist = pd.DataFrame({
            'tk': self.tk_history,
            'phase': self.phase_history,
            'eval_count': self.eval_count_history,
            'is_full_step': self.step_choice_history,
            'hit_tmax': self.hit_tmax_history,
            'stopped_by_domain': self.stopped_by_domain_history,
        })
        df_hist.index.name = 'iteration'
        df_hist.index = df_hist.index + 1
        df_hist.to_csv(detail_path)
        return detail_path

    def step_choice_summary(self):
        n = len(self.step_choice_history)
        if n == 0:
            return {}
        yk_count = sum(self.step_choice_history)
        tkd_count = n - yk_count
        expand_count = sum(1 for p in self.phase_history if p == 'expand')
        backtrack_count = sum(1 for p in self.phase_history if p == 'backtrack')
        return {
            'total': n,
            'yk_count': yk_count,
            'yk_ratio': yk_count / n,
            'tkd_count': tkd_count,
            'tkd_ratio': tkd_count / n,
            'expand_count': expand_count,
            'backtrack_count': backtrack_count,
            'tk_min': min(self.tk_history),
            'tk_max': max(self.tk_history),
            'hit_tmax_count': sum(self.hit_tmax_history),
            'stopped_by_domain_count': sum(self.stopped_by_domain_history),
            'invariant_violations': len(self.invariant_violations),
        }
