"""
class to sample from the moving-sum changepoint model
"""

import numpy as np
from model_sampler.segment import Segment


class ChangepointModel:

    def __init__(self, **kwargs):

        self.x = kwargs['x'] if 'x' in kwargs else None
        self.T = len(self.x) if self.x is not None else kwargs['T']
        self.x_distr = kwargs['x_distr']
        self.x_hyper = kwargs['x_hyper']

        self.tau = [1, self.T+1] if 'tau' not in kwargs else kwargs['tau']
        self.w = [1 for _ in self.tau] if 'w' not in kwargs else kwargs['w']

        if 1 not in self.tau:
            self.tau = [1] + self.tau
            self.w = [1] + self.w
        if self.T+1 not in self.tau:
            self.tau = self.tau + [self.T+1]
            self.w = self.w + [1]

        self.tau = np.array(self.tau)
        self.w = np.array(self.w)
        self.w_ind = np.where(self.w == 1)[0]
        #self.tau_w = self.tau[self.w_ind]
        self.theta = []

        self.p = kwargs['p'] if 'p' in kwargs else 1.0/self.T
        self.w_hyper = kwargs['w_hyper'] if 'w_hyper' in kwargs else 0.05

        w_k = len(self.w_ind)-1
        self.m_hyper = kwargs['m_hyper'] if 'm_hyper' in kwargs else 0.1
        self.m = [0 for _ in range(w_k)] if 'm' not in kwargs else kwargs['m']
        self.y_init = [np.array([]) for _ in range(w_k)] if 'y_init' not in kwargs else kwargs['y_init']
        self.y = [np.array([])]

    def sample_x(self, theta=None):
        x = []
        y = []
        for j in range(1, len(self.w_ind)):
            tau_s = self.tau[self.w_ind[j-1]:(self.w_ind[j]+1)] - self.tau[self.w_ind[j-1]]
            seg_j = Segment(x_distr=self.x_distr, x_hyper=self.x_hyper, tau_s=tau_s, m=self.m[j-1])
            if theta is None:
                seg_j.sample_x(theta=theta)
                self.theta += [seg_j.theta[:]]
            else:
                pos_w1 = np.where(self.w == 1)[0]
                theta_j = theta[pos_w1[j-1]:pos_w1[j]]
                seg_j.sample_x(theta=theta_j)
            x += [seg_j.x[:]]
            y += [seg_j.y[:]]
            self.y_init[j-1] = seg_j.y_init[:]
        self.x = np.concatenate(x)
        self.y = np.concatenate(y)

    def sample_m(self):
        w_k = len(self.w_ind) - 1
        self.m = np.array([np.random.geometric(p=self.m_hyper) for _ in range(w_k)])



