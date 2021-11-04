"""
Markov Chain Monte Carlo for the moving-sum changepoint model
"""


import time
import numpy as np
import pandas as pd
from model_sampler.changepoint_model import ChangepointModel
from model_sampler.segment import Segment

import matplotlib.pyplot as plt


class MCMCsampler(ChangepointModel):

    def __init__(self, **kwargs):

        ChangepointModel.__init__(self, **kwargs)

        # storage for the sampler
        self.sample_tau = []
        self.sample_w = []
        self.sample_m = []
        self.sample_y_init = []
        self.accept_prob = 0
        self.accept_count = {}
        self.accept_prop = {}

        # estimation
        self.post_distr_k = {}
        self.post_distr_kw = {}
        self.post_distr_tau = {}
        self.post_distr_tauw = {}
        self.k_MAP = None
        self.kw_MAP = None
        self.tau_MAP = None
        self.tauw_MAP = None
        self.m_MAP = None

    ################
    # MCMC moves
    ################

    def shift(self):

        if len(self.tau) < 3:
            pass
        else:
            j = np.random.randint(low=1, high=len(self.tau) - 1)
            tau_p = np.random.randint(low=self.tau[j - 1] + 1, high=self.tau[j + 1])

            if tau_p == self.tau[j]:
                pass
            else:
                if self.w[j] == 0:
                    j0 = max(self.w_ind[self.w_ind < j])
                    j1 = j - j0
                    wj = np.where(self.w_ind == j0)[0][0]

                    if self.m[wj] > 0:
                        seg_j = Segment(x=self.x[self.tau[j0] - 1:self.tau[j + 1] - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j0:(j + 2)] - self.tau[j0],
                                        m=self.m[wj], y_init=self.y_init[wj])

                        seg_j.get_y()

                        self.accept_prob = -seg_j.llik_y_given_m(sub=[j1, j1 + 1])
                        seg_j.tau_s[j1] = tau_p - self.tau[j0]
                        self.accept_prob += seg_j.llik_y_given_m(sub=[j1, j1 + 1])
                    else:
                        seg_j = Segment(x=None, x_distr=self.x_distr, x_hyper=self.x_hyper, m=0)
                        self.accept_prob = seg_j.llik_nu(y_full=self.x[(self.tau[j - 1] - 1):tau_p - 1], m=0)
                        self.accept_prob -= seg_j.llik_nu(y_full=self.x[(self.tau[j - 1] - 1):self.tau[j] - 1], m=0)
                        self.accept_prob += seg_j.llik_nu(y_full=self.x[(tau_p - 1):self.tau[j + 1] - 1], m=0)
                        self.accept_prob -= seg_j.llik_nu(y_full=self.x[(self.tau[j] - 1):self.tau[j + 1] - 1], m=0)

                    if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                        self.tau[j] = tau_p
                        self.accept_count['shift'] += 1
                else:

                    if tau_p - self.tau[j] > 0:

                        wj = np.where(self.w_ind == j)[0][0]
                        j0 = self.w_ind[wj - 1]  # index of w=1 on the left

                        # here could check if m == 0 to speed up code

                        seg_i = Segment(x=self.x[self.tau[j0] - 1:tau_p - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j0:(j + 1)] - self.tau[j0],
                                        m=self.m[wj - 1], y_init=self.y_init[wj - 1])
                        seg_i.get_y()
                        if not seg_i.valid_y():
                            self.accept_prob = -np.infty
                        else:

                            self.accept_prob = -seg_i.llik_y_given_m(sub=[j - j0])
                            seg_i.tau_s[-1] = tau_p - self.tau[j0]
                            self.accept_prob += seg_i.llik_y_given_m(sub=[j - j0])

                            seg_u = Segment(x=self.x[self.tau[j] - 1:self.tau[j + 1] - 1],
                                            x_distr=self.x_distr, x_hyper=self.x_hyper,
                                            tau_s=self.tau[j:(j + 2)] - self.tau[j],
                                            m=self.m[wj], y_init=self.y_init[wj])
                            seg_u.get_y()
                            y_full = np.concatenate([seg_u.y_init, seg_u.y])
                            self.accept_prob -= seg_u.llik_nu(y_full=y_full, m=seg_u.m)
                            y_trunc = y_full[(tau_p - self.tau[j]):]
                            seg_u.y_init = y_trunc[:seg_u.m]
                            self.accept_prob += seg_u.llik_nu(y_full=y_trunc, m=seg_u.m)

                    else:

                        wj = np.where(self.w_ind == j)[0][0]
                        j0 = self.w_ind[wj - 1]

                        seg_u = Segment(x=self.x[self.tau[j] - 1:self.tau[j + 1] - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j:(j + 2)] - self.tau[j],
                                        m=self.m[wj], y_init=self.y_init[wj])
                        seg_u.get_y()
                        self.accept_prob -= seg_u.llik_y_given_m(sub=[1])

                        seg_u.extend(n_x=self.x[tau_p - 1:self.tau[j] - 1])

                        if not seg_u.valid_y():
                            self.accept_prob = -np.infty
                        else:
                            self.accept_prob += seg_u.llik_y_given_m(sub=[1])

                            seg_i = Segment(x=self.x[(self.tau[j0] - 1):self.tau[j] - 1],
                                            x_distr=self.x_distr, x_hyper=self.x_hyper,
                                            tau_s=self.tau[j0:(j + 1)] - self.tau[j0],
                                            m=self.m[wj - 1], y_init=self.y_init[wj - 1])

                            seg_i.get_y()
                            j1 = j - j0
                            self.accept_prob -= seg_i.llik_y_given_m(sub=[j1])
                            seg_i.tau_s[j1] = tau_p - self.tau[j0]
                            self.accept_prob += seg_i.llik_y_given_m(sub=[j1])

                    if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                        self.tau[j] = tau_p
                        self.y_init[wj] = np.array(seg_u.y_init[:])
                        self.accept_count['shift'] += 1

    def prop_w(self):

        if len(self.tau) > 2 and 0 < self.w_hyper < 1:
            j = np.random.randint(1, len(self.tau) - 1)

            if self.w[j] == 0:

                j0 = max(self.w_ind[self.w_ind < j])
                j1 = min(self.w_ind[self.w_ind > j])
                wj = np.where(self.w_ind == j0)[0][0]

                seg_j = Segment(x=self.x[(self.tau[j0] - 1):self.tau[j1] - 1],
                                x_distr=self.x_distr, x_hyper=self.x_hyper,
                                tau_s=self.tau[j0:j1 + 1] - self.tau[j0],
                                m=self.m[wj], y_init=self.y_init[wj])
                seg_j.get_y()

                self.accept_prob += np.log(self.w_hyper / (1 - self.w_hyper))

                keep_left = np.random.uniform() < (self.tau[j] - self.tau[j0]) / float(self.tau[j1] - self.tau[j0])

                if keep_left:
                    # left segment keeps m_j

                    self.accept_prob -= seg_j.llik_y_given_m(sub=range(j + 1 - j0, j1 + 1 - j0))

                    seg_j_prime = Segment(x=self.x[self.tau[j] - 1:self.tau[j1] - 1],
                                          x_distr=self.x_distr, x_hyper=self.x_hyper,
                                          tau_s=self.tau[j:j1 + 1] - self.tau[j],
                                          m=0, y_init=np.array([]))

                else:
                    self.accept_prob -= seg_j.llik_y_given_m(sub=range(1, j + 2 - j0))

                    y_full = np.concatenate([seg_j.y_init, seg_j.y])
                    y_full = y_full[self.tau[j] - self.tau[j0]:]

                    self.accept_prob += seg_j.llik_nu(y_full=y_full, m=seg_j.m)

                    seg_j_prime = Segment(x=self.x[self.tau[j0] - 1:self.tau[j] - 1],
                                          x_distr=self.x_distr, x_hyper=self.x_hyper,
                                          tau_s=self.tau[j0:j + 1] - self.tau[j0],
                                          m=0, y_init=np.array([]))

                seg_j_prime.m, m_prop_prob = seg_j_prime.prop_m(ratio=0)
                self.accept_prob -= m_prop_prob
                self.accept_prob += seg_j_prime.lprior_m()

                seg_j_prime.get_y_bounds()
                seg_j_prime.approx_map_gamma()
                seg_j_prime.get_y()
                self.accept_prob += seg_j_prime.prop_gamma(prob_ratio=0, gamma_prime=None)
                self.accept_prob += seg_j_prime.llik_y_given_m()

                if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                    self.w[j] = 1
                    self.w_ind = np.where(self.w == 1)[0]
                    self.m.insert(wj + keep_left, seg_j_prime.m)
                    self.y_init.insert(wj + keep_left, np.array(seg_j_prime.y_init))
                    self.accept_count['w'] += 1

            else:

                j0 = max(self.w_ind[self.w_ind < j])
                j1 = min(self.w_ind[self.w_ind > j])
                wj = np.where(self.w_ind == j)[0][0]

                self.accept_prob += np.log((1 - self.w_hyper) / self.w_hyper)

                fixed_left = np.random.uniform() < (self.tau[j] - self.tau[j0]) / float(self.tau[j1] - self.tau[j0])
                if fixed_left:

                    seg_j = Segment(x=self.x[(self.tau[j0] - 1):self.tau[j1] - 1],
                                    x_distr=self.x_distr, x_hyper=self.x_hyper,
                                    tau_s=self.tau[j0:j1 + 1] - self.tau[j0],
                                    m=self.m[wj - 1], y_init=self.y_init[wj - 1])
                    seg_j.get_y()

                    if seg_j.valid_y():
                        self.accept_prob += seg_j.llik_y_given_m(sub=range(j + 1 - j0, j1 + 1 - j0))

                        seg_j_prime = Segment(x=self.x[(self.tau[j] - 1):self.tau[j1] - 1],
                                              x_distr=self.x_distr, x_hyper=self.x_hyper,
                                              tau_s=self.tau[j:j1 + 1] - self.tau[j],
                                              m=self.m[wj], y_init=self.y_init[wj])

                        self.accept_prob += seg_j_prime.prop_m(ratio=2)
                        self.accept_prob -= seg_j_prime.lprior_m()

                        seg_j_prime.get_y_bounds()
                        seg_j_prime.approx_map_gamma()
                        seg_j_prime.get_y()
                        self.accept_prob -= seg_j_prime.prop_gamma(prob_ratio=0, gamma_prime=np.array(self.y_init[wj]))
                        self.accept_prob -= seg_j_prime.llik_y_given_m()

                        if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                            self.w[j] = 0
                            self.w_ind = np.where(self.w == 1)[0]
                            self.accept_count['w'] += 1
                            self.m.pop(wj)
                            self.y_init.pop(wj)
                    else:
                        pass

                else:
                    seg_j = Segment(x=self.x[(self.tau[j] - 1):self.tau[j + 1] - 1],
                                    x_distr=self.x_distr, x_hyper=self.x_hyper,
                                    tau_s=self.tau[j:j + 2] - self.tau[j],
                                    m=self.m[wj], y_init=self.y_init[wj])
                    seg_j.get_y()

                    self.accept_prob -= seg_j.llik_y_given_m(sub=[1])

                    seg_j.extend(n_x=self.x[self.tau[j0] - 1:self.tau[j] - 1])

                    if seg_j.valid_y():
                        seg_j.tau_s = self.tau[j0:j + 2] - self.tau[j0]

                        self.accept_prob += seg_j.llik_y_given_m()

                        seg_j_prime = Segment(x=self.x[(self.tau[j0] - 1):self.tau[j] - 1],
                                              x_distr=self.x_distr, x_hyper=self.x_hyper,
                                              tau_s=self.tau[j0:j + 1] - self.tau[j0],
                                              m=self.m[wj - 1], y_init=self.y_init[wj - 1])

                        self.accept_prob += seg_j_prime.prop_m(ratio=2)
                        self.accept_prob -= seg_j_prime.lprior_m()

                        seg_j_prime.get_y_bounds()
                        seg_j_prime.approx_map_gamma()
                        seg_j_prime.get_y()
                        self.accept_prob -= seg_j_prime.prop_gamma(prob_ratio=0,
                                                                   gamma_prime=np.array(self.y_init[wj - 1]))
                        self.accept_prob -= seg_j_prime.llik_y_given_m()

                        if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                            self.w[j] = 0
                            self.w_ind = np.where(self.w == 1)[0]
                            self.accept_count['w'] += 1
                            self.y_init[wj] = np.array(seg_j.y_init[:])
                            self.m.pop(wj - 1)
                            self.y_init.pop(wj - 1)

                    else:
                        pass
        else:
            pass

    def birth(self):

        if len(self.tau) < self.T - 1:

            sample_from = np.setdiff1d(range(2, self.T + 1), self.tau)
            tau_p = np.random.choice(sample_from)

            self.accept_prob = np.log(self.p / (1 - self.p))

            j = min(np.where(tau_p < self.tau)[0])

            if self.m_hyper == 1:
                seg_j = Segment(x=None, x_distr=self.x_distr, x_hyper=self.x_hyper, m=0)
                self.accept_prob += seg_j.llik_nu(y_full=self.x[(self.tau[j - 1] - 1):tau_p - 1], m=0)
                self.accept_prob += seg_j.llik_nu(y_full=self.x[tau_p - 1:self.tau[j] - 1], m=0)
                self.accept_prob -= seg_j.llik_nu(y_full=self.x[self.tau[j - 1] - 1:self.tau[j] - 1], m=0)

                if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                    self.tau = np.insert(self.tau, j, tau_p)
                    self.w = np.insert(self.w, j, 1)
                    self.w_ind = np.where(self.w == 1)[0]
                    self.m.insert(j - 1, 0)
                    self.y_init.insert(j - 1, np.array([]))
                    self.accept_count['birth'] += 1

            else:
                j0 = max(self.w_ind[self.w_ind < j])
                j1 = min(self.w_ind[self.w_ind >= j])

                wj = np.where(self.w_ind == j0)[0][0]

                fixed_left = np.random.uniform() < (tau_p - self.tau[j0]) / float(self.tau[j1] - self.tau[j0])

                if fixed_left:
                    seg_u = Segment(x=self.x[tau_p - 1:self.tau[j1] - 1],
                                    x_distr=self.x_distr, x_hyper=self.x_hyper,
                                    tau_s=np.append(tau_p, self.tau[j:j1 + 1]) - tau_p,
                                    m=self.m[wj], y_init=np.array([]))
                    prob_m = seg_u.prop_m(ratio=2) if 0 < self.w_hyper < 1 else 0.0
                else:
                    seg_u = Segment(x=self.x[self.tau[j0] - 1:tau_p - 1],
                                    x_distr=self.x_distr, x_hyper=self.x_hyper,
                                    tau_s=np.append(self.tau[j0:j], tau_p) - self.tau[j0],
                                    m=self.m[wj], y_init=np.array([]))
                    prob_m = seg_u.prop_m(ratio=2) if 0 < self.w_hyper < 1 else 0.0

                if 0 < self.w_hyper < 1:
                    w_p = np.random.uniform() < (1 - np.exp(prob_m))
                elif self.w_hyper == 0:
                    w_p = 0
                else:
                    w_p = 1

                #
                if w_p == 0:

                    if 0 < self.w_hyper < 1:
                        self.accept_prob -= prob_m
                        self.accept_prob += np.log(1 - self.w_hyper)

                    seg_p = Segment(x=self.x[self.tau[j0] - 1:self.tau[j] - 1],
                                    x_distr=self.x_distr, x_hyper=self.x_hyper,
                                    tau_s=self.tau[j0:j + 1] - self.tau[j0],
                                    m=self.m[wj], y_init=np.array(self.y_init[wj]))

                    seg_p.get_y()

                    self.accept_prob -= seg_p.llik_y_given_m(sub=[j - j0])

                    seg_p.tau_s = np.insert(seg_p.tau_s, j - j0, tau_p - self.tau[j0])

                    self.accept_prob += seg_p.llik_y_given_m(sub=[j - j0, j - j0 + 1])

                    if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                        self.tau = np.insert(self.tau, j, tau_p)
                        self.w = np.insert(self.w, j, w_p)
                        self.w_ind = np.where(self.w == 1)[0]

                        self.accept_count['birth'] += 1

                else:
                    if 0 < self.w_hyper < 1:
                        self.accept_prob -= np.log(1 - np.exp(prob_m))
                        self.accept_prob += np.log(self.w_hyper)

                    if fixed_left:

                        seg_u.m, m_prop_prob = seg_u.prop_m(ratio=0)
                        self.accept_prob -= m_prop_prob
                        self.accept_prob += seg_u.lprior_m()

                        seg_u.get_y_bounds()
                        seg_u.approx_map_gamma()

                        seg_u.get_y()

                        self.accept_prob += seg_u.prop_gamma(prob_ratio=0, gamma_prime=None)

                        self.accept_prob += seg_u.llik_y_given_m()

                        seg_p = Segment(x=self.x[self.tau[j0] - 1:self.tau[j1] - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j0:j1 + 1] - self.tau[j0],
                                        m=self.m[wj], y_init=np.array(self.y_init[wj]))

                        seg_p.get_y()

                        self.accept_prob -= seg_p.llik_y_given_m(sub=range(j - j0, j1 + 1 - j0))

                        seg_p.tau_s = np.insert(seg_p.tau_s, j - j0, tau_p - self.tau[j0])

                        self.accept_prob += seg_p.llik_y_given_m(sub=[j - j0])

                        if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                            self.tau = np.insert(self.tau, j, tau_p)
                            self.w = np.insert(self.w, j, w_p)
                            self.w_ind = np.where(self.w == 1)[0]

                            self.m.insert(wj + 1, seg_u.m)
                            self.y_init.insert(wj + 1, np.array(seg_u.y_init))

                            self.accept_count['birth'] += 1

                    else:

                        seg_u.m, m_prop_prob = seg_u.prop_m(ratio=0)
                        self.accept_prob -= m_prop_prob
                        self.accept_prob += seg_u.lprior_m()

                        seg_u.get_y_bounds()
                        seg_u.approx_map_gamma()
                        seg_u.get_y()
                        self.accept_prob += seg_u.prop_gamma(prob_ratio=0, gamma_prime=None)
                        self.accept_prob += seg_u.llik_y_given_m()

                        seg_p = Segment(x=self.x[self.tau[j0] - 1:self.tau[j] - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j0:j + 1] - self.tau[j0],
                                        m=self.m[wj], y_init=np.array(self.y_init[wj]))

                        seg_p.get_y()

                        self.accept_prob -= seg_p.llik_y_given_m()

                        y_full = np.concatenate([seg_p.y_init, seg_p.y])
                        y_full = y_full[tau_p - self.tau[j0]:]

                        self.accept_prob += seg_p.llik_nu(y_full=y_full, m=seg_p.m)

                        if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                            self.tau = np.insert(self.tau, j, tau_p)
                            self.w = np.insert(self.w, j, w_p)
                            self.w_ind = np.where(self.w == 1)[0]

                            self.m.insert(wj, seg_u.m)
                            self.y_init.insert(wj, np.array(seg_u.y_init))

                            self.y_init[wj + 1] = np.array(y_full[:seg_p.m])

                            self.accept_count['birth'] += 1

        else:
            pass

    def death(self):

        if len(self.tau) > 2:

            self.accept_prob = np.log((1 - self.p) / self.p)

            j = np.random.randint(low=1, high=len(self.tau) - 1)

            if self.m_hyper == 1:
                seg_j = Segment(x=None, x_distr=self.x_distr, x_hyper=self.x_hyper, m=0)
                self.accept_prob -= seg_j.llik_nu(y_full=self.x[self.tau[j - 1] - 1:self.tau[j] - 1], m=0)
                self.accept_prob -= seg_j.llik_nu(y_full=self.x[self.tau[j] - 1:self.tau[j + 1] - 1], m=0)
                self.accept_prob += seg_j.llik_nu(y_full=self.x[self.tau[j - 1] - 1:self.tau[j + 1] - 1], m=0)

                if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                    self.tau = np.delete(self.tau, j)
                    self.w = np.delete(self.w, j)
                    self.w_ind = np.where(self.w == 1)[0]

                    self.w_ind = np.where(self.w == 1)[0]
                    self.m.pop(j - 1)
                    self.y_init.pop(j - 1)

            else:

                j0 = max(self.w_ind[self.w_ind < j])
                j1 = min(self.w_ind[self.w_ind > j])
                wj = np.where(self.w_ind == j0)[0][0]

                fixed_left = np.random.uniform() < (self.tau[j] - self.tau[j0]) / float(self.tau[j1] - self.tau[j0])

                if self.w[j] == 0:

                    seg_p = Segment(x=self.x[self.tau[j0] - 1:self.tau[j1] - 1],
                                    x_distr=self.x_distr, x_hyper=self.x_hyper,
                                    tau_s=self.tau[j0:j1 + 1] - self.tau[j0],
                                    m=self.m[wj], y_init=np.array(self.y_init[wj]))
                    seg_p.get_y()

                    self.accept_prob -= seg_p.llik_y_given_m(sub=[j - j0, j - j0 + 1])

                    seg_p.tau_s = np.delete(seg_p.tau_s, j - j0)

                    self.accept_prob += seg_p.llik_y_given_m(sub=[j - j0])

                    if 0 < self.w_hyper < 1:
                        if fixed_left:
                            seg_u = Segment(x=self.x[self.tau[j] - 1:self.tau[j1] - 1],
                                            x_distr=self.x_distr, x_hyper=self.x_hyper,
                                            tau_s=self.tau[j:j1 + 1] - self.tau[j],
                                            m=self.m[wj], y_init=np.array([]))
                        else:
                            seg_u = Segment(x=self.x[self.tau[j0] - 1:self.tau[j] - 1],
                                            x_distr=self.x_distr, x_hyper=self.x_hyper,
                                            tau_s=self.tau[j0:j + 1] - self.tau[j0],
                                            m=self.m[wj], y_init=np.array([]))

                        prob_m = seg_u.prop_m(ratio=2)
                        self.accept_prob += prob_m

                        self.accept_prob -= np.log(1 - self.w_hyper)

                    if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                        self.tau = np.delete(self.tau, j)
                        self.w = np.delete(self.w, j)
                        self.w_ind = np.where(self.w == 1)[0]
                        self.accept_count['death'] += 1

                else:

                    if 0 < self.w_hyper < 1:
                        self.accept_prob -= np.log(self.w_hyper)

                    if fixed_left:

                        seg_p = Segment(x=self.x[self.tau[j0] - 1:self.tau[j1] - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j0:j1 + 1] - self.tau[j0],
                                        m=self.m[wj], y_init=np.array(self.y_init[wj]))
                        seg_p.get_y()

                        if seg_p.valid_y():
                            self.accept_prob -= seg_p.llik_y_given_m(sub=[j - j0])

                            seg_p.tau_s = np.delete(seg_p.tau_s, j - j0)

                            self.accept_prob += seg_p.llik_y_given_m(sub=range(j - j0, len(seg_p.tau_s)))

                            seg_u = Segment(x=self.x[self.tau[j] - 1:self.tau[j1] - 1],
                                            x_distr=self.x_distr, x_hyper=self.x_hyper,
                                            tau_s=self.tau[j:j1 + 1] - self.tau[j],
                                            m=self.m[wj + 1], y_init=np.array(self.y_init[wj + 1]))

                            if 0 < self.w_hyper < 1:
                                prop_mu, prop_mp = seg_u.prop_m(ratio=3, m_prime=self.m[wj])
                                self.accept_prob += prop_mu
                                self.accept_prob += np.log(1 - np.exp(prop_mp) ) if 1 - np.exp(prop_mp) > 0.0 else -np.inf
                            else:
                                prop_mu = seg_u.prop_m(ratio=2)
                                self.accept_prob += prop_mu

                            self.accept_prob -= seg_u.lprior_m()

                            seg_u.get_y_bounds()
                            seg_u.approx_map_gamma()
                            seg_u.get_y()
                            self.accept_prob -= seg_u.prop_gamma(prob_ratio=0,
                                                                 gamma_prime=np.array(self.y_init[wj + 1]))
                            self.accept_prob -= seg_u.llik_y_given_m()

                            if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                                self.tau = np.delete(self.tau, j)
                                self.w = np.delete(self.w, j)
                                self.w_ind = np.where(self.w == 1)[0]
                                self.m.pop(wj + 1)
                                self.y_init.pop(wj + 1)
                                self.accept_count['death'] += 1
                        else:
                            pass

                    else:

                        seg_p = Segment(x=self.x[self.tau[j] - 1:self.tau[j + 1] - 1],
                                        x_distr=self.x_distr, x_hyper=self.x_hyper,
                                        tau_s=self.tau[j:j + 2] - self.tau[j],
                                        m=self.m[wj + 1], y_init=np.array(self.y_init[wj + 1]))
                        seg_p.get_y()

                        self.accept_prob -= seg_p.llik_y_given_m(sub=[1])

                        seg_p.extend(n_x=self.x[self.tau[j0] - 1:self.tau[j] - 1])

                        if seg_p.valid_y():
                            seg_p.tau_s = np.array(self.tau[j0:j + 2] - self.tau[j0])
                            seg_p.tau_s = np.delete(seg_p.tau_s, j - j0)

                            self.accept_prob += seg_p.llik_y_given_m()

                            seg_u = Segment(x=self.x[self.tau[j0] - 1:self.tau[j] - 1],
                                            x_distr=self.x_distr, x_hyper=self.x_hyper,
                                            tau_s=self.tau[j0:j + 1] - self.tau[j0],
                                            m=self.m[wj], y_init=np.array(self.y_init[wj]))

                            if 0 < self.w_hyper < 1:
                                prop_mu, prop_mp = seg_u.prop_m(ratio=3, m_prime=self.m[wj + 1])
                                self.accept_prob += prop_mu
                                self.accept_prob += np.log(1 - np.exp(prop_mp)) #if 0 < self.w_hyper < 1 else 0.0
                            else:
                                prop_mu = seg_u.prop_m(ratio=2)
                                self.accept_prob += prop_mu

                            self.accept_prob -= seg_u.lprior_m()

                            seg_u.get_y_bounds()
                            seg_u.approx_map_gamma()
                            seg_u.get_y()
                            self.accept_prob -= seg_u.prop_gamma(prob_ratio=0, gamma_prime=np.array(self.y_init[wj]))
                            self.accept_prob -= seg_u.llik_y_given_m()

                            if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                                self.tau = np.delete(self.tau, j)
                                self.w = np.delete(self.w, j)
                                self.w_ind = np.where(self.w == 1)[0]
                                self.y_init[wj + 1] = np.array(seg_p.y_init)
                                self.m.pop(wj)
                                self.y_init.pop(wj)
                                self.accept_count['death'] += 1
                        else:
                            pass

        else:
            pass

    def prop_gamma(self):

        if self.m_hyper < 1:
            j = np.random.randint(len(self.m))

            if self.m[j] == 0:
                self.accept_count['gamma'] += 1
            else:
                wj0, wj1 = self.w_ind[j], self.w_ind[j + 1]
                seg = Segment(x=self.x[self.tau[wj0] - 1:self.tau[wj1] - 1],
                              x_distr=self.x_distr, x_hyper=self.x_hyper,
                              tau_s=self.tau[wj0:wj1 + 1] - self.tau[wj0],
                              m=self.m[j], y_init=self.y_init[j])

                seg.get_y_bounds()
                seg.get_y()
                self.accept_prob = -seg.llik_y_given_m()

                self.accept_prob += seg.prop_gamma()

                seg.get_y()
                self.accept_prob += seg.llik_y_given_m()

                if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                    self.y_init[j] = np.array(seg.y_init[:])
                    self.accept_count['gamma'] += 1
        else:
            pass

    def prop_m_and_gamma(self):

        if self.m_hyper < 1:
            j = np.random.randint(len(self.m))

            wj0, wj1 = self.w_ind[j], self.w_ind[j + 1]

            seg = Segment(x=self.x[self.tau[wj0] - 1:self.tau[wj1] - 1],
                          x_distr=self.x_distr, x_hyper=self.x_hyper,
                          m_hyper=self.m_hyper,
                          tau_s=self.tau[wj0:wj1 + 1] - self.tau[wj0],
                          m=self.m[j], y_init=self.y_init[j])

            m_prop, self.accept_prob = seg.prop_m()

            if m_prop == self.m[j]:
                if self.m[j] == 0:
                    self.accept_count['m_and_gamma'] += 1
                else:

                    seg.get_y()
                    self.accept_prob = -seg.llik_y_given_m()

                    seg.get_y_bounds()
                    self.accept_prob += seg.prop_gamma()

                    self.accept_prob += seg.llik_y_given_m()

                    if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                        self.y_init[j] = np.array(seg.y_init[:])
                        self.accept_count['m_and_gamma'] += 1
            else:

                self.accept_prob -= seg.lprior_m()
                seg.get_y_bounds()
                seg.approx_map_gamma()
                seg.get_y()
                self.accept_prob -= seg.prop_gamma(prob_ratio=0, gamma_prime=np.array(self.y_init[j]))
                self.accept_prob -= seg.llik_y_given_m()

                seg.m = m_prop
                self.accept_prob += seg.lprior_m()
                seg.get_y_bounds()
                seg.approx_map_gamma()
                seg.get_y()
                self.accept_prob += seg.prop_gamma(prob_ratio=0, gamma_prime=None)
                self.accept_prob += seg.llik_y_given_m()

                if self.accept_prob > 0 or np.exp(self.accept_prob) >= np.random.uniform(1):
                    self.y_init[j] = np.array(seg.y_init[:])
                    self.m[j] = seg.m
                    self.accept_count['m_and_gamma'] += 1
        else:
            pass

    ################
    # run sampler
    ################

    def run(self, num_iter, burn_in=0, thinning=1, move_prob=None):
        overall_start_time = time.time()

        # prob for moves ['shift', 'birth', 'death', 'gamma', 'm_and_gamma', 'w']
        move_prob = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1] if move_prob is None else move_prob
        moves = ['shift', 'birth', 'death', 'gamma', 'm_and_gamma', 'w']
        self.accept_count = {move: 0 for move in moves}
        self.accept_prop = {move: 0 for move in moves}

        for iteration in range(0, burn_in+num_iter):
            self.accept_prob = 0.0
            #print(iteration)
            if iteration == 12:
                print('stop')
            move_type = np.random.choice(['shift', 'birth', 'death', 'gamma', 'm_and_gamma', 'w'], 1, p=move_prob)[0]
            self.accept_prop[move_type] += 1

            if move_type == 'shift':
                self.shift()
            elif move_type == 'birth':
                self.birth()
            elif move_type == 'death':
                self.death()
            elif move_type == 'gamma':
                self.prop_gamma()
            elif move_type == 'm_and_gamma':
                self.prop_m_and_gamma()
            elif move_type == 'w':
                self.prop_w()
            else:
                raise NameError('move_type')

            if iteration >= burn_in and iteration % thinning == 0:
                self.sample_tau += [self.tau[1:-1].tolist()]
                self.sample_w += [self.w[1:-1].tolist()]
                self.sample_m += [self.m[:]]
                self.sample_y_init += [[y.tolist() for y in self.y_init]]

            if iteration % 5000 == 0:
                print('iter - ', iteration, ' time ', time.time()-overall_start_time)

        overall_end_time = time.time()
        print('sampler ran in ', overall_end_time - overall_start_time)

    ################
    # estimations and diagnostics
    ################

    def plot_trace(self, type_='k'):
        if type_ == 'k':
            plt.plot([len(t) for t in self.sample_tau], 'k')
        elif type_ == 'kw':
            plt.plot([len(t) for t in self.sample_tau], '--')
            kw = [sum(w) for w in self.sample_w]
            plt.plot(kw, 'k')
        elif type_ == 'tau':
            tau_trace = pd.DataFrame(self.sample_tau)
            plt.plot(tau_trace, 's', marker='+', markersize=5, color='k')
        elif type_ == 'tauw':
            w_trace = pd.DataFrame(self.sample_w)
            tau_trace = pd.DataFrame(self.sample_tau)
            tauw_trace = w_trace * tau_trace
            tauw_trace[tauw_trace == 0] = np.nan
            plt.plot(tauw_trace, 's', marker='+', markersize=5, color='k')
        elif type_ == 'm':
            m_trace = pd.DataFrame(self.sample_m)
            plt.plot(m_trace.iloc[:, :], markersize=5)
        else:
            print('options are k, kw, tau, tauw')

    def get_map(self, type_=''):
        self.k_MAP = pd.Series([len(t) for t in self.sample_tau]).mode()[0]

        if self.k_MAP == 0:
            self.tau_MAP = np.array([])
        else:
            tau_sample = pd.DataFrame(self.sample_tau)
            tau_sample = tau_sample[tau_sample.count(1) == self.k_MAP]
            self.tau_MAP = np.array(tau_sample.mode().iloc[0])
            self.tau_MAP = self.tau_MAP[~np.isnan(self.tau_MAP)]

        if 'm' in type_ and self.w_hyper == 1:
            m_sample = pd.DataFrame(self.sample_m)
            m_sample = m_sample[m_sample.count(1) == self.k_MAP+1]
            self.m_MAP = m_sample.mode().iloc[0, :self.k_MAP+1].values

        if 'w' in type_:
            self.kw_MAP = pd.Series([sum(w) for w in self.sample_w]).mode()[0]

            if self.kw_MAP == 0:
                self.tauw_MAP = np.array([])
            else:
                sample_tauw = pd.DataFrame(self.sample_tau)*pd.DataFrame(self.sample_w)
                sample_tauw[sample_tauw == 0] = np.nan
                sample_tauw = sample_tauw[sample_tauw.count(1) == self.kw_MAP]
                sample_tauw = pd.DataFrame([tauw[~np.isnan(tauw)] for tauw in sample_tauw.values])
                self.tauw_MAP = np.array(sample_tauw.mode().iloc[0])

            if 'm' in type_:
                m_sample = pd.DataFrame(self.sample_m)
                m_sample = m_sample[m_sample.count(1) == self.k_MAP + 1]
                self.m_MAP = m_sample.mode().iloc[0, :self.k_MAP + 1].values


    def plot_map(self, type_='tau'):

        #plt.plot(self.x, marker='x', color='k')
        plt.plot(np.arange(len(self.x)), self.x, marker='x', color='k', markersize=3, linestyle='None')

        if type_ == 'tau':
            if self.tau_MAP is None:
                self.get_map()
            for t in self.tau_MAP:
                plt.axvline(x=t - 1, linestyle='--')

        elif type_ == 'tauw':
            if self.tauw_MAP is None:
                self.get_map()
            for t in self.tauw_MAP:
                plt.axvline(x=t - 1, linestyle='--')
        else:
            pass

    def post_distribution(self):

        post_distr_k = pd.Series([len(t) for t in self.sample_tau]).value_counts()
        self.post_distr_k = post_distr_k / post_distr_k.sum()
        self.k_MAP = self.post_distr_k.index[0]

        post_distr_kw = pd.Series([sum(w) for w in self.sample_w]).value_counts()
        self.post_distr_kw = post_distr_kw/post_distr_kw.sum()
        self.kw_MAP = self.post_distr_kw.index[0]

        if self.k_MAP == 0:
            self.post_distr_tau = pd.DataFrame([])
        else:
            tau_sample = pd.DataFrame(self.sample_tau)
            tau_sample = tau_sample[tau_sample.count(1) == self.k_MAP]
            len_sample = len(tau_sample)
            post_distr_tau = tau_sample.apply(pd.value_counts).fillna(0).sum(axis=1)
            self.post_distr_tau = post_distr_tau / len_sample

        if self.kw_MAP == 0:
            self.post_distr_tauw = pd.DataFrame([])
        else:
            sample_tauw = pd.DataFrame(self.sample_tau) * pd.DataFrame(self.sample_w)
            sample_tauw[sample_tauw == 0] = np.nan
            sample_tauw = sample_tauw[sample_tauw.count(1) == self.kw_MAP]
            sample_tauw = pd.DataFrame([tauw[~np.isnan(tauw)] for tauw in sample_tauw.values])
            len_sample = len(sample_tauw)
            post_distr_tau = sample_tauw.apply(pd.value_counts).fillna(0).sum(axis=1)
            self.post_distr_tauw = post_distr_tau / len_sample

