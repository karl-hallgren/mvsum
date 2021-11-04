"""
class for a segment for the moving-sum changepoint model
"""

import numpy as np
from functools import reduce
import scipy.special as special
from scipy.stats import nbinom
from scipy.stats import gamma as scigamma
from scipy.stats import norm as scinorm


class Segment:
    def __init__(self, x=None, x_distr=None, x_hyper=None, tau_s=None, m=None, y_init=None, m_hyper=None):

        self.x = np.array([]) if x is None else x
        self.x_distr = x_distr
        self.x_hyper = x_hyper
        self.tau_s = np.array([0, len(self.x)]) if tau_s is None else tau_s
        self.y = np.array([])
        self.m = 0 if m is None else m
        self.y_init = np.array([]) if y_init is None else y_init
        self.y_bounds = None
        self.m_hyper = 0.1 if m_hyper is None else m_hyper
        self.theta = []

    ################
    # SIMULATIONS
    ################
    def sample_theta(self):
        if self.x_distr == 'Poisson':
            # x_hyper = [alpha, beta]
            return np.random.gamma(shape=self.x_hyper[0], scale=1 / self.x_hyper[1])
        elif self.x_distr == 'Normal':
            # x_hyper = [mu_0, sigma_0^2, sigma^2]
            return np.random.normal(self.x_hyper[0], np.sqrt(self.x_hyper[1]) )
        elif self.x_distr == 'NormalGamma':
            # x_hyper = [mu_0, lambda, alpha, beta]
            rho_ = np.random.gamma(shape=self.x_hyper[2], scale=1/self.x_hyper[3])
            mu_ = np.random.normal(loc=self.x_hyper[0], scale=np.sqrt(1/(rho_ * self.x_hyper[1])) )
            return mu_, rho_
        elif self.x_distr == 'Gamma':
            # x_hyper = [alpha, beta, lambda]
            return np.random.gamma(shape=self.x_hyper[0], scale=1/self.x_hyper[1])
        elif self.x_distr == 'NegativeBinomial':
            # x_hyper = [r, alpha, beta]
            return np.random.beta(a=self.x_hyper[1], b=self.x_hyper[2])
        else:
            pass

    def sample_y(self, j, theta=None):
        n = self.tau_s[j] - self.tau_s[j-1] + (self.m if j == 1 else 0)
        theta = self.sample_theta() if theta is None else theta
        self.theta += [theta]
        if self.x_distr == 'Poisson':
            return np.random.poisson(theta/(self.m+1), n)
        elif self.x_distr == 'Normal':
            return np.random.normal(theta/(self.m+1), np.sqrt(self.x_hyper[2]/(self.m+1)), n)
        elif self.x_distr == 'NormalGamma':
            return np.random.normal(theta[0] / (self.m + 1), np.sqrt(1 / (theta[1]*(self.m + 1))), n)
        elif self.x_distr == 'Gamma':
            return np.random.gamma(shape=self.x_hyper[2]/(self.m+1), scale=1/theta, size=n)
        elif self.x_distr == 'NegativeBinomial':
            return np.random.negative_binomial(n=self.x_hyper[0]/(self.m+1), p=1-theta, size=n)
        else:
            pass

    def sample_x(self, theta=None):
        self.y = np.concatenate([self.sample_y(j, theta[j-1] if theta is not None else None) for j in range(1, len(self.tau_s))])
        self.y_init = self.y[:self.m]
        y_cusum = self.y.cumsum()
        y_cusum[(self.m+1):] = y_cusum[(self.m+1):] - y_cusum[:-(self.m+1)]
        self.x = y_cusum[self.m:]
        self.y = self.y[self.m:]

    ####################
    # Latent variables
    ####################
    def get_y(self):
        if self.m == 0:
            self.y = np.array(self.x)
        else:
            n = len(self.x)
            self.y = np.zeros(n)
            xdiff = np.diff(np.insert(self.x, 0, 0))
            for r in range(self.m+1):
                sub_i = np.where(np.arange(n) % (self.m+1) == r)[0]
                gamma_ell = -sum(self.y_init) if r == 0 else self.y_init[r-1]
                self.y[sub_i] = np.cumsum(xdiff[sub_i]) + gamma_ell

    def get_y_bounds(self):

        if self.x_distr in ['Poisson', 'Gamma', 'NegativeBinomial']:
            if self.m == 0:
                pass
            else:
                xdiff = np.append(0, -np.diff(self.x))
                n = len(xdiff)

                sub_i = np.append(0, np.where(np.arange(n) % (self.m + 1) == 1)[0])
                sub_i = np.repeat([sub_i], self.m+1, axis=0)
                sub_i[:, 1:] = sub_i[:, 1:] + np.arange(self.m+1)[:, None]
                sub_i[sub_i >= n] = 0
                self.y_bounds = xdiff[sub_i].cumsum(1).max(1)
                self.y_bounds[-1] = self.x[0]-self.y_bounds[-1]
        else:
            pass

    def valid_y(self, bounds=0):
        if self.x_distr in ['Poisson', 'Gamma', 'NegativeBinomial']:
            if self.m == 0:
                return 1
            else:
                if bounds == 0:
                    if self.x_distr == 'Gamma':
                        return (np.concatenate([self.y_init, self.y]) > 0).all()
                    else:
                        return (np.concatenate([self.y_init, self.y]) >= 0).all()
                else:
                    return (self.y_bounds[-1] - sum(self.y_bounds[:-1]) ) > 0
        else:
            return 1

    def extend(self, n_x, direction='left'):
        if direction == 'left':
            n = len(n_x)
            if self.m == 0:
                self.y = np.concatenate([n_x[:], self.y[:]])
            else:
                n_y = np.zeros(n)
                xdiff = np.diff(np.insert(n_x[::-1], 0, 0))
                r_y_init = self.y_init[::-1]
                for r in range(self.m + 1):
                    sub_i = np.where(np.arange(n) % (self.m + 1) == r)[0]
                    gamma_ell = -sum(self.y_init) if r == 0 else r_y_init[r - 1]
                    n_y[sub_i] = np.cumsum(xdiff[sub_i]) + gamma_ell
                y_full = np.concatenate([n_y[::-1], self.y_init[:], self.y[:]])
                if self.x_distr in ['Poisson', 'NegativeBinomial']:
                    y_full = y_full.astype(int)
                self.y = y_full[self.m:]
                self.y_init = y_full[:self.m]
            self.tau_s[1:] += n

        else:
            pass

    ####################
    # Likelihood
    ####################

    def llik_nu(self, y_full=None, m=None):

        if self.x_distr == 'Poisson':

            n = len(y_full)
            alpha_y_dot = self.x_hyper[0] + np.sum(y_full)

            out = special.gammaln(alpha_y_dot) - special.gammaln(self.x_hyper[0])
            out += self.x_hyper[0]*np.log(self.x_hyper[1]*(m+1)) - alpha_y_dot*np.log(n+self.x_hyper[1]*(m+1))
            out -= sum(special.gammaln(y_full + 1))
            return out

        elif self.x_distr == 'Normal':
            n = len(y_full)
            sigma_dot = self.x_hyper[1]*self.x_hyper[2]
            sigma_mn = self.x_hyper[2] + self.x_hyper[1]*n/(m+1)

            out = -(n/2)*np.log(2*np.pi*self.x_hyper[2]/(m+1)) - 0.5 * np.log(2*np.pi*self.x_hyper[1]) + 0.5 * np.log( sigma_dot /sigma_mn )
            out += ( self.x_hyper[1]*sum(y_full) + self.x_hyper[0]*self.x_hyper[2]  )**2 / (2* sigma_dot * sigma_mn)
            out -= ( (m+1)*self.x_hyper[1]*sum( y_full**2 ) + self.x_hyper[0]**2 * self.x_hyper[2] ) / (2*sigma_dot)

            return out

        elif self.x_distr == 'NormalGamma':
            n = len(y_full)
            y_dot = sum(y_full)
            y_dot_sq = sum(y_full**2)

            lambda_p = (n + (m+1)*self.x_hyper[1]) / (m + 1)
            alpha_p = n/2 + self.x_hyper[2]
            beta_p = self.x_hyper[3] + (m+1)*y_dot_sq/2 + self.x_hyper[1] * self.x_hyper[0]**2 / 2
            beta_p -= (m+1) * (y_dot + self.x_hyper[1] * self.x_hyper[0])**2 / (2 * (n + (m+1)*self.x_hyper[1]))

            out = - (n/2) * np.log(2*np.pi) + (n/2) * np.log(m+1)
            out += self.x_hyper[2]*np.log(self.x_hyper[3]) - special.gammaln(self.x_hyper[2])
            out += special.gammaln(alpha_p) - alpha_p * np.log(beta_p)
            out += (1/2) * np.log(self.x_hyper[1]/lambda_p)

            return out

        elif self.x_distr == 'Gamma':
            n = len(y_full)
            lambda_m = self.x_hyper[2] / (m + 1)

            out = special.gammaln( self.x_hyper[0] + n*lambda_m) - special.gammaln(self.x_hyper[0]) - n*special.gammaln(lambda_m)
            out += self.x_hyper[0]*np.log(self.x_hyper[1])
            out += (lambda_m-1)*sum(np.log(y_full))
            out -= (self.x_hyper[0] + n*lambda_m)* np.log(self.x_hyper[1] + sum(y_full))
            return out

        elif self.x_distr == 'NegativeBinomial':
            n = len(y_full)
            y_dot = sum(y_full)
            r_m = self.x_hyper[0] / (m + 1)

            out = special.gammaln(self.x_hyper[1]+self.x_hyper[2]) - special.gammaln(self.x_hyper[1]) - special.gammaln(self.x_hyper[2])
            out += special.gammaln(y_dot + self.x_hyper[1]) + special.gammaln(n * r_m + self.x_hyper[2])
            out -= special.gammaln(y_dot + self.x_hyper[1] + n * r_m + self.x_hyper[2])
            out += sum(special.gammaln(y_full + r_m))
            out -= sum(special.gammaln(y_full + 1))
            out -= n*special.gammaln(r_m)
            return out

        else:
            raise NameError('Need to give correct x_distr')

    def llik_y_given_m(self, sub=None):

        sub = range(1, len(self.tau_s)) if sub is None else sub

        out = 0.0
        for i in sub:
            y_full = self.y[self.tau_s[i-1]:self.tau_s[i]]
            if i == 1:
                y_full = np.concatenate([self.y_init, y_full])
            out += self.llik_nu(y_full=y_full, m=self.m)

        return out

    def llik_nu_vec(self, y_full=None, m=None):

        if self.x_distr == 'Poisson':

            n = y_full.shape[1]
            alpha_y_dot = self.x_hyper[0] + y_full.sum(1)

            out = special.gammaln(alpha_y_dot) - special.gammaln(self.x_hyper[0])
            out += self.x_hyper[0]*np.log(self.x_hyper[1]*(m+1)) - alpha_y_dot*np.log(n+self.x_hyper[1]*(m+1))
            out -= special.gammaln(y_full + 1).sum(1)

            return out

        elif self.x_distr == 'Normal':
            n = y_full.shape[1]
            sigma_dot = self.x_hyper[1]*self.x_hyper[2]
            sigma_mn = self.x_hyper[2] + self.x_hyper[1]*n/(m+1)

            sum_y_full_square = y_full**2
            sum_y_full_square = sum_y_full_square.sum(1)

            out = -(n/2)*np.log(2*np.pi*self.x_hyper[2]/(m+1)) - 0.5 * np.log(2*np.pi*self.x_hyper[1]) + 0.5 * np.log( sigma_dot /sigma_mn )
            out += ( self.x_hyper[1]*y_full.sum(1) + self.x_hyper[0]*self.x_hyper[2]  )**2 / (2* sigma_dot * sigma_mn)
            out -= ( (m+1)*self.x_hyper[1]*sum_y_full_square + self.x_hyper[0]**2 * self.x_hyper[2] ) / (2*sigma_dot)

            return out

        elif self.x_distr == 'NormalGamma':
            n = y_full.shape[1]
            y_dot = y_full.sum(1)
            y_dot_sq = y_full**2
            y_dot_sq = y_dot_sq.sum(1)

            lambda_p = (n + (m+1)*self.x_hyper[1]) / (m + 1)
            alpha_p = n/2 + self.x_hyper[2]
            beta_p = self.x_hyper[3] + (m+1)*y_dot_sq/2 + self.x_hyper[1] * self.x_hyper[0]**2 / 2
            beta_p -= (m+1) * (y_dot + self.x_hyper[1] * self.x_hyper[0])**2 / (2 * (n + (m+1)*self.x_hyper[1]))

            out = - (n/2) * np.log(2*np.pi) + (n/2) * np.log(m+1)
            out += self.x_hyper[2]*np.log(self.x_hyper[3]) - special.gammaln(self.x_hyper[2])
            out += special.gammaln(alpha_p) - alpha_p * np.log(beta_p)
            out += (1/2) * np.log(self.x_hyper[1]/lambda_p)

            return out

        elif self.x_distr == 'Gamma':
            n = y_full.shape[1]
            lambda_m = self.x_hyper[2] / (m + 1)

            out = special.gammaln( self.x_hyper[0] + n*lambda_m) - special.gammaln(self.x_hyper[0]) - n*special.gammaln(lambda_m)
            out += self.x_hyper[0]*np.log(self.x_hyper[1])
            out += (lambda_m-1)*np.log(y_full).sum(1)
            out -= (self.x_hyper[0] + n*lambda_m) * np.log(self.x_hyper[1] + y_full.sum(1))

            return out

        elif self.x_distr == 'NegativeBinomial':
            n = y_full.shape[1]
            y_dot = y_full.sum(1)
            r_m = self.x_hyper[0] / (m + 1)

            out = special.gammaln(self.x_hyper[1]+self.x_hyper[2]) - special.gammaln(self.x_hyper[1]) - special.gammaln(self.x_hyper[2])
            out += special.gammaln(y_dot + self.x_hyper[1]) + special.gammaln(n * r_m + self.x_hyper[2])
            out -= special.gammaln(y_dot + self.x_hyper[1] + n * r_m + self.x_hyper[2])

            out += special.gammaln(y_full + r_m).sum(1)
            out -= special.gammaln(y_full + 1).sum(1)
            out -= n*special.gammaln(r_m)
            return out

        else:
            raise NameError('Need to give correct x_distr')

    ####################
    # Propose latent variables
    ####################

    def get_Y_r(self, r, stype, max_len=150, gamma_r=None, interval=0.99):

        Y_r = []
        bin_Y_r = []

        #if self.x_distr == 'Poisson':
        if self.x_distr in ['Poisson', 'NegativeBinomial']:

            Y_r_start = self.y_bounds[r]
            Y_r_stop = self.y_bounds[-1] - sum(self.y_init[:r]) - sum(self.y_init[r + 1:])
            Y_r = np.arange(start=Y_r_start, stop=Y_r_stop + 1)

            if stype == 'ApprGibbs' and len(Y_r) > max_len:
                x_ = self.x[:self.tau_s[1]]
                if self.x_distr == 'Poisson':
                    n = len(x_)
                    densities = nbinom.pmf(Y_r, sum(x_) + self.x_hyper[0], 1 - 1 / ((self.x_hyper[1] + n) * (self.m+1) + 1))
                elif self.x_distr == 'NegativeBinomial':
                    n = len(x_)
                    x_dot = sum(x_)
                    hat_theta = (self.x_hyper[1] + x_dot) / (self.x_hyper[1] + x_dot + self.x_hyper[2] + n*self.x_hyper[0])
                    densities = nbinom.pmf(Y_r, self.x_hyper[0]/(self.m + 1), 1 - hat_theta)
                Y_r = Y_r[densities.argsort()][-max_len:]
                if self.y_init[r] not in Y_r:
                    Y_r = np.append(self.y_init[r], Y_r)
                if gamma_r is not None and gamma_r not in Y_r:
                    Y_r = np.append(gamma_r, Y_r)
                Y_r.sort()

            bin_Y_r = np.ones(len(Y_r))

        elif self.x_distr == 'Gamma':

            max_len = 100

            x_ = self.x[:self.tau_s[1]]
            n = len(x_)
            hat_theta = (n * self.x_hyper[2] + self.x_hyper[0]) / (self.x_hyper[1] + sum(x_))
            low, up = scigamma.interval(interval, a=self.x_hyper[2] / (self.m + 1), scale=1 / hat_theta)

            U = self.y_bounds[-1] - sum(self.y_init[:r]) - sum(self.y_init[r + 1:])

            Y_r_start = max(low, self.y_bounds[r])
            Y_r_stop = min(up, U)
            if Y_r_stop < Y_r_start:
                Y_r_start = self.y_bounds[r]
                Y_r_stop = U


            Y_r = np.linspace(Y_r_start, Y_r_stop, num=max_len + 2)[1:-1]

            #Y_r = np.linspace(Y_r_start, Y_r_stop, num=max_len+1)[:-1]
            step = Y_r[1] - Y_r[0]

            if self.y_init[r] not in Y_r:
                Y_r = np.append(Y_r, self.y_init[r])

            if gamma_r is not None and gamma_r not in Y_r:
                Y_r = np.append(Y_r, gamma_r)

            Y_r.sort()
            bin_Y_r = np.append(np.diff(Y_r), step)
            bin_Y_r[bin_Y_r > step] = step

            shift = np.arange(len(Y_r))
            to_rem = [np.where(Y_r == self.y_init[r])[0][0]]
            if gamma_r is not None and gamma_r in Y_r:
                to_rem += [ np.where(Y_r == gamma_r)[0][0]]

            to_rem = list(set(to_rem))
            to_rem.sort()
            for to_r in to_rem[::-1]:
                shift = np.delete(shift, to_r)

            Y_r[shift] += np.random.uniform(0, 1, len(shift))*bin_Y_r[shift]

        else:

            if self.x_distr == 'Normal':
                x_ = self.x[:self.tau_s[1]]
                n = len(x_)

                loc_ = ( sum(x_)*self.x_hyper[1] + self.x_hyper[0]*self.x_hyper[2]) / (n * self.x_hyper[1] + self.x_hyper[2])
                loc_ = loc_ / (self.m+1)

                var = self.x_hyper[2]/(self.m+1) + self.x_hyper[1]*self.x_hyper[2]/( ( n*self.x_hyper[1] + self.x_hyper[2] ) * (self.m + 1) )

                Y_r_start, Y_r_stop = scinorm.interval(interval, loc=loc_, scale=np.sqrt(var))
                Y_r = np.linspace(Y_r_start, Y_r_stop, num=max_len + 1)[:-1]

            elif self.x_distr == 'NormalGamma':
                # correct posterior predictive is non central t distribution
                x_ = self.x[:self.tau_s[1]]
                x_dot = sum(x_)
                x_dot_sq = sum(x_**2)
                n = len(x_)

                mu_p = (x_dot + self.x_hyper[0]*self.x_hyper[1]) / (n + self.x_hyper[1])

                alpha_p = n / 2 + self.x_hyper[2]
                beta_p = self.x_hyper[3] + x_dot_sq / 2 + self.x_hyper[1] * self.x_hyper[0] ** 2 / 2
                beta_p -= (x_dot + self.x_hyper[1] * self.x_hyper[0]) ** 2 / (2 * (n + self.x_hyper[1]))

                rho_p = alpha_p / beta_p

                mu_p = mu_p / (self.m + 1)
                rho_p = rho_p * (self.m + 1)

                Y_r_start, Y_r_stop = scinorm.interval(interval, loc=mu_p, scale=np.sqrt(1/rho_p))
                Y_r = np.linspace(Y_r_start, Y_r_stop, num=max_len + 1)[:-1]

            #Y_r = np.linspace(Y_r_start, Y_r_stop, num=max_len+1)[:-1]
            step = Y_r[1] - Y_r[0]

            if self.y_init[r] not in Y_r:
                Y_r = np.append(Y_r, self.y_init[r])

            if gamma_r is not None and gamma_r not in Y_r:
                Y_r = np.append(Y_r, gamma_r)

            Y_r.sort()
            bin_Y_r = np.append(np.diff(Y_r), step)
            bin_Y_r[bin_Y_r > step] = step

            shift = np.arange(len(Y_r))
            to_rem = [np.where(Y_r == self.y_init[r])[0][0]]
            if gamma_r is not None and gamma_r in Y_r:
                to_rem += [ np.where(Y_r == gamma_r)[0][0]]

            to_rem = list(set(to_rem))
            to_rem.sort()
            for to_r in to_rem[::-1]:
                shift = np.delete(shift, to_r)

            Y_r[shift] += np.random.uniform(0, 1, len(shift))*bin_Y_r[shift]

        return Y_r, bin_Y_r

    def prop_gamma(self, stype='ApprGibbs', prob_ratio=1, gamma_prime=None):

        out = 0.0

        r_indices = np.random.permutation(self.m)

        if gamma_prime is not None and self.x_distr in ['Poisson', 'Gamma', 'NegativeBinomial']:
            for r in range(self.m):
                if gamma_prime[r_indices[r]] > self.y_bounds[-1] - sum(gamma_prime[r_indices[:r]]) - sum(self.y_init[r_indices[r+1:]]):
                    gamma_diff = gamma_prime - self.y_init
                    r_indices = gamma_diff.argsort()
                    break

        for r in r_indices:     # range(self.m):

            if 'Gibbs' in stype:

                Y_r, bin_Y_r = self.get_Y_r(r=r, stype=stype, gamma_r=gamma_prime[r] if gamma_prime is not None else None)
                #print(Y_r)
                n = len(self.y)
                y_full = np.concatenate([self.y_init, self.y])
                sub_r = np.where(np.arange(n) % (self.m + 1) == r + 1)[0] + self.m
                sub_r = np.append([r], sub_r)
                sub_0 = np.where(np.arange(n) % (self.m + 1) == 0)[0] + self.m

                y_full = np.array([y_full for _ in range(len(Y_r))])
                y_full[:, sub_r] += Y_r[:, None] - self.y_init[r]
                y_full[:, sub_0] += -Y_r[:, None] + self.y_init[r]

                densities = np.zeros(len(Y_r))
                for j in range(1, len(self.tau_s)):

                    y_density = y_full[:, self.tau_s[j-1] + self.m:self.tau_s[j] + self.m] if j > 1 else y_full[:, self.tau_s[j-1]:self.tau_s[j] + self.m]
                    densities += self.llik_nu_vec(y_full=y_density, m=self.m)
                if -np.inf in densities:
                    ind_inf = np.where(densities > -np.inf)[0]
                    if len(ind_inf) == 0:
                        densities = np.ones(len(densities))
                    else:
                        densities = np.exp(densities - max(densities[ind_inf]))
                else:
                    densities = np.exp(densities - max(densities))

                densities = densities / sum(densities)

                norm_constant = sum(densities * bin_Y_r)  # 1 if discrete data. Relevant for continuous data only

                if prob_ratio:
                    prob = densities[np.where(Y_r == self.y_init[r])][0]/norm_constant
                    out += np.log(prob) if prob > 0 else -np.inf

                if gamma_prime is None:
                    try:
                        self.y_init[r] = np.random.choice(a=Y_r, p=densities)
                    except:
                        print(densities)
                        break
                else:
                    self.y_init[r] = gamma_prime[r]

                ind_prop = np.where(Y_r == self.y_init[r])

                prob = densities[ind_prop][0]/norm_constant
                out -= np.log(prob) if prob > 0 else -np.inf

                if len(ind_prop[0]) > 1:
                    ind_prop = ind_prop[0][0]

                self.y[:] = y_full[ind_prop, self.m:]

            else:
                # Posterior predictive. Not doing this anymore
                pass
        return out

    def approx_map_gamma(self):

        if self.m == 0:
            y_init = np.array([])
        else:
            x_ = self.x[:self.tau_s[1]]

            if self.x_distr == 'Normal':
                n = len(x_)

                post_mean = ( sum(x_)*self.x_hyper[1] + self.x_hyper[0]*self.x_hyper[2] ) / ( n *self.x_hyper[1] + self.x_hyper[2])
                post_mean = post_mean / (self.m+1)

                y_init = np.array([post_mean for _ in range(self.m)])

            elif self.x_distr == 'NormalGamma':
                n = len(x_)
                x_dot = sum(x_)
                post_mean = (x_dot + self.x_hyper[0] * self.x_hyper[1]) / (n + self.x_hyper[1])
                post_mean = post_mean / (self.m + 1)

                y_init = np.array([post_mean for _ in range(self.m)])

            elif self.x_distr == 'Gamma':
                y_init = np.array(self.y_bounds[:-1])
                hat_theta = (len(x_) * self.x_hyper[2] + self.x_hyper[0]) / (self.x_hyper[1] + sum(x_))
                post_mean = self.x_hyper[2] / (hat_theta * (self.m + 1))

                for r in range(self.m):
                    if post_mean <= self.y_bounds[r]:
                        U = self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:])
                        y_init[r] = self.y_bounds[r] + (U-self.y_bounds[r]) / (self.m + 1)  #+ (self.y_bounds[-1] - sum(self.y_bounds[:-1]))/(self.m + 1)
                    else:
                        if post_mean < self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:]):
                            y_init[r] = post_mean
                        else:
                            #r_up += [r]
                            U = self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:])
                            y_init[r] = U - (U-self.y_bounds[r])/(self.m + 1)
                    if y_init[r] <= self.y_bounds[r]:
                        print('issue')

            elif self.x_distr == 'Poisson':
                y_init = np.array(self.y_bounds[:-1])

                post_mean = int((sum(x_) + self.x_hyper[0]) / ((len(x_) + self.x_hyper[1]) * (self.m + 1)))

                for r in range(self.m):
                    if post_mean < self.y_bounds[r]:
                        y_init[r] = self.y_bounds[r]
                    else:
                        if post_mean < self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:]):
                            y_init[r] = post_mean
                        else:
                            y_init[r] = self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:])
            elif self.x_distr == 'NegativeBinomial':
                y_init = np.array(self.y_bounds[:-1])

                n = len(x_)
                x_dot = sum(x_)
                theta_hat = (self.x_hyper[1] + x_dot) / (self.x_hyper[1] + x_dot + self.x_hyper[2] + n * self.x_hyper[0])
                post_mean = self.x_hyper[0] * theta_hat / (1-theta_hat)
                post_mean = post_mean / (self.m+1)

                for r in range(self.m):
                    if post_mean < self.y_bounds[r]:
                        y_init[r] = self.y_bounds[r]
                    else:
                        if post_mean < self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:]):
                            y_init[r] = post_mean
                        else:
                            y_init[r] = self.y_bounds[-1] - sum(y_init[:r]) - sum(y_init[r + 1:])

            else:
                raise NameError('Need to give correct x_distr')


        self.y_init = np.array(y_init)

    ####################
    # Propose m
    ####################

    def factors(self, n):
        ''' from https://stackoverflow.com/questions/6800193 '''
        #return set(reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))
        return np.array( reduce(list.__add__, ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)) )

    def get_M(self, max_m, xdiff=None):

        xdiff = -np.diff(self.x) if xdiff is None else xdiff

        if self.x_distr in ['Poisson', 'Gamma', 'NegativeBinomial']:
            out = np.zeros(max_m+1)
            out[0] = 1

            xdiff = np.append(0, xdiff)
            n = len(xdiff)

            for m in range(max_m, 0, -1):
                if out[m] == 0:
                    sub_i = np.append(0, np.where(np.arange(n) % (m + 1) == 1)[0])
                    sub_i = np.repeat([sub_i], m+1, axis=0)
                    sub_i[:, 1:] = sub_i[:, 1:] + np.arange(m+1)[:, None]
                    sub_i[sub_i >= n] = 0
                    bounds = xdiff[sub_i].cumsum(1).max(1)
                    if self.x[0] - sum(bounds) > 0:
                        out[self.factors(m + 1)-1] = 1
                else:
                    pass
            return out
        else:
            return np.ones(max_m+1)

    def lprior_m(self, m_s=None):
        m_s = self.m if m_s is None else m_s
        out = m_s*np.log(1-self.m_hyper) + np.log(self.m_hyper)
        return out

    def jump_estim_llik(self, x_, m_s):
        xdiff = np.diff(x_)
        if self.x_distr == 'Poisson':
            estim_var = (sum(x_) + self.x_hyper[0]) / (len(x_) + self.x_hyper[1])
            estim_var = estim_var / (m_s + 1)
        elif self.x_distr == 'Gamma':
            hat_theta = (len(x_) * self.x_hyper[2] + self.x_hyper[0]) / (self.x_hyper[1] + sum(x_))
            estim_var = self.x_hyper[2] / ((m_s + 1) * hat_theta ** 2)
        elif self.x_distr == 'Normal':
            estim_var = self.x_hyper[2] / (m_s + 1)
        elif self.x_distr == 'NormalGamma':

            x_dot = sum(x_)
            x_dot_sq = sum(x_ ** 2)
            n = len(x_)

            alpha_p = n / 2 + self.x_hyper[2]
            beta_p = self.x_hyper[3] + x_dot_sq / 2 + self.x_hyper[1] * self.x_hyper[0] ** 2 / 2
            beta_p -= (x_dot + self.x_hyper[1] * self.x_hyper[0]) ** 2 / (2 * (n + self.x_hyper[1]))

            estim_rho = alpha_p / beta_p
            estim_var = estim_rho * (m_s + 1)
            estim_var = 1 / estim_var

        elif self.x_distr == 'NegativeBinomial':
            n = len(x_)
            x_dot = sum(x_)
            theta_hat = (self.x_hyper[1] + x_dot) / (self.x_hyper[1] + x_dot + self.x_hyper[2] + n * self.x_hyper[0])
            estim_var = self.x_hyper[0] * theta_hat / (1-theta_hat)**2
            estim_var = estim_var / (m_s+1)
        else:
            raise NameError('Need to give correct x_distr')
        estim_var = 2*estim_var

        return -len(xdiff)*np.log(estim_var)/2 - 0.5 * sum(xdiff**2)/estim_var

    def prop_m(self, ratio=1, m_prime=None):

        x_ = self.x[:self.tau_s[1]]

        m_s = np.arange(start=0, stop=100)
        if len(x_) > 1:
            densities = self.jump_estim_llik(x_, m_s) + self.lprior_m(m_s)
            densities = densities*(1/np.log(len(x_)))
        else:
            densities = self.lprior_m(m_s)

        densities = np.exp(densities-max(densities))
        densities = densities/sum(densities)

        if densities[0] < 0.999:
            max_m = max(np.where(np.cumsum(densities) < 0.999)[0])
        else:
            max_m = 0
        if self.m > max_m:
            max_m = self.m
        if m_prime is not None and m_prime > max_m:
            max_m = m_prime

        densities = densities[:max_m+1] * self.get_M(max_m)
        densities = densities/sum(densities)
        if np.isnan(np.sum(densities)):
            print(densities)

        if ratio == 1:
            m_prop = np.random.choice(a=np.arange(max_m + 1), p=densities)
            prob_proposal = np.log(densities[self.m]) - np.log(densities[m_prop])
            return m_prop, prob_proposal
        elif ratio == 0:
            m_prop = np.random.choice(a=np.arange(max_m + 1), p=densities)
            prob_proposal = np.log(densities[m_prop])
            return m_prop, prob_proposal
        elif ratio == 2:
            return np.log(densities[self.m])
        else:
            return np.log(densities[self.m]), np.log(densities[m_prime])

