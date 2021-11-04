
import numpy as np
import pandas as pd

from model_sampler.segment import Segment

np.random.seed(88)

m_s = np.arange(0, 30)
num_sim = 50

for N in [200, 400, 800]:
    for theta in [0.4, 0.8]:

        results_ind = np.zeros((len(m_s), len(m_s)))
        results_length = np.zeros((len(m_s), len(m_s)))

        for _ in range(num_sim):
            for m in m_s:
                print(m)
                seg = Segment(x_distr='NegativeBinomial', tau_s=[0, N], m=m, x_hyper=[300, 1, 1])
                seg.sample_x(theta=[theta])

                for d in m_s:
                    seg.m = d
                    seg.get_y_bounds()
                    D = seg.y_bounds[-1] - sum(seg.y_bounds[:-1]) if d > 0 else 1

                    results_length[d, m] += max(float(D)/num_sim, 0)
                    results_ind[d, m] += float(D >= 0)/num_sim

        pd.DataFrame(results_ind).to_csv('simulations/results/pspace_ind_N'+str(N)+'_theta'+str(int(theta*10))+'.csv', index=False)

        pd.DataFrame(results_length).to_csv('simulations/results/pspace_len_N'+str(N)+'_theta'+str(int(theta*10))+'.csv', index=False)


