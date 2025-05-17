import sys,os,math

import numpy as np
import scipy.stats as sstats

import matplotlib.pyplot as plt
import seaborn as sns



##################
# objective function
##################
def ackley(x_nd, a=20, b=0.2, c=6.28318530718):
	if x_nd.ndim<2:
		x_nd = x_nd[None,:]

	y_n = a * np.exp(-b*np.sqrt(np.mean(x_nd**2,axis=1))) + np.exp(np.mean(np.cos(c*x_nd), axis=1)) - a - 2.718281828459

	return y_n

##################
# grid generators
##################
def grid_uniform(space_d):
	grid_d = np.meshgrid(*space_d, indexing='ij')

	for d in range(len(grid_d)):
		# calculate interval
		diff = np.diff(sorted(list(set(grid_d[d].ravel()))))
		if diff.size == 0:
			continue
		interval = np.mean(diff)

		if np.isfinite(interval):
			grid_d[d] += np.random.uniform(high=interval, size=grid_d[d].shape)

	return np.c_[[g.ravel() for g in grid_d]].T

def num_grid_uniform(num_d, low=0, high=1):
	space_d = [np.linspace(low,high,num+1)[:num] for num in num_d]

	return grid_uniform(space_d)

def fixed_grid_uniform(num_d, l_fix, low=0, high=1):
	space_d = [np.linspace(low,high,num+1)[:num] for num in num_d]
	for d,val in l_fix:
		space_d[d] = np.asarray([val])

	return grid_uniform(space_d)


def lhs_uniform(x_d, len_d, l_fix=[], num=64):
	uni01_nd = np.asarray(lhsmdu.sample(x_d.size, num)).T
	grid_X_nd = np.clip(x_d[np.newaxis,:] + len_d[np.newaxis,:] * (uni01_nd - 0.5), a_min=0, a_max=1)

	for d,val in l_fix:
		grid_X_nd[:,d] = val

	return grid_X_nd

####################################
# acquisition for closeness
####################################
def PI_target(m_n, s_n, target, delta):
	normalized_upper_n = (target + delta - m_n) / s_n
	normalized_lower_n = (target - delta - m_n) / s_n

	return sstats.norm.cdf(normalized_upper_n) - sstats.norm.cdf(normalized_lower_n)

def EI_target(m_n, s_n, target, delta):
	upper_n = target + delta - m_n
	lower_n = target - delta - m_n
	center_n = target - m_n

	normalized_upper_n = upper_n / s_n
	normalized_lower_n = lower_n / s_n
	normalized_center_n = center_n / s_n

	ei_n = upper_n * sstats.norm.cdf(normalized_upper_n) + lower_n * sstats.norm.cdf(normalized_lower_n) - 2*center_n * sstats.norm.cdf(normalized_center_n)
	ei_n += s_n * (sstats.norm.pdf(normalized_upper_n) + sstats.norm.pdf(normalized_lower_n) - 2*sstats.norm.pdf(normalized_center_n))

	return ei_n

def chi2_target(m_n, s_n, target, q):
	if isinstance(q, np.ndarray):
		q_n = q
	else:
		q_n = np.ones_like(m_n) * q

	dof_n = np.ones_like(m_n)
	nc_n = ((target - m_n)/s_n)**2 #np.power((target - m_n) / s_n, 2)

	lcb_n = sstats.ncx2.ppf(q_n, dof_n, nc_n)

	return -lcb_n

def chi2_target_dim(m_nk, s_nk, target_k, q):
	if isinstance(q, np.ndarray):
		q_n = q
	else:
		q_n = np.ones(m_nk.shape[0]) * q

	gammasq_n = np.mean(s_nk**2, axis=1)
	lmd_n = np.sum((m_nk - target_k[None,:])**2, axis=1) / gammasq_n

	# # NC -> normal
	K = m_nk.shape[1]
	# r1_n = lmd_n + K
	# r2_n = 2*(K + lmd_n*2)
	# r3_n = 8*(K + lmd_n*3)
	# l_n = 1 - r1_n * r3_n / (3*r2_n**2)

	# approx_m_n = 1 + l_n * (l_n - 1) * (r2_n / (2*r1_n**2) - (2 - l_n)*(1 - 3*l_n) * (r2_n**2) / (8*r1_n**4))
	# approx_s_n = l_n * (r2_n**2) / r1_n * (1 - (1 - l_n) * (1 - 3*l_n) * r2_n / (4 * r1_n**2))

	# z_n = sstats.norm.ppf(q_n, approx_m_n, approx_s_n) # frequent z_n<0
	# t_n = np.power(z_n, -l_n) * (K + lmd_n)
	# d_n = t_n * gammasq_n

	d_n = sstats.ncx2.ppf(q_n * gammasq_n, np.ones_like(q_n)*K, lmd_n) / gammasq_n

	return -d_n

#########################
# data/unit conversion
#########################
def wlen_to_ev(l_n, exponent=9):
	return np.power(10,exponent - 6) * 1.239843288822867 / l_n

def ppm_to_gc(d_n, base_grid=5.8696):
	return base_grid * (1 - d_n*1e-6)

if __name__=='__main__':
	sns.set_context('talk', font_scale=1.1)

	D=3
	x_nd = fixed_grid_uniform(np.asarray((50,)*D), [], low=-.5, high=2)

	y_n = ackley(x_nd, a=3, c=4)

	# print('x_nd[:,x_3==0] = \n', x_nd[np.fabs(x_nd[:,2])<0.25])
	# print('x_nd.shape =', x_nd[np.fabs(x_nd[:,2])<1e-1].shape)
	# print('x_nd =\n', x_nd)
	# print('x_nd.shape =', x_nd.shape)

	# cm = sns.cubehelix_palette(start=.05, gamma=1.2, reverse=True, as_cmap=True)
	# plt.imshow(y_n.reshape((50,50)), origin='lower', extent=[-.5,2,-.5,2], cmap=cm)
	# plt.colorbar()
	plt.hist(y_n, bins=50)
	plt.show()