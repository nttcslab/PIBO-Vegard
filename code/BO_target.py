import sys,os,argparse

import math
import numpy as np
import scipy.stats as sstat
import scipy.linalg as slin
import scipy.special as ssp
import sklearn.gaussian_process as GP
import sklearn.linear_model as Linear

# import torch


class naive_BO:
	def __init__(self, kernel, conf):
		self.init_kernel_ = kernel
		self.kernel_ = kernel
		self.floor_ = conf['floor']

		self.gp_ = None

	def fit(self, X_nd, Y_n):
		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		_Y_n = Y_n.copy()

		if (self.floor_ is None) or (self.floor_ is np.nan):
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				_Y_n[np.isnan(_Y_n)] = np.min(fY_n)
			else:
				_Y_n[np.isnan(_Y_n)] = 0
		else:
			_Y_n[np.isnan(_Y_n)] = self.floor_

		return self._fit(X_nd, _Y_n)

	def _fit(self, X_nd, Y_n):
		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=4, normalize_y=False)

		'''
		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		if (self.floor_ is None) or (self.floor_ is np.nan):
			fY_n = Y_n[np.isfinite(Y_n)]
			if fY_n.size > 0:
				Y_n[np.isnan(Y_n)] = np.min(fY_n)
			else:
				Y_n[np.isnan(Y_n)] = 0
		else:
			Y_n[np.isnan(Y_n)] = self.floor_
		'''

		gp.fit(X_nd, Y_n)

		self.gp_ = gp

		return gp

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		return m_n, s_n
		
	def acquisition(self, X_nd, return_prob=False, return_ms=False):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.predict(X_nd, return_std=True)

		diff_best_n = (m_n - self.gp_.y_train_.max())
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, np.ones_like(EI_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret
		# return m_n + .5 * s_n

	def ucb(self, X_nd, return_prob=False, return_ms=False, scale=.1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.predict(X_nd, return_std=True)
		ucb_n = m_n + scale * s_n

		ret = (ucb_n, np.ones_like(ucb_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return ucb_n
		else:
			return ret


	def acquisition_MES(self, X_nd, beta=0, return_prob=False):
		###########################################
		# subroutine
		###########################################
		def binary_search(func, val, f_n, x_n, thres):
			min_idx = np.argsort(np.fabs(f_n - val))[0]
			if np.abs(f_n[min_idx] - val) < thres:
				return x_n[min_idx]

			if f_n[min_idx] > val:
				lx = x_n[min_idx-1]
				ux = x_n[min_idx]
			else:
				lx = x_n[min_idx]
				ux = x_n[min_idx+1]

			cnt = 1000
			mx = (lx + ux)/2
			f_mid = func(mx)

			while cnt > 0 and np.abs(f_mid - val) > thres:
				if f_mid > val:
					ux = mx
				else:
					lx = mx
				mx = (lx + ux)/2
				f_mid = func(mx)

				cnt -= 1

			return mx
		def MES(mean_n, sig_n, y_m, lmd=1e-3, K=10):
			# sample max
			left = np.max(y_m)
			# sig_n = np.sqrt(var_n)

			prodp_left = np.prod(sstat.norm.cdf((left - mean_n)/sig_n))

			if prodp_left < .25:
				right = np.max(mean_n + 5 * sig_n)
				while np.prod(sstat.norm.cdf((right - mean_n)/sig_n)) < .75:
					right += right - left
				ygrid_m = np.linspace(left, right, 100)

				p_m = np.prod(sstat.norm.cdf((ygrid_m[np.newaxis,:] - mean_n[:,np.newaxis])/sig_n[:,np.newaxis]), axis=0)

				# find quantiles
				prod_cdf = lambda x: np.prod(sstat.norm.cdf(x - mean_n)/sig_n)
				q05 = binary_search(prod_cdf, .5, p_m, ygrid_m, .01)
				q025 = binary_search(prod_cdf, .25, p_m, ygrid_m, .01)
				q075 = binary_search(prod_cdf, .75, p_m, ygrid_m, .01)

				b = (q075 - q025) / (math.log(math.log(4)) - math.log(math.log(4/3)))
				a = q05 + b * math.log(math.log(2))

				ymax_k = a - b*np.log(-np.log(np.random.uniform(size=K)))
				mmax = left + 5 * math.sqrt(lmd)
				ymax_k[ymax_k < mmax] = mmax
			else:
				ymax_k = (left + 5 * math.sqrt(lmd)) * np.ones(K)

			z_nk = (ymax_k[np.newaxis,:] - mean_n[:,np.newaxis])/sig_n[:,np.newaxis]
			cdf_nk = sstat.norm.cdf(z_nk)
			pdf_nk = sstat.norm.pdf(z_nk)

			mes_n = np.mean(.5 * z_nk * pdf_nk / cdf_nk - np.log(cdf_nk), axis=1)

			return mes_n

		#############################################
		# acquisition function
		#############################################
		if self.gp_ is None:
			uni_n = np.random.uniform(size=(X_nd.shape[0]))
			# uni_n = np.ones(X_nd.shape[0])
			return uni_n * prob_n if not return_prob else (uni_n*prob_n, prob_n, )

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		prob_n = np.ones_like(m_n)
		mes_n = MES(m_n, s_n, self.gp_.y_train_, self.gp_.kernel_.k2.noise_level)

		acq_n = mes_n
		return acq_n if not return_prob else (acq_n, prob_n, )

class variable_prior_BO:
	def __init__(self, kernel, conf):
		self.init_kernel_ = kernel
		self.kernel_ = kernel
		self.floor_ = conf['floor']
		self.alpha_ = conf['alpha']
		self.prior_ = 0

		self.gp_ = None

	def fit(self, X_nd, Y_n):
		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		_Y_n = Y_n.copy()

		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.prior_ = np.min(fY_n)
			else:
				self.prior_ = 0
		else:
			self.prior_ = self.floor_

		_Y_n[np.isnan(_Y_n)] = self.prior_

		# this prior setup is focused for maximization?
		self.prior_ = 0
		return self._fit(X_nd, _Y_n - self.prior_)

	def _fit(self, X_nd, Y_n):
		gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=4, normalize_y=False, alpha=self.alpha_)

		'''
		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		if (self.floor_ is None) or (self.floor_ is np.nan):
			fY_n = Y_n[np.isfinite(Y_n)]
			if fY_n.size > 0:
				Y_n[np.isnan(Y_n)] = np.min(fY_n)
			else:
				Y_n[np.isnan(Y_n)] = 0
		else:
			Y_n[np.isnan(Y_n)] = self.floor_
		'''

		gp.fit(X_nd, Y_n)

		self.gp_ = gp

		return gp

	def _trunc_acq(self, m_ni, s_ni, thres=0):
		# m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)

		# raw EI
		diff_best_ni = (m_ni - self.gp_.y_train_.max())
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		# raw_EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		# truncated
		trunc_diff = m_ni - thres
		trunc_z_ni = diff_best_ni / s_ni
		trunc_cdf_ni = sstat.norm.cdf(trunc_z_ni)
		trunc_pdf_ni = sstat.norm.pdf(trunc_z_ni)

		trunc_EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni - trunc_diff*trunc_cdf_ni - s_ni*trunc_pdf_ni)

		return trunc_EI_ni

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_

		return m_n, s_n

	def acquisition(self, X_nd, return_prob=False, return_ms=False, **kwargs):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_

		diff_best_n = (m_n - (self.gp_.y_train_.max() + self.prior_))
		z_n = diff_best_n / s_n

		cdf_n = sstat.norm.cdf(z_n)
		pdf_n = sstat.norm.pdf(z_n)

		EI_n = np.fmax(0, diff_best_n*cdf_n + s_n*pdf_n)

		ret = (EI_n, 
			np.ones_like(EI_n) if return_prob else None, 
			m_n if return_ms else None, s_n if return_ms else None, 
			self.prior_ if 'return_prob' in kwargs and kwargs['return_prob'] else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

		# return (EI_n, np.ones(EI_n.size)) if return_prob else EI_n

	def trunc_acquisition(self, X_nd, return_prob=False, return_ms=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_
		EI_n = self._trunc_acq(m_n, s_n)

		ret = (EI_n, 
			np.ones_like(EI_n) if return_prob else None, 
			m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_n
		else:
			return ret

	def acquisition_MES(self, X_nd, beta=0, return_prob=False):
		###########################################
		# subroutine
		###########################################
		def binary_search(func, val, f_n, x_n, thres):
			min_idx = np.argsort(np.fabs(f_n - val))[0]
			if np.abs(f_n[min_idx] - val) < thres:
				return x_n[min_idx]

			if f_n[min_idx] > val:
				lx = x_n[min_idx-1]
				ux = x_n[min_idx]
			else:
				lx = x_n[min_idx]
				ux = x_n[min_idx+1]

			cnt = 1000
			mx = (lx + ux)/2
			f_mid = func(mx)

			while cnt > 0 and np.abs(f_mid - val) > thres:
				if f_mid > val:
					ux = mx
				else:
					lx = mx
				mx = (lx + ux)/2
				f_mid = func(mx)

				cnt -= 1

			return mx
		def MES(mean_n, sig_n, y_m, lmd=1e-3, K=10):
			# sample max
			left = np.max(y_m)
			# sig_n = np.sqrt(var_n)

			prodp_left = np.prod(sstat.norm.cdf((left - mean_n)/sig_n))

			if prodp_left < .25:
				right = np.max(mean_n + 5 * sig_n)
				while np.prod(sstat.norm.cdf((right - mean_n)/sig_n)) < .75:
					right += right - left
				ygrid_m = np.linspace(left, right, 100)

				p_m = np.prod(sstat.norm.cdf((ygrid_m[np.newaxis,:] - mean_n[:,np.newaxis])/sig_n[:,np.newaxis]), axis=0)

				# find quantiles
				prod_cdf = lambda x: np.prod(sstat.norm.cdf(x - mean_n)/sig_n)
				q05 = binary_search(prod_cdf, .5, p_m, ygrid_m, .01)
				q025 = binary_search(prod_cdf, .25, p_m, ygrid_m, .01)
				q075 = binary_search(prod_cdf, .75, p_m, ygrid_m, .01)

				b = (q075 - q025) / (math.log(math.log(4)) - math.log(math.log(4/3)))
				a = q05 + b * math.log(math.log(2))

				ymax_k = a - b*np.log(-np.log(np.random.uniform(size=K)))
				mmax = left + 5 * math.sqrt(lmd)
				ymax_k[ymax_k < mmax] = mmax
			else:
				ymax_k = (left + 5 * math.sqrt(lmd)) * np.ones(K)

			z_nk = (ymax_k[np.newaxis,:] - mean_n[:,np.newaxis])/sig_n[:,np.newaxis]
			cdf_nk = sstat.norm.cdf(z_nk)
			pdf_nk = sstat.norm.pdf(z_nk)

			mes_n = np.mean(.5 * z_nk * pdf_nk / cdf_nk - np.log(cdf_nk), axis=1)

			return mes_n

		#############################################
		# acquisition function
		#############################################
		if self.gp_ is None:
			uni_n = np.random.uniform(size=(X_nd.shape[0]))
			# uni_n = np.ones(X_nd.shape[0])
			return uni_n * prob_n if not return_prob else (uni_n*prob_n, prob_n, )

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.prior_
		prob_n = np.ones_like(m_n)
		mes_n = MES(m_n, s_n, self.gp_.y_train_ + self.prior_, self.gp_.kernel_.k2.noise_level)

		acq_n = mes_n

		ret = (acq_n, np.ones_like(acq_n) if return_prob else None, m_n if return_ms else None, s_n if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return acq_n
		else:
			return ret


class rnd_mean_prior_BO(variable_prior_BO):
	def __init__(self, kernel, conf):
		super(rnd_mean_prior_BO, self).__init__(kernel, conf)
		self.last_prior_ = None


	def fit(self, X_nd, Y_n, t_num=None):
		if t_num is None or type(t_num) is not tuple:
			val_n = np.isfinite(Y_n)
			num_val = np.sum(val_n)
			num_nan = np.sum(1 - val_n)

			self.beta_ = (num_val if num_val > 0 else .1, num_nan if num_nan > 0 else .1)
		else:
			self.beta_ = t_num[:2]

		# Y_n[np.logical_not(np.isfinite(Y_n))] = self.floor_
		_Y_n = Y_n.copy()

		# floor padding
		if self.floor_ is None:
			fY_n = _Y_n[np.isfinite(_Y_n)]
			if fY_n.size > 0:
				self.floor_ = np.min(fY_n)
			else:
				self.floor_ = 0

		_Y_n[np.isnan(_Y_n)] = self.floor_


		'''
		# online floor padding
		if self.floor_ is None:
			if np.isnan(_Y_n[0]):
				_Y_n[0] = 0
			for n in range(1,Y_n.size):
				_Y_n[n] = Y_n[n] if np.isfinite(Y_n[n]) else np.nanmin(_Y_n[:n])
		else:
			_Y_n[np.isnan(_Y_n)] = self.floor_
		'''

		self.prior_ = 0

		self.gp_ = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=1, normalize_y=False, alpha=self.alpha_)
		self.gp_.fit(X_nd, _Y_n)

		return self.gp_

	def predict(self, X_nd, num_trials=1):
		m_ni = np.zeros((X_nd.shape[0], num_trials))
		# s_n = np.zeros(X_nd.shape[0],)
		s_ni = np.zeros_like(m_ni)

		# u_i = np.random.beta(self.beta_[0], self.beta_[1], size=num_trials)
		u_i = np.random.uniform(size=num_trials)
		ceil = np.nanmax(self.gp_.y_train_)
		floor = np.nanmin(self.gp_.y_train_)

		for i in range(num_trials):
			# prior level
			prior = floor + u_i[i] * (ceil - floor)
			# print('@@@ prior: {:.3g} [{:.3g}, {:.3g}]'.format(prior, floor, ceil))

			# GP fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
			gp.fit(self.gp_.X_train_, self.gp_.y_train_ - prior)

			# if i < 1:
			# 	ms, s_n = gp.predict(X_nd, return_std=True)
			# else:
			# 	ms = gp.predict(X_nd)
			m, s = gp.predict(X_nd, return_std=True)

			m_ni[:,i] = m + prior
			s_ni[:,i] = s

		return np.squeeze(m_ni), np.squeeze(s_ni)

	def _pred_prior(self, X_nd, num_trials=1):
		m_ni = np.zeros((X_nd.shape[0], num_trials))
		# s_n = np.zeros(X_nd.shape[0],)
		s_ni = np.zeros_like(m_ni)

		# u_i = np.random.beta(self.beta_[0], self.beta_[1], size=num_trials)
		u_i = np.random.uniform(size=num_trials)
		ceil = np.nanmax(self.gp_.y_train_)
		prior_i = self.floor_ + u_i * (ceil - self.floor_)

		for i in range(num_trials):
			# prior level
			prior = prior_i[i]

			# GP fitting
			gp = GP.GaussianProcessRegressor(kernel=self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
			gp.fit(self.gp_.X_train_, self.gp_.y_train_ - prior)

			# if i < 1:
			# 	ms, s_n = gp.predict(X_nd, return_std=True)
			# else:
			# 	ms = gp.predict(X_nd)
			m, s = gp.predict(X_nd, return_std=True)

			m_ni[:,i] = m + prior
			s_ni[:,i] = s

		return np.squeeze(m_ni), np.squeeze(s_ni), prior_i, u_i

	def _acq_prior(self, X_nd, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)

		diff_best_ni = (m_ni - self.gp_.y_train_.max())
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		return EI_ni, prior_i, u_i


	def acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		# m_ni, s_ni = self.predict(X_nd, num_trials)
		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)
		self.last_prior_ = np.mean(prior_i)

		diff_best_ni = (m_ni - self.gp_.y_train_.max())
		z_ni = diff_best_ni / s_ni

		cdf_ni = sstat.norm.cdf(z_ni)
		pdf_ni = sstat.norm.pdf(z_ni)

		EI_ni = np.fmax(0, diff_best_ni*cdf_ni + s_ni*pdf_ni)

		ret = (EI_ni, 
			np.ones_like(EI_ni) if return_prob else None, 
			m_ni if return_ms else None, s_ni if return_ms else None, 
			prior_i if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_ni
		else:
			return ret

	def trunc_acquisition(self, X_nd, return_prob=False, return_ms=False, return_prior=False, num_trials=1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni, prior_i, u_i = self._pred_prior(X_nd, num_trials)
		EI_ni = self._trunc_acq(m_ni, s_ni)
		self.last_prior_ = np.mean(prior_i)

		ret = (EI_ni, 
			np.ones_like(EI_ni) if return_prob else None, 
			m_ni if return_ms else None, s_ni if return_ms else None, 
			prior_i if return_prior else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return EI_ni
		else:
			return ret

	def ucb(self, X_nd, return_prob=False, return_ms=False, num_trials=1, scale=.1):
		if self.gp_ is None:
			return np.random.uniform(size=(X_nd.shape[0]))

		m_ni, s_ni = self.predict(X_nd, num_trials)
		ucb_ni = m_ni + s_ni * scale

		ret = (ucb_ni, np.ones_like(ucb_ni) if return_prob else None, m_ni if return_ms else None, s_ni if return_ms else None)
		ret = tuple([elem for elem in ret if elem is not None])

		if len(ret)==1:
			return ucb_ni
		else:
			return ret


class BO_linear_priormean(naive_BO):
	def __init__(self, kernel, conf, lprior=None):
		super(BO_linear_priormean,self).__init__(kernel, conf)
		self.ridge_lmd_ = conf['ridge']

		self.lprior_ = lprior

	def fit(self, X_nd, Y_n, prior_X_nd=None, prior_Y_n=None):
		# first renew linear regression model
		if prior_X_nd is not None and prior_Y_n is not None:
			lprior = Linear.Ridge(alpha=self.ridge_lmd_)
			lprior.fit(prior_X_nd, prior_Y_n)

			self.lprior_ = lprior

		if self.lprior_ is None:
			my_n = np.zeros(X_nd.shape[0])
		else:
			my_n = self.lprior_.predict(X_nd)
		self.gp_ = super(BO_linear_priormean,self).fit(X_nd, Y_n - my_n)

	def predict(self, X_nd):
		if self.lprior_ is None:
			my_n = np.zeros(X_nd.shape[0])
		else:
			my_n = self.lprior_.predict(X_nd)

		m_n, s_n = self.gp_.predict(X_nd, return_std=True)

		return m_n + my_n, s_n

class BO_monotonic_priormean_1d(naive_BO):
	def __init__(self, kernel, conf,):
		super(BO_monotonic_priormean_1d, self).__init__(kernel, conf)
		self.alpha_ = conf['alpha']
		self.param = {'a':0., 'b':0.}

	def fit(self, X_n, Y_n, ab_iter=10, gp_iter=3):
		# renew linear
		a = self.param['a']
		b = self.param['b']
		# initialize a, b
		for abit in range(ab_iter):
			# a part
			x2 = np.sum(X_n * X_n)
			if x2 < 1e-6:
				a = 0
			else:
				a = np.sum(X_n * (Y_n - b)) / np.sum(X_n * X_n)

			if a < 0:
				a = 0
			# b part
			b = np.sum(Y_n - a*X_n) / Y_n.size

		gp = GP.GaussianProcessRegressor(self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
		for gpit in range(gp_iter):
			gp.fit(X_n[:,None], Y_n - a * X_n - b)
			K_nn = gp.kernel_(X_n[:,None])

			Kinvx = np.linalg.solve(K_nn, X_n)
			Kinv1 = np.linalg.solve(K_nn, np.ones(X_n.size))
			Kxx = np.sum(Kinvx * X_n)
			K11 = np.sum(Kinv1)

			for abit in range(ab_iter):
				# a part
				a_nume = np.sum(Kinvx * (Y_n - b))
				a = 0 if a_nume < 0 else a_nume / Kxx
				# b part
				b = np.sum(Kinv1 * (Y_n - a*X_n)) / K11

		self.param['a'] = a
		self.param['b'] = b
		self.gp_ = gp
		return self.gp_

	def predict(self, X_n):
		m_n, s_n = self.gp_.predict(X_n[:,None], return_std=True)
		m_n += self.param['a'] * X_n + self.param['b']

		return m_n, s_n

class BO_monotonic_priormean_multi(naive_BO):
	def __init__(self, kernel, conf, dim):
		super(BO_monotonic_priormean_multi, self).__init__(kernel, conf)
		self.alpha_ = conf['alpha']
		self.dim_ = dim #conf['dim']
		self.param = {'a':0, 'b':0}

	def fit(self, X_nd, Y_n, ab_iter=10, gp_iter=3):
		# renew linear
		a = self.param['a']
		b = self.param['b']

		X_n = X_nd[:,self.dim_]

		for abit in range(ab_iter):
			# a part
			x2 = np.sum(X_n * X_n)
			if x2 < 1e-6:
				a = 0
			else:
				a = np.sum(X_n * (Y_n - b)) / np.sum(X_n * X_n)

			if a < 0:
				a = 0
			# b part
			b = np.sum(Y_n - a*X_n) / Y_n.size

		gp = GP.GaussianProcessRegressor(self.init_kernel_, n_restarts_optimizer=5, normalize_y=False, alpha=self.alpha_)
		for gpit in range(gp_iter):
			gp.fit(X_nd, Y_n - a * X_n - b)
			K_nn = gp.kernel_(X_nd)

			Kinvx = np.linalg.solve(K_nn, X_n)
			Kinv1 = np.linalg.solve(K_nn, np.ones(X_n.size))
			Kxx = np.sum(Kinvx * X_n)
			K11 = np.sum(Kinv1)

			for abit in range(ab_iter):
				# a part
				a_nume = np.sum(Kinvx * (Y_n - b))
				a = 0 if a_nume < 0 else a_nume / Kxx
				# b part
				b = np.sum(Kinv1 * (Y_n - a*X_n)) / K11

		# randomize b and refit
		diff_n = Y_n - a*X_n
		b_min = diff_n.min()
		b_max = diff_n.max()
		# print('b_min, b_max = {}, {}'.format(b_min, b_max))
		b = np.random.uniform() * (b_max - b_min) + b_min
		gp.fit(X_nd, Y_n - a*X_n - b)

		self.param['a'] = a
		self.param['b'] = b
		self.gp_ = gp

		return self.gp_

	def predict(self, X_nd):
		m_n, s_n = self.gp_.predict(X_nd, return_std=True)
		m_n += self.param['a'] * X_nd[:,self.dim_] + self.param['b']

		return m_n, s_n
