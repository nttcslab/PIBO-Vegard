import sys,os,math

import numpy as np
import scipy.optimize
import sklearn

import BO_target

####################################
# normalizers
# x: [0,1]
# raw: values used in actual experiments
####################################
def uni_to_raw(x_d, bb):
	return bb[0,:] + x_d * (bb[1,:] - bb[0,:])

def raw_to_uni(raw_d, bb):
	return (raw_d - bb[0,:]) / (bb[1,:] - bb[0,:])

def all_u2r(x_nd, bb):
	return np.asarray([uni_to_raw(x_nd[n,:], bb) for n in range(x_nd.shape[0])])

def all_r2u(raw_config_nd, bb):
	return np.asarray([raw_to_uni(raw_config_nd[n,:], bb) for n in range(raw_config_nd.shape[0])])

BASE_GRID=5.8696
#########################
# data/unit conversion
#########################
def wlen_to_ev(l_n, exponent=9):
	return np.power(10,exponent - 6) * 1.239843288822867 / l_n

def ev_to_wlen(ev_n, exponent=9):
	return np.power(10,exponent - 6) * 1.239843288822867 / ev_n

def ppm_to_gc(d_n, base_grid=5.8696):
	return base_grid * (1 - d_n*1e-6)

def gc_to_ppm(gc_n, base_grid=5.8696):
	return 1e6 * (1 - gc_n/base_grid)

def Ga_raw_to_norm(ga_n):
	return ga_n / (ga_n + 120.)

def Ga_norm_to_raw(r_n):
	return 120 * r_n / (1 - r_n)

######################################
# device input to composition proportion
######################################
def uv(w0, x_nd):
	u_n = w0[0] * x_nd[:,0] + w0[1] #Ga_raw_to_norm(x_nd[:,0]) + w0[1]
	v_n = w0[2] * x_nd[:,1] + w0[3]

	return u_n, v_n

def dev_gc(w0, x_nd):
	u_n, v_n = uv(w0, x_nd)

	return BASE_GRID + 0.1894*v_n - 0.4184*u_n + 0.013*u_n*v_n

def dev_ev(w0, x_nd):
	u_n, v_n = uv(w0, x_nd)

	return 1.35 + 0.672*u_n - 1.091*v_n + 0.758*(u_n**2) + 0.101*(v_n**2) - 0.157*u_n*v_n - 0.312*(u_n**2)*v_n + 0.109*u_n*(v_n**2)

def delta_gc(u_n, v_n, gc_n):
	return BASE_GRID + 0.1894*v_n - 0.4184*u_n + 0.013*u_n*v_n - gc_n

def delta_ev(u_n, v_n, ev_n):
	return 1.35 + 0.672*u_n - 1.091*v_n + 0.758*(u_n**2) + 0.101*(v_n**2) - 0.157*u_n*v_n - 0.312*(u_n**2)*v_n + 0.109*u_n*(v_n**2) - ev_n

def squared_loss_gc(w0, x_nd, y_n, ridge=1e-2):
	u_n, v_n = uv(w0, x_nd)

	delta_n = delta_gc(u_n, v_n, y_n)
	return (delta_n**2).mean()/2 + np.sum(w0**2)*ridge / 2

def grad_sq_gc(w0, x_nd, y_n, ridge=1e-2):
	u_n, v_n = uv(w0, x_nd)

	delta_n = delta_gc(u_n, v_n, y_n)

	g1_n = delta_n * (0.013 * v_n - 0.4184)
	g0_n = g1_n * x_nd[:,0] #Ga_raw_to_norm(x_nd[:,0])
	g3_n = delta_n * (0.1894 - 0.013*u_n)
	g2_n = g3_n * x_nd[:,1]

	return np.mean(np.c_[g0_n, g1_n, g2_n, g3_n], axis=0) + ridge*w0

def loss_grad_gc(w0, x_nd, y_n, ridge=1e-2):
	u_n, v_n = uv(w0, x_nd)

	delta_n = delta_gc(u_n, v_n, y_n)

	loss = (delta_n**2).mean()/2 + np.sum(w0**2)*ridge / 2
	g1_n = delta_n * (0.013 * v_n - 0.4184)
	g0_n = g1_n * x_nd[:,0] #Ga_raw_to_norm(x_nd[:,0])
	g3_n = delta_n * (0.1894 - 0.013*u_n)
	g2_n = g3_n * x_nd[:,1]

	grad = np.mean(np.c_[g0_n, g1_n, g2_n, g3_n], axis=0) + ridge*w0

	return (loss, grad)

def squared_loss_ev(w0, x_nd, y_n, ridge=1e-2):
	u_n, v_n = uv(w0, x_nd)

	delta_n = delta_ev(u_n, v_n, y_n)
	return (delta_n**2).mean()/2 + np.sum(w0**2)*ridge / 2

def loss_grad_ev(w0, x_nd, y_n, ridge=1e-2):
	u_n, v_n = uv(w0, x_nd)

	delta_n = delta_ev(u_n, v_n, y_n)
	loss = (delta_n**2).mean()/2 + np.sum(w0**2)*ridge / 2

	g1_n = delta_n * (0.672 + 1.516*u_n - 0.157*v_n - 0.624*u_n*v_n + 0.109*(v_n**2))
	g0_n = g1_n * x_nd[:,0] #Ga_raw_to_norm(x_nd[:,0])
	g3_n = delta_n * (-1.091 + 0.202*v_n - 0.157*u_n - 0.312*(u_n**2) + 0.218*u_n*v_n)
	g2_n = g3_n * x_nd[:,1]

	grad = np.mean(np.c_[g0_n, g1_n, g2_n, g3_n], axis=0) + ridge*w0

	return (loss, grad)

# def loss_grad_ev_gc(w0, x_nd, y_n, ridge=1e-2, ratio=.5):
# 	u_n, v_n = uv(w0, x_nd)

# 	ev_delta_n = delta_ev
# 	gc_delta_n = delta_gc

def grad_sq_ev(w0, x_nd, y_n, ridge=1e-2):
	u_n, v_n = uv(w0, x_nd)

	delta_n = delta_ev(u_n, v_n, y_n)

	g1_n = delta_n * (0.672 + 1.516*u_n - 0.157*v_n - 0.624*u_n*v_n + 0.109*(v_n**2))
	g0_n = g1_n * x_nd[:,0] #Ga_raw_to_norm(x_nd[:,0])
	g3_n = delta_n * (-1.091 + 0.202*v_n - 0.157*u_n - 0.312*(u_n**2) + 0.218*u_n*v_n)
	g2_n = g3_n * x_nd[:,1]

	return np.mean(np.c_[g0_n, g1_n, g2_n, g3_n], axis=0) + ridge*w0

# object
class VegardInputConvertEV:
	def __init__(self,ridge=1e-2):
		self.w0 = np.asarray([1, 0.1, 0.1, 0.1])
		self.ridge = ridge

	def fit(self, x_nd, y_n):
		w0 = self.w0.copy()
		# bounds = scipy.optimize.Bounds(np.asarray([0, -np.inf, 0, -np.inf]), np.asarray([np.inf, np.inf, np.inf, np.inf]))
		bounds = scipy.optimize.Bounds(np.asarray([0.01, 0, 0.01, 0]), np.asarray([np.inf, 1, np.inf, 1]))

		# def show(w0):
		# 	loss, grad = loss_grad_ev(w0,x_nd,y_n,self.ridge)
		# 	print('loss={}, grad='.format(loss),grad)
		res = scipy.optimize.minimize(loss_grad_ev, w0, args=(x_nd, y_n, self.ridge), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxls':50,'maxcor':8, 'ftol':1e-12, 'gtol':1e-12})
		# res = scipy.optimize.minimize(squared_loss_ev, w0, args=(x_nd, y_n, self.ridge), method='L-BFGS-B', bounds=bounds)
		self.w0 = res.x

		return res

	def predict(self,x_nd):
		return dev_ev(self.w0, x_nd)

class VegardInputConvertGC:
	def __init__(self,ridge=1e-2):
		self.w0 = np.asarray([0.01, 0.1, 0.1, 0.1])
		self.ridge = ridge

	def fit(self, x_nd, y_n):
		w0 = self.w0.copy()
		# bounds = scipy.optimize.Bounds(np.asarray([0, -np.inf, 0, -np.inf]), np.asarray([np.inf, np.inf, np.inf, np.inf]))
		bounds = scipy.optimize.Bounds(np.asarray([1e-3, 0, 1e-3, 0]), np.asarray([np.inf, 1, np.inf, 1]))

		# def show(w0):
		# 	loss, grad = loss_grad_ev(w0,x_nd,y_n,self.ridge)
		# 	print('loss={}, grad='.format(loss),grad)
		res = scipy.optimize.minimize(loss_grad_gc, w0, args=(x_nd, y_n, self.ridge), jac=True, method='L-BFGS-B', bounds=bounds, options={'maxls':50,'maxcor':8, 'ftol':1e-12, 'gtol':1e-12})
		self.w0 = res.x

		return res

	def predict(self,x_nd):
		return dev_gc(self.w0, x_nd)



#################################
# BO object
#################################
class BO_vegard_priorEV(BO_target.naive_BO):
	def __init__(self, kernel, conf, vegard=None):
		super(BO_vegard_priorEV, self).__init__(kernel, conf)
		
		self.ridge_lmd_ = conf['ridge']
		self.vegard_ = vegard
		self.xbb_ = conf['xbb']
		self.ybb_ = conf['ybb']

	def fit(self, X_nd, Y_n, prior_X_nd=None, prior_Y_n=None):
		# first renew linear regression model
		if (prior_X_nd is not None) and (prior_Y_n is not None):
			vegard = VegardInputConvertEV(self.ridge_lmd_)
			vegard.fit(prior_X_nd, prior_Y_n)

			self.vegard_ = vegard

		if self.vegard_ is None:
			my_n = np.zeros(X_nd.shape[0])
		else:
			my_n = self.vegard_.predict(X_nd)

		# normalization
		ux_nd = all_r2u(X_nd, self.xbb_)
		uy_n = raw_to_uni(Y_n - my_n, self.ybb_)
		self.gp_ = super(BO_vegard_priorEV,self).fit(ux_nd, uy_n)

	def predict(self, X_nd):
		if self.vegard_ is None:
			my_n = np.zeros(X_nd.shape[0])
		else:
			my_n = self.vegard_.predict(X_nd)

		um_n, us_n = self.gp_.predict(all_r2u(X_nd, self.xbb_), return_std=True)

		return uni_to_raw(um_n, self.ybb_) + my_n, us_n * np.fabs(np.diff(self.ybb_.ravel())).mean()

class BO_vegard_priorGC(BO_target.naive_BO):
	def __init__(self, kernel, conf, vegard=None):
		super(BO_vegard_priorGC, self).__init__(kernel, conf)

		self.ridge_lmd_ = conf['ridge']
		self.vegard_ = vegard
		self.xbb_ = conf['xbb']
		self.ybb_ = conf['ybb']

	def fit(self, X_nd, Y_n, prior_X_nd=None, prior_Y_n=None):
		# renew linear regression
		if (prior_X_nd is not None) and (prior_Y_n is not None):
			vegard = VegardInputConvertGC(self.ridge_lmd_)
			vegard.fit(prior_X_nd, prior_Y_n)

			self.vegard_ = vegard
		if self.vegard_ is None:
			my_n = np.zeros(X_nd.shape[0])
		else:
			my_n = self.vegard_.predict(X_nd)

		ux_nd = all_r2u(X_nd, self.xbb_)
		uy_n = raw_to_uni(Y_n - my_n, self.ybb_)

		self.gp_ = super(BO_vegard_priorGC,self).fit(ux_nd, uy_n)

	def predict(self, X_nd):
		if self.vegard_ is None:
			my_n = np.zeros(X_nd.shape[0])
		else:
			my_n = self.vegard_.predict(X_nd)

		um_n, us_n = self.gp_.predict(all_r2u(X_nd, self.xbb_), return_std=True)

		return uni_to_raw(um_n, self.ybb_) + my_n, us_n * np.fabs(np.diff(self.ybb_.ravel())).mean()

class BO_vegard_prior_indp(BO_target.naive_BO):
	def __init__(self, kernel, conf, ev_vegard=None, gc_vegard=None):
		if type(kernel)==list:
			super(BO_vegard_prior_indp, self).__init__(kernel[0], conf)
			self.ev_ = BO_vegard_priorEV(kernel[0], conf, ev_vegard)
			self.gc_ = BO_vegard_priorGC(kernel[1], conf, gc_vegard)
		else:
			super(BO_vegard_prior_indp, self).__init__(kernel, conf)

			self.ev_ = BO_vegard_priorEV(kernel, conf, ev_vegard)
			self.gc_ = BO_vegard_priorGC(kernel, conf, gc_vegard)

	def fit(self, X_nd, Y_nk, prior_X_nd=None, prior_Y_nk=None):
		if prior_Y_nk is None:
			self.ev_.fit(X_nd, Y_nk[:,0])
			self.gc_.fit(X_nd, Y_nk[:,1])
		else:
			self.ev_.fit(X_nd, Y_nk[:,0], prior_X_nd, prior_Y_nk[:,0])
			self.gc_.fit(X_nd, Y_nk[:,1], prior_X_nd, prior_Y_nk[:,1])

		return self

	def predict(self, X_nd):
		evm_n, evs_n = self.ev_.predict(X_nd)
		gcm_n, gcs_n = self.gc_.predict(X_nd)
		return (np.c_[evm_n, gcm_n], np.c_[evs_n, gcs_n])


class BO_uv_indp(BO_target.naive_BO):
	def __init__(self, kernel, conf, ):
		if type(kernel)==list:
			super(BO_uv_indp, self).__init__(kernel[0], conf)
			self.gp_u_ = BO_target.rnd_mean_prior_BO(kernel[0], conf)
			self.gp_v_ = BO_target.rnd_mean_prior_BO(kernel[1], conf)
		else:
			super(BO_uv_indp, self).__init__(kernel, conf)
			self.gp_u_ = BO_target.rnd_mean_prior_BO(kernel, conf)
			self.gp_v_ = BO_target.rnd_mean_prior_BO(kernel, conf)

	def fit(self, X_nd, UV_nk, ):
		self.gp_u_.fit(X_nd, UV_nk[:,0])
		self.gp_v_.fit(X_nd, UV_nk[:,1])

		return self

	def predict(self, X_nd):
		um_n, us_n = self.gp_u_.predict(X_nd)
		vm_n, vs_n = self.gp_v_.predict(X_nd)

		return (np.c_[um_n, vm_n], np.c_[us_n, vs_n])

class BO_uv_monotonic_1d():
	def __init__(self, kernel, conf, dim=2):
		self.l_gp = [BO_target.BO_monotonic_priormean_1d(kernel, conf) for d in range(dim)]
		self.kernel_ = kernel
		self.conf = conf

		if 'ab_iter' not in self.conf:
			self.conf['ab_iter'] = 10
		if 'gp_iter' not in self.conf:
			self.conf['gp_iter'] = 3

	def fit(self, X_nd, UV_nk):
		# assert(d==k)
		dim = len(self.l_gp)
		for d in range(dim):
			self.l_gp[d].fit(X_nd[:,d], UV_nk[:,d], self.conf['ab_iter'], self.conf['gp_iter'])

		return self

	def predict(self, X_nd):
		dim = len(self.l_gp)
		m_nd = np.zeros_like(X_nd)
		s_nd = np.zeros_like(X_nd)

		for d in range(dim):
			m_n, s_n = self.l_gp[d].predict(X_nd[:,d])
			m_nd[:,d] = m_n
			s_nd[:,d] = s_n

		return m_nd, s_nd

class BO_uv_monotonic():
	def __init__(self, kernel, conf, dim=2):
		self.l_gp = [BO_target.BO_monotonic_priormean_multi(kernel, conf, d) for d in range(dim)]
		self.kernel_ = kernel
		self.conf = conf

		if 'ab_iter' not in self.conf:
			self.conf['ab_iter'] = 10
		if 'gp_iter' not in self.conf:
			self.conf['gp_iter'] = 3

	def fit(self, X_nd, UV_nk):
		# assert(d==k)
		dim = len(self.l_gp)
		for d in range(dim):
			self.l_gp[d].fit(X_nd, UV_nk[:,d], self.conf['ab_iter'], self.conf['gp_iter'])

		return self

	def predict(self, X_nd):
		dim = len(self.l_gp)
		m_nd = np.zeros_like(X_nd)
		s_nd = np.zeros_like(X_nd)

		for d in range(dim):
			m_n, s_n = self.l_gp[d].predict(X_nd)
			m_nd[:,d] = m_n
			s_nd[:,d] = s_n

		return m_nd, s_nd


if __name__=='__main__':
	# x = np.random.uniform(size=10)*5
	x = np.asarray([5,8.])
	print('convert x=\n', Ga_raw_to_norm(x))

	print('x=\n', x)
	print('dec(enc(x))=\n',Ga_norm_to_raw(Ga_raw_to_norm(x)))