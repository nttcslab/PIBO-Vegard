import sys,os,argparse,time
import tqdm

import math
import numpy as np
import sklearn.gaussian_process as GP
import sklearn.linear_model as Linear

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

# import torch

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import lhsmdu # latin hypercube sampling package
import BO_target, util, vegard


###################################
# acquisition utility
###################################
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
# [0,1] -- raw config value convertor
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

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	'''
	Function to offset the "center" of a colormap. Useful for
	data with a negative min and positive max and you want the
	middle of the colormap's dynamic range to be at zero.

	Input
	-----
	  cmap : The matplotlib colormap to be altered
	  start : Offset from lowest point in the colormap's range.
		  Defaults to 0.0 (no lower offset). Should be between
		  0.0 and `midpoint`.
	  midpoint : The new center of the colormap. Defaults to 
		  0.5 (no shift). Should be between 0.0 and 1.0. In
		  general, this should be  1 - vmax / (vmax + abs(vmin))
		  For example if your data range from -15.0 to +5.0 and
		  you want the center of the colormap at 0.0, `midpoint`
		  should be set to  1 - 5/(5 + 15)) or 0.75
	  stop : Offset from highest point in the colormap's range.
		  Defaults to 1.0 (no upper offset). Should be between
		  `midpoint` and 1.0.
	'''
	cdict = {
		'red': [],
		'green': [],
		'blue': [],
		'alpha': []
	}

	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)

	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False), 
		np.linspace(midpoint, 1.0, 129, endpoint=True)
	])

	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)

		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))

	newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	# plt.register_cmap(cmap=newcmap)
	matplotlib.colormaps.register(newcmap)

	return newcmap

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='BO')
	parser.add_argument('--file', default='../data/data_1st.csv', help='observations in .csv format')
	parser.add_argument('--xrange', default='2:8,5:13', help='range (min and max values) of each dimension. Dimensions should match the csv. Example: "--xrange 0:2,n3:3,0:1" for three dimensions.')
	parser.add_argument('--yrange', help='range for y value. Example: "--yrange -3:3"')
	parser.add_argument('--alpha', type=float, default=1e-3, help='alpha for GP regression model (observation variance)')
	parser.add_argument('--seed', type=int, help='seed for pseudo random number generator.')
	# search space
	parser.add_argument('--fix', help='fixes the value of some dimensions during the search. Example: "--fix *,650.0,*" means the 2nd dim is fixed to 650.0 while the other two dimensions are searched.')
	parser.add_argument('--ngrid', default=200, type=int, help='number of grid cells in each dimension. Two-step search is disabled. --fix option is recommended to reduce the search space. Example: "--ngrid 100"')
	# visualization & file
	parser.add_argument('--noshow', action='store_false', help='plot the acquisition function in either one or two dimensions. Needs combined with --fix and --ngrid options to reduce the search space.')
	parser.add_argument('--srange', type=float, default=0, help='normalized distance from --fix plane for visualizing observations. Value assumed in [0, 1]. Example: "--fix *,650,* --show --srange 0.1" shows observations between *,640,* and *,660,* when --xrange spans [600,700].')
	# target value & prior mean
	parser.add_argument('--target', default='0.1953,0.4247', type=str)
	# parser.add_argument('--nprior', type=int, default=1)
	# parser.add_argument('--ridge', type=float, default=1e-1)

	args = parser.parse_args()
	if args.seed is None:
		np.random.seed()
	else:
		np.random.seed(args.seed)

	# read data
	all_csv = np.loadtxt(args.file, delimiter=',')
	if all_csv.ndim < 2:
		all_csv = all_csv[np.newaxis,:]
	# y_n = all_csv[:,0]
	UV_nk = all_csv[:,:2]
	raw_config_nd = all_csv[:,2:]
	D = raw_config_nd.shape[1]


	# parse bounding boxes
	xbb = np.zeros((2,D)); xbb[1,:]=1 # range is [0,1] by default
	if args.xrange is not None:
		for d,lowhigh in enumerate(args.xrange.replace('n','-').split(',')):
			l, h = [float(s) for s in lowhigh.split(':')]
			xbb[0,d] = l
			xbb[1,d] = h

	ybb = np.asarray([[0],[1]]) # shape=(2,1)
	if args.yrange is not None:
		ybb = np.asarray([[float(v)] for v in args.yrange.split(':')][:2])

	# parse search space
	if args.fix is None:
		l_fix = []
	else:
		# use normalized value
		l_fix = [(d, (float(val) - xbb[0,d])/(xbb[1,d] - xbb[0,d])) for d,val in enumerate(args.fix.split(',')) if not val.find('*')>=0]

	args.show = args.noshow

	# normalized values
	X_nd = all_r2u(raw_config_nd, xbb)
	targetUV_k = np.asarray([float(val) for val in args.target.split(',')])

	# BO setup
	base_kernel = GP.kernels.Product(GP.kernels.ConstantKernel(constant_value=1., constant_value_bounds=(1e-2,1e2)), GP.kernels.Matern(length_scale=0.2, length_scale_bounds=(.05, .5),nu=2.5))
	# base_kernel = GP.kernels.Product(GP.kernels.ConstantKernel(constant_value=1., constant_value_bounds=(1e-2,1e2)), GP.kernels.RBF(length_scale=0.2, length_scale_bounds=(.05, .5)))
	l_kernel = [GP.kernels.Sum(base_kernel, GP.kernels.WhiteKernel(1e-5, noise_level_bounds=(1e-5,1e-4))), 
		GP.kernels.Sum(base_kernel, GP.kernels.WhiteKernel(1e-5, noise_level_bounds=(1e-5,1e-4)))]

	simplefilter('ignore', category=ConvergenceWarning)

	bo = vegard.BO_uv_monotonic(l_kernel[0], {'floor':None, 'alpha':args.alpha, 'ridge':0, 'xbb':xbb, 'ybb':ybb, 'ab_iter':10, 'gp_iter':3})
	bo.fit(X_nd, UV_nk, )
	# print('a =', np.asarray([gp.param['a'] for gp in bo.l_gp]))
	# print('b =', np.asarray([gp.param['b'] for gp in bo.l_gp]))


	target_acq_func = util.PI_target
	percentile_acq = 0.2

	# closest data point by Mahalanobis distance
	dist_n = np.sum(np.power(UV_nk - targetUV_k[None,:], 2), axis=1)
	min_idx = np.argmin(dist_n)
	delta_k = np.fabs(UV_nk[min_idx,:] - targetUV_k)

	if args.ngrid is None:
		num_results = (3, 5) # [0] grids, [1] from each
		# for visualization purpose
		progress_bar = tqdm.tqdm(total=num_results[0]+1, leave=False)
		# two-step acquisition
		# grid_X_nd = num_grid_uniform(np.asarray([10]*D)) # split each dimension by 10 regions
		grid_X_nd = util.fixed_grid_uniform(np.asarray([10]*D), l_fix)
		m_nk, s_nk = bo.predict(grid_X_nd)

		# acq_n0 = target_acq_func(m_nk[:,0], s_nk[:,0], targetUV_k[0], delta_k[0])
		# acq_n1 = target_acq_func(m_nk[:,1], s_nk[:,1], targetUV_k[1], delta_k[1])
		# acq_n = acq_n0 * acq_n1
		acq_n = util.chi2_target_dim(m_nk, s_nk, targetUV_k, percentile_acq)

		# show refined results from the three top grids
		all_X = np.zeros((num_results)+(D,))
		all_acq = np.zeros(num_results)
		all_mean = np.zeros(num_results+(2,))
		all_std = np.zeros(num_results+(2,))
		progress_bar.update(1)
		for i, idx in enumerate(np.argsort(acq_n)[::-1]):
			if i>=num_results[0]:
				break
			lhs_X_nd = lhs_uniform(grid_X_nd[idx,:], np.asarray([.1]*D), l_fix, num=64) # Latin hypercube sampling, [.1]*D means each grid has .1 width in each dimension
			m_nk, s_nk = bo.predict(lhs_X_nd)

			# acq_n0 = target_acq_func(m_nk[:,0], s_nk[:,0], targetUV_k[0], delta_k[0])
			# acq_n1 = target_acq_func(m_nk[:,1], s_nk[:,1], targetUV_k[1], delta_k[1])
			# lhs_acq_n = acq_n0 * acq_n1
			lhs_acq_n = util.chi2_target_dim(m_nk, s_nk, targetUV_k, percentile_acq)

			lhs_idx_n = np.argsort(lhs_acq_n)[::-1] # in descending order (larger comes first)

			_top = lhs_idx_n[:num_results[1]]
			all_X[i,:,:] = uni_to_raw(lhs_X_nd[_top,:], xbb)
			all_acq[i,:] = lhs_acq_n[_top]
			all_mean[i,:,:] = m_nk[_top,:] * (ybb[1,0] - ybb[0,0]) + ybb[0,0]
			all_std[i,:,:] = s_nk[_top,:] * (ybb[1,0] - ybb[0,0])
			progress_bar.update(1)
			# show five results
			# print('[{}] from grid centered at'.format(i, ), uni_to_raw(grid_X_nd[idx,:], xbb))
			# print('acq\tconfig')
			# for j in range(num_results[1]):
			# 	print('{:1.3f}\t'.format(lhs_acq_n[lhs_idx_n[j]] * (ybb[1,0] - ybb[0,0])), uni_to_raw(lhs_X_nd[lhs_idx_n[j],:], xbb))
			# print('') # line break
		# show best result
		_top = np.argsort(all_acq.ravel())[-1]
		top_acq = all_acq.ravel()[_top]
		top_m_k = all_mean.reshape((-1,2))[_top,:]
		top_s_k = all_std.reshape((-1,2))[_top,:]
		top_x = all_X.reshape(np.prod(num_results), -1)[_top,:]

		strx = '({})'.format(', '.join(['{:.4g}'.format(v) for v in top_x]))
		# print_template = '{:.4g}\t\t{:.4g},{:.4g}\t{}'
		# print('acq\tmean,std\tinput value')
		# print(print_template.format(top_acq, top_m, top_s, strx))
		print('')
		print('---- next point ----')
		print('acq'.ljust(11) + 'mean\u00b1std'.ljust(16), 'input value')
		print('{:.4g}'.format(top_acq).ljust(11) + '{:.4g}\u00b1{:.4g}'.format(top_m_k[0], top_s_k[0]).ljust(16)  + '{:.4g}\u00b1{:.4g}'.format(top_m_k[1], top_s_k[1]).ljust(16) + strx)

	else:
		# one-shot acquisition
		grid_X_nd = fixed_grid_uniform(np.asarray((args.ngrid,)*D), l_fix)
		grid_rawX_nd = all_u2r(grid_X_nd, xbb)
		m_nk, s_nk = bo.predict(grid_X_nd)
		# acq_n0 = target_acq_func(m_nk[:,0], s_nk[:,0], targetUV_k[0], delta_k[0])
		# acq_n1 = target_acq_func(m_nk[:,1], s_nk[:,1], targetUV_k[1], delta_k[1])
		# acq_n = acq_n0 * acq_n1
		acq_n = util.chi2_target_dim(m_nk, s_nk, targetUV_k, percentile_acq)

		idx_n = np.argsort(acq_n)[::-1]
		print('') # line break

		# show results
		print('---- grid search results ----')
		print('acq'.ljust(11) + 'mean\u00b1std'.ljust(32), 'input value')
		# show 10 results
		for j in range(10):
			# print('{:1.3f}\t'.format(acq_n[idx_n[j]] * (ybb[1,0] - ybb[0,0])), uni_to_raw(grid_X_nd[idx_n[j],:], xbb))
			txt = '{:.4g}'.format(acq_n[idx_n[j]]).ljust(11)
			# txt += '{:.4g}\u00b1{:.4g}'.format(m_n[idx_n[j]] * (ybb[1,0] - ybb[0,0]) + ybb[0,0], s_n[idx_n[j]] * (ybb[1,0] - ybb[0,0])).ljust(16)
			txt += '{:.3g}\u00b1{:.3g}'.format(m_nk[idx_n[j],0], s_nk[idx_n[j],0]).ljust(16)
			txt += '{:.3g}\u00b1{:.3g}'.format(m_nk[idx_n[j],1], s_nk[idx_n[j],1]).ljust(16)
			txt += '({})'.format(', '.join(['{:.4g}'.format(v) for v in uni_to_raw(grid_X_nd[idx_n[j],:], xbb)]))
			print(txt)
		print('') # line break

		if args.show and (D - len(l_fix)==1):
			# 1-dim plot
			dim = list(set(range(D)) - set([d for d,val in l_fix]))
			xaxis_n = np.squeeze(all_u2r(grid_X_nd, xbb)[:,dim])

			m_nk, s_nk = bo.predict(grid_X_nd) # m_n, s_n are in raw scale
			# m_n = uni_to_raw(m_n, ybb)
			# s_n *= ybb[1,0] - ybb[0,0]

			cp = sns.color_palette('Set2')
			plt.figure(1, figsize=[18,5])
			plt.subplot(1,3,1)
			plt.plot(xaxis_n, acq_n, color=cp[2], lw=3)
			plt.title('acquisition')

			plt.subplot(1,3,2)
			# u_n0 = vegard.ev_to_wlen(m_nk[:,0]+s_nk[:,0])
			# u_n1 = vegard.ev_to_wlen(m_nk[:,0]-s_nk[:,0])
			# plt.fill_between(xaxis_n, np.max(np.c_[u_n0, u_n1], axis=1), np.min(np.c_[u_n0,u_n1], axis=1), color=cp[1], alpha=.5)
			plt.fill_between(xaxis_n, m_nk[:,0]+s_nk[:,0], m_nk[:,0]-s_nk[:,0], color=cp[1], alpha=.5)
			plt.plot(xaxis_n, m_nk[:,0], color=cp[1], lw=3, )
			# plt.plot(xaxis_n, np.nanmax(rawy_n)*np.ones_like(xaxis_n), lw=1.5, color=(.6,.6,.6))
			plt.title('prediction u')


			# observations to plot
			data_idx_m = np.ones(X_nd.shape[0])
			for d, u in l_fix:
				data_idx_m *= X_nd[:,d] >= (u - args.srange)
				data_idx_m *= X_nd[:,d] <= (u + args.srange)
			_x_m = uni_to_raw(np.squeeze(X_nd[data_idx_m>0,:][:,dim]), xbb[:,dim])
			_y_m = UV_nk[data_idx_m>0,:] #uni_to_raw(Y_n[data_idx_m>0], ybb)
			# plt.plot(_x_m, _y_m[:,0], marker='+', mew=3, markersize=10, lw=0, color=cp[1], alpha=.8)
			plt.plot(_x_m[:7], _y_m[:7,0], marker='^', mew=3, markersize=10, lw=0, color=cp[1], alpha=.8)
			if _x_m.shape[0]>7:
				plt.plot(_x_m[7:], _y_m[7:,0], marker='o', mew=3, markersize=10, lw=0, color=cp[1], alpha=.8)
				


			plt.subplot(1,3,3)
			# u_n0 = vegard.gc_to_ppm(m_nk[:,1]+s_nk[:,1])
			# u_n1 = vegard.gc_to_ppm(m_nk[:,1]-s_nk[:,1])
			# plt.fill_between(xaxis_n, np.max(np.c_[u_n0, u_n1], axis=1), np.min(np.c_[u_n0,u_n1], axis=1), color=cp[1], alpha=.5)
			plt.fill_between(xaxis_n, m_nk[:,1]+s_nk[:,1], m_nk[:,1]-s_nk[:,1], color=cp[1], alpha=.5)
			plt.plot(xaxis_n, m_nk[:,1], color=cp[1], lw=3, )
			plt.title('prediction v')
			# plt.plot(_x_m, _y_m[:,1], marker='+', mew=3, markersize=10, lw=0, color=cp[1], alpha=.8)
			plt.plot(_x_m[:7], _y_m[:7,1], marker='^', mew=3, markersize=10, lw=0, color=cp[1], alpha=.8)
			if _x_m.shape[0]>7:
				plt.plot(_x_m[7:], _y_m[7:,1], marker='o', mew=3, markersize=10, lw=0, color=cp[1], alpha=.8)


			# if args.csv is not None:
			# 	save_array = np.c_[xaxis_n, acq_n*(ybb[1,0] - ybb[0,0]),m_n,s_n]
			# 	np.savetxt(args.csv, save_array, fmt='%.6e', delimiter=',', header='x, acquisition, mean, std')

			plt.show()

		elif args.show and (D - len(l_fix)==2):
			# 2-dim plot
			dim = sorted(list(set(range(D)) - set([d for d,val in l_fix])))
			shape = (args.ngrid,)*2
			xaxis_ijd = all_u2r(grid_X_nd, xbb)[:,dim].reshape(shape+(len(dim),))
			# extent = [xaxis_ijd[0,0,1], xaxis_ijd[0,-1,1], xaxis_ijd[0,0,0], xaxis_ijd[-1,0,0]]
			extent = [xaxis_ijd[0,0,0], xaxis_ijd[-1,0,0], xaxis_ijd[0,0,1], xaxis_ijd[0,-1,1]]

			m_nk, s_nk = bo.predict(grid_X_nd) # prediction is in raw scale already

			cm0 = sns.cubehelix_palette(start=.5, rot=-.5, reverse=False, as_cmap=True)
			# cm1 = sns.cubehelix_palette(reverse=True, as_cmap=True)
			cm1 = sns.color_palette('coolwarm', as_cmap=True)
			cm2 = sns.cubehelix_palette(start=.05, gamma=1.2, reverse=True, as_cmap=True)
			cp = sns.color_palette('Set2')
			data_idx_m = np.ones(X_nd.shape[0])
			for d, u in l_fix:
				data_idx_m *= X_nd[:,d] >= (u - args.srange)
				data_idx_m *= X_nd[:,d] <= (u + args.srange)
			_x_md = all_u2r(X_nd[data_idx_m>0,:][:,dim], xbb[:,dim])

			next_d = uni_to_raw(grid_X_nd[idx_n[0],:], xbb)

			plt.figure(1,figsize=[11,10])
			plt.subplot(2,2,1)
			lognorm = matplotlib.colors.LogNorm(vmin=0.01, vmax=1000)
			plt.imshow((-acq_n).reshape(shape).T, aspect='auto', origin='lower', cmap=cm0, norm=lognorm, extent=extent)
			if _x_md.size > 0:
				# plt.plot(_x_md[:,0], _x_md[:,1], marker='+', mew=2, markersize=9, lw=0, color=cp[1], alpha=.7)
				plt.plot(_x_md[:7,0], _x_md[:7,1], marker='^', mew=2, markersize=7, lw=0, color=cp[1], alpha=.7)
				if _x_md.shape[0]>7:
					plt.plot(_x_md[7:,0], _x_md[7:,1], marker='o', mew=2, markersize=7, lw=0, color=cp[1], alpha=.7)
			plt.plot(next_d[0], next_d[1], marker='P', mew=1.2, markersize=12, color=(.1,.1,.1), markeredgecolor=(.9,)*3, alpha=.7)
			plt.colorbar()
			plt.title('negative acquisition')
			plt.subplot(2,2,3)
			# cn0 = matplotlib.colors.CenteredNorm(targetUV_k[0], 0.195)
			# cn1 = matplotlib.colors.CenteredNorm(targetUV_k[1], 0.2)
			cn0 = shiftedColorMap(cm1, midpoint=(targetUV_k[0] - 0)/0.4, name='0')
			cn1 = shiftedColorMap(cm1, midpoint=(targetUV_k[1] - 0.2)/(0.5-0.2), name='1')
			# plt.imshow(m_n.reshape(shape), vmin=vmin, vmax=vmax, aspect='auto', origin='lower', cmap=cm1, extent=extent)
			# plt.imshow(m_nk[:,0].reshape(shape).T, aspect='auto', origin='lower', cmap=cm1, norm=cn0, extent=extent)
			plt.imshow(m_nk[:,0].reshape(shape).T, aspect='auto', origin='lower', cmap=cn0, norm=matplotlib.colors.Normalize(0,0.4), extent=extent)
			if _x_md.size > 0:
				# plt.plot(_x_md[:,0], _x_md[:,1], marker='+', mew=2, markersize=9, lw=0, color=(0.1,)*3, alpha=.7)
				plt.plot(_x_md[:7,0], _x_md[:7,1], marker='^', markersize=7, lw=0, color=(0.1,)*3, alpha=.7)
				if _x_md.shape[0]>7:
					plt.plot(_x_md[7:,0], _x_md[7:,1], marker='o', markersize=7, lw=0, color=(0.1,)*3, alpha=.7)
			plt.plot(next_d[0], next_d[1], marker='P', mew=0, markersize=12, color=(.2,0.8,.6), markeredgecolor=(.2,.8,.7), alpha=.7)
			plt.colorbar()
			plt.title('prediction x')
			plt.subplot(2,2,4)
			# plt.imshow((s_n).reshape(shape), vmin=vmin, vmax=vmax, aspect='auto', origin='lower', cmap=cm1, extent=extent)
			# plt.imshow(m_nk[:,1].reshape(shape).T, aspect='auto', origin='lower', cmap=cm1, norm=cn1, extent=extent)
			plt.imshow(m_nk[:,1].reshape(shape).T, aspect='auto', origin='lower', cmap=cn1, norm=matplotlib.colors.Normalize(0.2,0.5), extent=extent)
			if _x_md.size > 0:
				# plt.plot(_x_md[:,0], _x_md[:,1], marker='+', mew=2, markersize=9, lw=0, color=(0.1,)*3, alpha=.7)
				plt.plot(_x_md[:7,0], _x_md[:7,1], marker='^', markersize=7, lw=0, color=(0.1,)*3, alpha=.7)
				if _x_md.shape[0]>7:
					plt.plot(_x_md[7:,0], _x_md[7:,1], marker='o', markersize=7, lw=0, color=(0.1,)*3, alpha=.7)
			plt.plot(next_d[0], next_d[1], marker='P', mew=0, markersize=12, color=(.2,0.8,.6), markeredgecolor=(.2,.8,.7), alpha=.7)
			plt.colorbar()
			plt.title('prediction y')

			plt.show()