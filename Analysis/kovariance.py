#kovariance.py by Kovi Bodek (2025)
#functions to assist in covariance analysis of timepix data

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm, Normalize
from matplotlib import colormaps, colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import poisson

#Colormap similar to jet but becomes white at 0
def get_jetwhite(breakp=1):
	turbo = plt.get_cmap('jet', 256)
	jetwhitecolors = turbo(np.linspace(0, 1, 256))
	jetwhitecolors[:breakp, :] = np.linspace([1, 1, 1, 1], jetwhitecolors[breakp, :], breakp)
	return colors.ListedColormap(jetwhitecolors, name='jetwhite')

#Return object for 2D covariance
#You can access results like cov_object.poscov
#Or plot easily with cov_object.PlotSummary()
class Cov2DResult:
	def __init__(self, EXY, EXEY, bins1, bins2):
		self.EXY = EXY
		self.EXEY = EXEY
		self.bins1 = bins1
		self.bins2 = bins2

		self.covariance = self.EXY - self.EXEY
		self.poscov = self.covariance * (self.covariance>0)
		self.negcov = self.covariance * (self.covariance<0) * -1

	def PlotSummary(self, axis1="", axis2="", title="Covariance Summary", figsize=(9,8), polar=False, logscale = None, cmap="viridis", xlim=None, ylim=None, clim=None, only_positive=False):
		if logscale is None: 
			if clim is None: map_norm = Normalize(vmin=np.min(self.poscov[self.poscov!=0]),vmax=np.max(self.poscov))
			else: map_norm = Normalize(vmin=clim[0],vmax=clim[1])
		elif logscale is True: 
			if clim is None: map_norm = LogNorm(vmin=np.min(self.poscov[self.poscov!=0]),vmax=np.max(self.poscov))
			else: LogNorm(vmin=clim[0],vmax=clim[1])
		else: 
			map_norm = LogNorm(vmin=logscale[0],vmax=logscale[1])

		if polar: kw = {'projection': 'polar'}
		else: kw = {}
		if only_positive:
			fig, ax = plt.subplots(subplot_kw = kw, figsize = figsize)
			pos=ax.pcolormesh(self.bins2, self.bins1, self.poscov, norm=map_norm, cmap=cmap)
			plt.colorbar(pos, ax=ax)
			if not polar:
				ax.set_xlabel(axis1)
				ax.set_ylabel(axis2)
			ax.set_title("Positive Covariance")
		else:
			fig, ax = plt.subplots(subplot_kw = kw, ncols=2, nrows=2, figsize = figsize)
			hist=ax[0][0].pcolormesh(self.bins2, self.bins1, self.EXY, norm=map_norm, cmap=cmap)
			back=ax[1][0].pcolormesh(self.bins2, self.bins1, self.EXEY, norm=map_norm, cmap=cmap)
			pos=ax[0][1].pcolormesh(self.bins2, self.bins1, self.poscov, norm=map_norm, cmap=cmap)
			neg=ax[1][1].pcolormesh(self.bins2, self.bins1, self.negcov, norm=map_norm, cmap=cmap)

			plt.colorbar(hist, ax=ax[0][0])
			plt.colorbar(back, ax=ax[1][0])
			plt.colorbar(pos, ax=ax[0][1])
			plt.colorbar(neg, ax=ax[1][1])
			if not polar:
				ax[1][0].set_xlabel(axis1)
				ax[1][1].set_xlabel(axis1)
				ax[0][0].set_ylabel(axis2)
				ax[1][0].set_ylabel(axis2)

			ax[0][0].set_title("EXY")
			ax[1][0].set_title("EXEY")
			ax[0][1].set_title("Positive Covariance")
			ax[1][1].set_title("Negative Covariance")
			fig.suptitle(title)
			plt.tight_layout()

		if not xlim is None:
			plt.setp(ax, xlim=xlim)
		if not ylim is None:
			plt.setp(ax, ylim=ylim)
		plt.show()

#Return object for 1D covariance
#You can access results like cov_object.covariance
#Or plot easily with cov_object.PlotSummary()
class Cov1DResult:
	def __init__(self, EXY, EXEY, bins):
		self.EXY = EXY
		self.EXEY = EXEY
		self.bins = bins
		self.covariance = self.EXY - self.EXEY

	def PlotSummary(self, axis="", title="Covariance Summary", figsize=(15,5), lim=None, logscale = False):

		fig, ax = plt.subplots(ncols=2, figsize = figsize)
		if logscale:
			ax[0].semilogy(self.bins[1:],self.EXY)
			ax[0].semilogy(self.bins[1:],self.EXEY)
			ax[1].semilogy(self.bins[1:],self.covariance)
		else:
			ax[0].plot(self.bins[1:],self.EXY)
			ax[0].plot(self.bins[1:],self.EXEY)
			ax[1].plot(self.bins[1:],self.covariance)

		ax[0].set_title("EXY/EXEY")
		ax[1].set_title("Covariance")

		plt.setp(ax, xlabel=axis)
		if not lim is None:
			plt.setp(ax, xlim=lim)

		fig.suptitle(title)
		plt.tight_layout()
		plt.show()

class TPXItem:
	def __init__(self, trig, tof, x, y, tot=None):
		self.trig=trig.astype(int)
		self.tof = tof
		self.x = x
		self.y = y
		self.tot = tot

		self.centerx = None
		self.centery = None
		self.centert = None
		self.x0 = None
		self.y0 = None
		self.t0 = None
		self.radii = None
		self.angles = None

		self.eV = None

	def center(self, x0, y0, t0, summary=True, title=""):
		self.x0 = x0
		self.y0 = y0
		self.t0 = t0
		self.centerx = self.x-self.x0
		self.centery = self.y-self.y0
		self.centert = self.tof-self.t0
		self.radii = np.sqrt(self.centerx**2 + self.centery**2)
		self.angles = np.arctan2(self.centery, self.centerx)

		if summary:
			self.center_summary(title=title)
	def calibrate_energy(self, VMI_voltage, experiment_t0, mass, k_xy = 0.0026/2400, k_t= 0.0035):
		self.mass = mass
		self.k_xy = k_xy
		self.k_t = k_t
		self.VMI_voltage = VMI_voltage
		self.experiment_t0 = experiment_t0
		calibrated_k_xy = k_xy * self.VMI_voltage
		zeroed_tof = self.tof-self.experiment_t0
		x_eV = calibrated_k_xy * self.centerx**2
		y_eV = calibrated_k_xy * self.centery**2
		t_eV = k_t * (((self.t0-self.experiment_t0)**2-(zeroed_tof)**2) * self.VMI_voltage / (zeroed_tof))**2 / self.mass
		self.eV = x_eV+y_eV+t_eV

	def energy_gate(self, min_eV=0, max_eV=None):
		index = (self.eV>=min_eV)
		if not max_eV is None:
			index = index & (self.eV<=max_eV)
		temparray = np.array([self.trig, self.tof, self.x, self.y, self.tot]).T
		returnitem = TPXItem(*temparray[index].T)
		returnitem.center(self.x0, self.y0, self.t0, title="Energy Gate")
		returnitem.calibrate_energy(self.VMI_voltage, self.experiment_t0, self.mass, k_xy = self.k_xy, k_t = self.k_t)
		return returnitem
	def get_subset(self, mintof=0, maxtof=None, minradius=None, maxradius=None, maxangle=None, index = None, summary=True):
		if index is None:
			index = (self.tof>=mintof)
		if not maxtof is None: 
			index = index & (self.tof<=maxtof)
		if not minradius is None: 
			index = index & (self.radii>=minradius)
		if not maxradius is None: 
			index = index & (self.radii<=maxradius)
		if not maxangle is None: 
			index = index & ((np.abs(self.angles)<maxangle) | (np.abs(self.angles)>(np.pi-maxangle)))

		temparray = np.array([self.trig, self.tof, self.x, self.y, self.tot]).T
		returnitem = TPXItem(*temparray[index].T)
		returnitem.center(self.x0, self.y0, self.t0, title="Subset", summary=summary)
		if not self.eV is None:
			returnitem.calibrate_energy(self.VMI_voltage, self.experiment_t0, self.mass, k_xy = self.k_xy, k_t = self.k_t)
		return returnitem
	def center_summary(self, title=""):
		fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
		ion1hist, ion1bins = np.histogram(self.tof,bins=100)
		ax[0][0].set_ylabel("Count")
		ax[0][0].set_xlabel("Microseconds")
		ax[0][0].axvline(x=self.t0, color='r', linewidth=0.5, label=f"Center: ({self.t0})")
		ax[0][0].set_title("t0")
		ax[0][0].plot(ion1bins[:-1],ion1hist)
		ax[0][0].legend(loc="lower left")

		xybins = (np.arange(256),np.arange(256))
		rbins = np.linspace(0,150,101)
		tbins = np.arange(np.min(self.centert),np.max(self.centert),0.002)
		abins = np.linspace(-math.pi,math.pi,151)

		centerhist, _, _ = np.histogram2d(self.x,self.y,bins=xybins)
		ax[1][0].pcolor(*xybins, centerhist.T, norm=LogNorm(vmin=centerhist[centerhist!=0].min(), vmax=centerhist.max()))
		ax[1][0].axvline(x=self.x0, color='r', linewidth=0.5, label=f"x0 = {self.x0}")
		ax[1][0].axhline(y=self.y0, color='orange', linewidth=0.5, label=f"y0 = {self.y0}")
		ax[1][0].legend(loc="lower left")
		ax[1][0].set_title("x0 and y0")

		polarhist, _, _ = np.histogram2d(self.radii,self.angles,bins=(rbins, abins))
		ax[0][1].pcolor((abins*180/np.pi), rbins, polarhist, norm=LogNorm(vmin=polarhist[polarhist!=0].min(), vmax=polarhist.max()))
		ax[0][1].set_xlabel("Angle (deg)")
		ax[0][1].set_ylabel("Distance to center (pixels)")
		ax[0][1].set_title("Radius/Angle Distribution")

		xyt_hist, _, _ = np.histogram2d(self.radii,self.centert,bins=(rbins, tbins))
		ax[1][1].pcolor(tbins, rbins, xyt_hist)
		ax[1][1].axvline(x=0, color='white', linewidth=0.5)
		ax[1][1].set_xlabel("Relative TOF")
		ax[1][1].set_ylabel("xy radius (pix)")
		ax[1][1].set_title("xy radius vs relative TOF")
		fig.suptitle(title)
		plt.show()
	def get_polarhist(self, abins, rbins):
		return np.histogram2d(self.radii,self.angles,bins=(rbins, abins))[0]



def tofgating(trig, tof, x, y, tot, gates, tofbins = np.arange(4,8.5,0.005)):
	tofhist, tofbins = np.histogram(tof,bins=tofbins)
	plt.figure(figsize=(15, 5))
	plt.semilogy(tofbins[:-1],tofhist)

	for gate in gates:
		plt.axvline(x=gate[0], color='r', linewidth=0.5)
		plt.axvline(x=gate[1], color='r', linewidth=0.5)

	plt.minorticks_on()
	plt.ylabel("Count")
	plt.xlabel("Microseconds")
	plt.title("Gating")
	plt.show()

	temparray = np.array([trig, tof, x, y, tot]).T
	items = []
	for gate in gates:
		gate_index = (tof>gate[0]) & (tof<gate[1])
		temptrig, temptof, tempx, tempy, temptot = temparray[gate_index].T
		items.append(TPXItem(temptrig.astype(int), temptof, tempx, tempy, temptot))
	return items


def radius_covariance(item1, item2, rbins=np.arange(0,90)):
	if (item1.radii is None) or (item2.radii is None):
		print("Items not centered")
		return
	return cov2D_type1(item1.trig, item2.trig, item1.radii, item2.radii, rbins, rbins)
def angle_covariance(item1, item2, abins = np.linspace(-math.pi,math.pi,151)):
	if (item1.angles is None) or (item2.angles is None):
		print("Items not centered")
		return
	return cov2D_type1(item1.trig, item2.trig, item1.angles, item2.angles, abins, abins)

def dication_KER_plot(item1, item2, KERbins = np.linspace(0,7,120)):
	item1hist, _ = np.histogram(item1.eV, bins=KERbins)
	item2hist, _ = np.histogram(item2.eV, bins=KERbins)
	item1KER, _ = np.histogram(item1.eV * (1+item1.mass/item2.mass), bins=KERbins)
	item2KER, _ = np.histogram(item2.eV * (1+item2.mass/item1.mass), bins=KERbins)

	fig, ax = plt.subplots(ncols=2, figsize=(15,5))

	ax[0].plot(KERbins[:-1],item1hist, label="item1 energy")
	ax[0].plot(KERbins[:-1],item2hist, label="item2 energy")
	ax[0].legend()
	ax[1].plot(KERbins[:-1],item1KER, label="KER calculated from item1")
	ax[1].plot(KERbins[:-1],item2KER, label="KER calculated from item2")
	ax[1].legend()
	plt.show()


#Special function for x-x covariance like tof-tof
def self_covariance(trig, p, bins, verbose=True):
	if verbose: print("Preparing...")
	phist,pbins = np.histogram(p,bins=bins)
	trigindex = (trig - trig[0])
	diff_trigindex = np.diff(trigindex) #>0 if new shot
	EXEY = phist[:, np.newaxis] * phist[np.newaxis, :]
	EXY = np.zeros_like(EXEY) #empty cov array
	ind_p = np.digitize(p, pbins,right=True)-1
	ind_p[(ind_p < 0)] = -1
	ind_p[ind_p>=(len(pbins)-1)] = -1
	shots = []
	last_ind = 0
	if verbose: print("Splitting...")
	for ind in range(len(diff_trigindex)): #splits into shots 
		if diff_trigindex[ind]>0:
			shots.append(ind_p[last_ind:ind+1])
			last_ind = ind+1
	if verbose: print("Incrementing array...")
	for shotn in range(len(shots)): #increments cov array for each combination in shot
	    shot = shots[shotn]
	    shot = shot[shot!=-1]
	    for i in range(len(shot)-1):
	        for j in range(i+1,len(shot)):
	            EXY[shot[i], shot[j]]+=1
	            EXY[shot[j], shot[i]]+=1
	    if (verbose and shotn%300==0): print(f"{round(100*shotn/len(shots), 2)}%", end="\r")
	if verbose: print("\nDone.")

	EXEY=(EXEY)/np.sum(diff_trigindex)**2.0
	EXY=(EXY.astype(float))/np.sum(diff_trigindex)

	return Cov2DResult(EXY,EXEY,pbins,pbins)

#2D covariance of type Cov(N1(q1), N2(q2))
def cov2D_type1(trigindex1, trigindex2, q1, q2, bins1, bins2):
	print("Preparing...")
	hist1,_ = np.histogram(q1,bins=bins1)
	hist2,_= np.histogram(q2,bins=bins2)
	EXEY = hist1[:, np.newaxis] * hist2[np.newaxis, :]
	EXY = np.zeros_like(EXEY)
	ind_q1 = np.digitize(q1, bins1,right=True)-1
	ind_q2 = np.digitize(q2, bins2,right=True)-1
	ind_q1[(ind_q1 < 0)] = -1
	ind_q2[(ind_q2 < 0)] = -1
	ind_q1[ind_q1>=(len(bins1)-1)] = -1
	ind_q2[ind_q2>=(len(bins2)-1)] = -1

	n_shots = max(np.max(trigindex1), np.max(trigindex2))+1
	shots = [([],[]) for i in range(n_shots+1)]

	print("Splitting...")
	for i in range(len(trigindex1)):
		shots[trigindex1[i]][0].append(ind_q1[i])
	for i in range(len(trigindex2)):
		shots[trigindex2[i]][1].append(ind_q2[i])



	print("Incrementing array...")
	for shot in range(len(shots)):
		for i in range(len(shots[shot][0])):
			for j in range(len(shots[shot][1])):
				if (shots[shot][0][i]==-1 or shots[shot][1][j]==-1):
					continue
				EXY[shots[shot][0][i], shots[shot][1][j]]+=1
		if (shot%100==0): print(f"{round(100*shot/len(shots), 2)}%", end="\r")
	print("\nDone.")

	EXEY=(EXEY)/n_shots**2.0
	EXY=(EXY.astype(float))/n_shots
	
	return Cov2DResult(EXY,EXEY,bins1,bins2)

#2D covariance of type Cov(N1(p, q), N2)
def cov2D_type2(trigindex1, trigindex2, p, q, pbins, qbins, verbose=True):
	if verbose: print("Preparing...")
	EXEY = np.histogram2d(p, q,bins=(pbins,qbins))[0].astype(float)
	EXY = np.zeros_like(EXEY)
	ind_q = np.digitize(q, qbins,right=True)-1
	ind_p = np.digitize(p, pbins,right=True)-1

	ind_q[(ind_q < 0)] = -1
	ind_p[(ind_p < 0)] = -1
	ind_q[ind_q>=(len(qbins)-1)] = -1
	ind_p[ind_p>=(len(pbins)-1)] = -1

	n_shots = int(np.max(trigindex1)+1)
	weight = np.histogram(trigindex2, np.arange(n_shots+1))[0]

	if verbose: print("Incrementing array...")
	for i in range(len(trigindex1)):
		if (ind_p[i]==-1 or ind_q[i]==-1):continue
		EXY[ind_p[i], ind_q[i]]+=weight[trigindex1[i]]
		if (verbose and i%1000==0): print(f"{round(100*i/len(trigindex1), 2)}%", end="\r")
	if verbose: print("\nDone.")

	EXY  /= n_shots
	EXEY *= (np.sum(weight)/(n_shots**2.0))  #np.sum(weight)/n_shots is equivalent to <y>

	return Cov2DResult(EXY, EXEY, pbins, qbins)

@np.vectorize
def coco_background(meas_counts, bg_dist, *bg_dist_args):
	"""Estimate background counts given a background distribution and observed counts.

	Parameters
	----------
	meas_counts
		number of observed (signal + background) counts in observing run
	bg_dist
		scipy.stats distribution of background counts expected over total observing run (e.g. scipy.stats.poisson)
	*bg_dist_args
		arguments passed to scipy to inform the distribution (for poisson, just the mean background counts)
	"""
	acc = 0.0
	for b in range(meas_counts + 1):
		acc += b * bg_dist.pmf(b, *bg_dist_args)
	return acc / bg_dist.cdf(meas_counts, *bg_dist_args)

#2D corrected coincidence
def coco2D_type2(trigindex1, trigindex2, p, q, pbins, qbins, verbose=True, simple=True):
	if verbose: print("Preparing...")

	histogram = np.zeros((len(pbins), len(qbins)))
	background = np.zeros((len(pbins), len(qbins)))
	n_shots = np.max(trigindex1) + 1
	coincidences_each_shot = np.histogram(trigindex2, n_shots + 1)[0]

	ind_q = np.digitize(q, qbins, right=True) - 1
	ind_p = np.digitize(p, pbins, right=True) - 1
	ind_q[ind_q < 0] = -1
	ind_p[ind_p < 0] = -1
	ind_q[ind_q >= len(qbins) - 1] = -1
	ind_p[ind_p >= len(pbins) - 1] = -1

	coincidence_shots = 0
	last_coincident_shot = -1
	if verbose: print("Incrementing array...")
	for i, trig in enumerate(trigindex1):
		if ind_p[i] == -1 or ind_q[i] == -1: continue
		coincidences_this_shot = coincidences_each_shot[trig]
		if coincidences_this_shot > 0:
			histogram[ind_p[i], ind_q[i]] += coincidences_this_shot
			if trig > last_coincident_shot:
				last_coincident_shot = trig
				coincidence_shots += 1
		else:
			background[ind_p[i], ind_q[i]] += 1
		if verbose and i%1000==0: print(f"{round(100*i/len(trigindex1), 2)}%", end="\r")
	if verbose: print(f"Done. {coincidence_shots} coincidences out of {n_shots}")

	if simple:
		background /= (n_shots - coincidence_shots)
		histogram /= coincidence_shots
	else:
		background *= coincidence_shots / (n_shots - coincidence_shots)
		background = coco_background(histogram.astype(int), poisson, background)
		background /= coincidence_shots
		histogram /= coincidence_shots

	return Cov2DResult(histogram, background, pbins, qbins)


#1D covariance of type Cov(N1(p), N2)
def cov1D(trigindex1, trigindex2, p, bins, verbose=True):
	if verbose: print("Preparing...")
	EXEY = np.histogram(p, bins)[0].astype(float)
	EXY = np.zeros_like(EXEY)
	ind = np.digitize(p, bins, right=True)-1

	ind[(ind < 0)] = -1
	ind[ind>=(len(bins)-1)] = -1

	n_shots = int(np.max(trigindex1)+1)
	weight = np.histogram(trigindex2, np.arange(n_shots+1))[0]

	if verbose: print("Incrementing array...")
	for i in range(len(trigindex1)):
		if ind[i]==-1: continue
		EXY[ind[i]]+=weight[trigindex1[i]]
		if (i%1000==0 and verbose): print(f"{round(100*i/len(trigindex1), 2)}%", end="\r")
	if verbose: print("\nDone.")

	EXY  /= n_shots
	EXEY *= (np.sum(weight)/(n_shots**2.0))  #np.sum(weight)/n_shots is equivalent to <y>
	return Cov1DResult(EXY, EXEY, bins)

###################################################################
#FUNCTIONS BELOW THIS LINE ARE MORE SPECIALIZED OR WORK IN PROGRESS
###################################################################


#Prints tot-tof histogram
def timewalk_summary(tof, tot, tofbins=np.arange(4,8.5,0.005), totbins = np.arange(0,0.5,0.0255),title=""):
	full_timewalk_hist, full_timewalk_bin1, full_timewalk_bin2 = np.histogram2d(tot,tof,bins=(totbins,tofbins))
	plt.pcolor(full_timewalk_bin2, full_timewalk_bin1,full_timewalk_hist,norm=LogNorm(vmin=np.min(full_timewalk_hist[full_timewalk_hist>0]), vmax = np.max(full_timewalk_hist)))
	plt.colorbar()
	plt.title(title)
	plt.show()


def inv_func(x, k, a, b): return k/(x+a)+b


#returns a correcting function for tof in a given range
#for example,
#fix = get_timewalk_fit(<params>)
#corr_tof = fix(tof)
#timewalk_summary(corr_tof, tot)
def get_timewalk_fit(tof, tot, totrange, tofrange):
	b = (np.arange(*totrange,0.0255),np.arange(*tofrange,0.005)) #microseconds
	timewalk_hist, timewalk_bin1, timewalk_bin2 = np.histogram2d(tot,tof,bins=b)
	mom = timewalk_hist*b[1][np.newaxis,:-1]
	averages = np.sum(mom, axis=1)/np.sum(timewalk_hist,axis=1)
	x_grid=b[0][:-1]
	print(len(averages), len(x_grid))
	is_nan=np.isnan(averages)
	averages=averages[~is_nan]
	x_grid=x_grid[~is_nan]
	print(len(averages), len(x_grid))
	plt.pcolor(b[1],b[0],timewalk_hist,norm=LogNorm(vmin=np.min(timewalk_hist[timewalk_hist>0]), vmax = np.max(timewalk_hist)))
	plt.plot(averages,x_grid)
	plt.show()
	fit_params, _ = scipy.optimize.curve_fit(inv_func, x_grid, averages)

	plt.plot(x_grid,averages, label="Average TOF for given TOT in peak")
	plt.plot(b[0][:-1], inv_func(b[0][:-1], *fit_params), label=f"Inverse fit")
	plt.xlabel("TOT (μs)")
	plt.ylabel("TOF (μs)")
	plt.title("Timewalk Fit")
	plt.legend()
	plt.show()
	print(f"tof = [ION t0 {round(fit_params[2],3)}] + {round(fit_params[0],3)} / (tot + {round(fit_params[1],3)})")
	def outfunc(tof):
		adjustment = fit_params[0]/(fit_params[1]+tot)
		return tof-adjustment
	return outfunc

#Makes a GIF of 1D covariance progressively adding shots
def cov1DMovie(trigindex1, trigindex2, p, bins, frames=100, title="CovMovie"):
	fig, ax = plt.subplots()
	line, = ax.plot([], [])
	ax.set_xlim(np.min(bins),np.max(bins))
	maxtrig = np.max(trigindex1)
	def animate(i):
		print(f"[{i} / {frames}]", end="\r")
		trig1gate = trigindex1<round(maxtrig*(i+1)/frames)
		trig2gate = trigindex2<round(maxtrig*(i+1)/frames)
		frame = cov1D(trigindex1[trig1gate],trigindex2[trig2gate], p[trig1gate], bins, verbose=False).covariance
		line.set_data(bins[1:], frame)
		ax.set_ylim(0,np.max(frame))
		ax.set_title(f"{round(100*(i+1)/frames, 1)}% of shots")
		
		return line,
	ani = FuncAnimation(fig, animate, frames=frames, interval=50)
	ani.save(title+".gif", writer=PillowWriter(fps=30) )

#Makes a GIF of 1D covariance progressively adding shots (compare two on the same plot)
#This function is a little bit situational so you might need to modify depending on your goal
def cov1DMovieComparison(trigindex1, trigindex2, trigindex3, p, bins, frames=100, title="CovMovie", label1="A", label2="B"):
	fig, ax = plt.subplots()
	line1, = ax.plot([], [], label=label1)
	line2, = ax.plot([], [], label=label2)
	ax.set_xlim(np.min(bins),np.max(bins))
	ax.legend()
	maxtrig = np.max(trigindex1)
	def animate(i):
		print(f"[{i} / {frames}]", end="\r")
		trig1gate = trigindex1<round(maxtrig*(i+1)/frames)
		trig2gate = trigindex2<round(maxtrig*(i+1)/frames)
		trig3gate = trigindex3<round(maxtrig*(i+1)/frames)
		frame1 = cov1D(trigindex1[trig1gate],trigindex2[trig2gate], p[trig1gate], bins, verbose=False).covariance
		frame2 = cov1D(trigindex1[trig1gate],trigindex3[trig3gate], p[trig1gate], bins, verbose=False).covariance
		line1.set_data(bins[1:], frame1)
		line2.set_data(bins[1:], frame2)
		ax.set_ylim(0,max(np.max(frame1),np.max(frame2)))
		ax.set_title(f"{round(100*(i+1)/frames, 1)}% of shots")
		
		return line1, line2
	ani = FuncAnimation(fig, animate, frames=frames, interval=50)
	ani.save(title+".gif", writer=PillowWriter(fps=30) )


#A function to do type 2 covariance with a sliding window with which to calculate covariance

#For example, covariance of electrons (x, y) with a sliding window of time of flight
#In this case, trigindex1 would be electrons trigger, and p, q would be x and y.
#trigger2 would be trigger index for ions, with p2 being the corresponding ion TOF
#p2width is the width of the TOF window, and p2range is the range to slide over
#you might set conversion_function to a tof_to_mass function

#Will choose a number of locations based on an even division of p2range for the number of frames

def cov2D_SlidingMovie(trigindex1, trigindex2, p, q, p2, pbins, qbins, p2width, p2range, frames=10, fps=10, title="Cov2DMovie", conversion_function=None):
	p2locations = np.linspace(p2range[0]+p2width/2, p2range[1]-p2width/2, frames)

	data = np.empty((frames, len(pbins)-1, len(qbins)-1)) #whole movie will be stored here
	print("Calculating...")
	for i in range(len(p2locations)):
		print(f"[{i} / {frames}]", end="\r")
		frame_trigindex2 = trigindex2[(p2>(p2locations[i]-p2width/2)) & (p2<(p2locations[i]+p2width/2))] #if p2 in range
		data[i] = cov2D_type2(trigindex1, frame_trigindex2, p, q, pbins, qbins, verbose = False).poscov #covariance with this range
	
	print("Rendering Movie...") #plotting
	fig, ax = plt.subplots()
	cax = ax.pcolormesh(qbins, pbins, np.zeros_like(data[0]))
	fig.colorbar(cax)
	def animate(i):
		cax.set_array(data[i].flatten())
		cax.set_clim(vmin=0, vmax=np.max(data[i])) #renormalized colorbar
		if conversion_function is None:
			plt.title(f'({round(p2locations[i]-p2width/2,1)}, {round(p2locations[i]+p2width/2,1)})')
		else:
			plt.title(f'{round(conversion_function(p2locations[i]),1)} ({round(p2locations[i],1)})')
	anim = FuncAnimation(fig, animate, frames=frames)
	anim.save(f"{title}.gif", writer=PillowWriter(fps=fps))
	print(f"Done. Saved as {title}.gif")


#does self covariance like tof-tof but splits the dataset into a number of slices according to division
#division integer n>=1
def microCov_self(trig, p, bins, division):
	print(f"Covariance dividing into {division} sections:")
	print(f"[0 / {division}]", end="\r")
	trig-=trig[0]
	cuts = np.round(np.linspace(0,np.max(trig),division+1))
	bool_index = (trig<=cuts[1])
	outresult = self_covariance(trig[bool_index], p[bool_index], bins, verbose=False).poscov
	print(f"[1 / {division}]", end="\r")
	for i in range(len(cuts)):
		if i<2: continue
		bool_index = (trig<=cuts[i]) & (trig>cuts[i-1])
		outresult += self_covariance(trig[bool_index], p[bool_index], bins, verbose=False).poscov
		print(f"[{i} / {division}]", end="\r")
	print("\nDone.")
	#returns 2D array of summed positive covariance with axis equal to the bins argument
	return outresult