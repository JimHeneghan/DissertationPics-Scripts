#!/usr/local/apps/anaconda/3-2.2.0/bin/python
################################################################################
##### To be run from /scratch/dermoth/AgPaper/PlanarHM_Full/2_38um/HeatMap #####
################################################################################
import time 
import numpy as np
import scipy as sp
import itertools as it
import math
import collections as cl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH
from mpl_toolkits.axes_grid1.axes_divider import AxesLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["font.family"] = 'STIXGeneral'
pane   = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

# fig, ((ax1, ax0), (ax2, ax3)) = plt.subplots(2,2, figsize = (16,15))#, constrained_layout=True)

fig, axs = plt.subplots(2,5, figsize = (18,12))
Frame = []
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
shades = ['black', 'red', 'orange', 'yellow', 'lime', 'green', 'springgreen', 'teal', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']

# freq = ['6.1', '6.3', '6.5', '6.7', '6.8', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
# selectionBoxAli = [1, 2, 5, 6, 9]
# selectionBoxJim = [1, 2, 3, 4, 5, 6, 9]
sbox = [120, 110, 90, 50, 40]
q = 1
for z in range(0, 5):
	
	lam = 6.0 + 0.01*sbox[z] # 6.0 + 0.1*sbox 
	# Field = "%s_um.txt" %freq[z]
	Fieldx = "Raw/ExHeatMap%.2f.txt" %lam
	Fieldy = "Raw/EyHeatMap%.2f.txt" %lam
	Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(-2.13,  2.13, Ny)
	X = np.linspace(-1.23,  1.23, Nx)

	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )


	Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
	Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
	Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

	for i in range (0, Ny):
		Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
		Full[i] = np.sqrt(Full[i])
		# print(min(Full[i]))

	# norm = mpl.colors.Normalize(vmin=0, vmax=1) 
	if (z == 0):
		norm = mpl.colors.Normalize(vmin=0.0, vmax=0.41) 
	if (z == 1):
		norm = mpl.colors.Normalize(vmin=0.00, vmax=0.44) 
	if (z == 2):
		norm = mpl.colors.Normalize(vmin=0.00, vmax=0.75) 
	if (z == 3):
		norm = mpl.colors.Normalize(vmin=0.0, vmax=0.4) 
	if (z == 4):
		norm = mpl.colors.Normalize(vmin=0.0, vmax=0.4) 
	pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

	im = axs[q,z].pcolormesh(X, Y, 1e20*Full, norm=norm, **pc_kwargs)
	axs[q,z].set_aspect('equal')

	axs[q,z].text(-.35,1.0,'(%s) ' %pane[z+5],
	    horizontalalignment='center',
	    transform=axs[q,z].transAxes, fontsize = '25', fontweight = 'bold')
	# axs[q,z].set_xticks([])
	# axs[q,z].set_yticks([])
	axs[q,z].tick_params(which= 'major', length=15, width=3, labelsize=20)
	axs[q,z].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
	axs[q,z].set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '35')
	plt.setp(axs[q,z].spines.values(), linewidth=2)
# plt.tick_params(left = False, bottom = False)

# cbar = fig.colorbar(im, ax=ax0)
# cbar.set_label(label = r"$\rm |E| \ \ Field \ (arb. units)$", size = '20')

q = 0
#####################################################################3
Nx = 125
Ny = 250
sbox = [12, 11, 9, 5, 4]
for z in range(0,5):

	base  = 6.0 + 0.1*sbox[z]

	Fieldx = "Ali%.1f_um.txt" %base
	print(Fieldx)
	# Fieldy = "Ali%s_um.txt" %freq[z]
	# Fieldz = "Ali%s_um.txt" %freq[z] 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	Ex = np.zeros((Ny, Nx), dtype = np.double)
	# Ey = np.zeros((Ny, Nx), dtype = np.double)
	# Ez = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  10, Ny)
	X = np.linspace(0,  5, Nx)

	Full = np.loadtxt(Fieldx,  skiprows= 0, unpack =True )

	if (z == 0):
		norm = mpl.colors.Normalize(vmin=0.25, vmax=0.55)
		axs[q,z].set_title(r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), fontsize = '30', pad = 20)
	if (z == 1):
		norm = mpl.colors.Normalize(vmin=0.28, vmax=0.95) 
		axs[q,z].set_title(r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), fontsize = '30', pad = 20)
	if (z == 2):
		norm = mpl.colors.Normalize(vmin=0.15, vmax=0.450) 
		axs[q,z].set_title(r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), fontsize = '30', pad = 20)
	if (z == 3):
		norm = mpl.colors.Normalize(vmin=0.0, vmax=1.1) 
		axs[q,z].set_title(r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), fontsize = '30', pad = 20)
	if (z == 4):
		norm = mpl.colors.Normalize(vmin=0.0, vmax=1.1) 
		axs[q,z].set_title(r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), fontsize = '30', pad = 20)

	pc_kwargs = {'rasterized': True, 'cmap':'jet'}


	im = axs[q,z].pcolormesh(X, Y, Full, norm=norm, **pc_kwargs, snap = True)
	axs[q,z].set_aspect('equal')#, anchor = (0.1, 0.5), adjustable = 'box')

	axs[q,z].text(-.35,1.0,'(%s) ' %pane[z],
	    horizontalalignment='center',
	    transform=axs[q,z].transAxes, fontsize = '25', fontweight = 'bold')
	# axs[q,z].set_xticks([])
	# axs[q,z].set_yticks([])
	axs[q,z].tick_params(which= 'major', length=15, width=3, labelsize=20)
	axs[q,z].set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
	axs[q,z].set_ylabel(r"$ \rm y \ (\mu m)$", fontsize = '35')
	plt.setp(axs[q,z].spines.values(), linewidth=2)


plt.tight_layout()
plt.savefig("Fig6.png")
plt.savefig("Fig6.pdf")