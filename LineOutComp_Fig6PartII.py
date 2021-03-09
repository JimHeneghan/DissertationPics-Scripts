#!/usr/local/apps/anaconda/3-2.2.0/bin/python

################################################################################
##### To be run from /scratch/dermoth/AgPaper/PlanarHM_Full/2_38um/HeatMap #####
################################################################################


import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import colors as c
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
# import matplotlib.pyplot.rcParams
# import matplotlib.font_manager

from matplotlib.font_manager import FontProperties

plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["font.family"] = 'STIXGeneral'
pane   = ['(k)', '(l)', '(c)']
#______________________________________________________________________#
############################## |E| Field ###############################
#______________________________________________________________________#

fig, (ax2, ax3)= plt.subplots(1,2, figsize = (18,12))#, constrained_layout=True)
# plt.gcf().subplots_adjust(left = 0.05, right = 0.99, top = 0.97, bottom = -0.1)

# mpl.rcParams['axes.linewidth'] = 10
Frame = []
Nx = 119
Ny = 207
z = 0
dt = 5.945e-18
T = 2e6
#print(z)
shades = ['red', 'limegreen', 'blue', 'purple', 'magenta', 'red', 'orange', 'gold', 'green', 'blue', 'purple', 'magenta']

freq = ['6.1', '6.3', '6.5', '6.7', '6.8', '6.9', '7.0', '7.1', '7.2', '7.5', '8.0', '9.0', '9.5', '10.0', '10.5']
selectionBoxAli = [1, 2, 5, 6, 8, 9]
selectionBoxJim = [1, 2, 3, 4, 5, 6, 8, 9]

ShadesRed  = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']

sboxAli = [12, 11, 9, 5, 4]


for z in range(0, 5):

	base  = 6.0 + 0.1*sboxAli[z]
	Field = "../../AliData/%s_um.txt" %base
	# z = z
	x, E = np.loadtxt(Field, usecols=(0,1), skiprows= 3, unpack =True )
	x = x - max(x)/2
	# print(min(E))
	# print(max(E))
	# if (z == 2):
	# 	E = E + 0.15
	# mult = 0.4
	# if (z > 1):
	# 	mult = 0.25
	print(E[0])
	print(len(E))

	if ((z == 4)):
		ax2.plot(x*1e6, 3*E  - 3*E[0] + 0.35*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	if ( (z == 3)):
		ax2.plot(x*1e6, 2.8*E  - 2.8*E[0] + 0.35*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	if ((z == 2)):
		ax2.plot(x*1e6, 1.8*E  - 1.8*E[0] + 0.4*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	if (z == 1):
		ax2.plot(x*1e6, 1.1*E  -  1.1*E[0] + 0.35*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	if (z == 0):
		ax2.plot(x*1e6, 2.5*E  -  2.5*E[0] + 0.35*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])

	#plt.savefig("EFieldXSec43THz.pdf")
ax2.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax2.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax2.set_xlim(-1.19, 1.19)
ax2.set_ylim(-0.1, 1.75)

ax2.tick_params(direction = 'in', which= 'minor', length=10, width=2)  
ax2.tick_params(direction = 'in', which= 'major', length=15, width=3, labelsize=20)  

ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_locator(MultipleLocator(0.2))

ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))

# plt.setp(ax2.spines.values(), linewidth=1)
ax2.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax2.set_ylabel(r"$\rm s-SNOM \ Signal \ (arb. units)$", fontsize = '35')
for axis in ['top','bottom','left','right']:
  ax2.spines[axis].set_linewidth(2)
Nx = 119
Ny = 207
###############################################################
print("\n \n")
sboxJim = [120, 110, 90, 50, 40]

for z in range(0, 5):
	lam = 6.0 + 0.01*sboxJim[z] # 6.0 + 0.1*sbox 
	# Field = "%s_um.txt" %freq[z]
	Fieldx = "Raw/ExHeatMap%.2f.txt" %lam
	Fieldy = "Raw/EyHeatMap%.2f.txt" %lam
	Fieldz = "Raw/EzHeatMap%.2f.txt" %lam 

	Full = np.zeros((Ny, Nx), dtype = np.double)
	ExAb = np.zeros((Ny, Nx), dtype = np.double)
	EyAb = np.zeros((Ny, Nx), dtype = np.double)
	EzAb = np.zeros((Ny, Nx), dtype = np.double)

	Y = np.linspace(0,  Ny, Ny)
	X = np.linspace(-1.19,  1.19, Nx)

	Ex = np.loadtxt(Fieldx, usecols=(0,), skiprows= 1, unpack =True )
	Ey = np.loadtxt(Fieldy, usecols=(0,), skiprows= 1, unpack =True )
	Ez = np.loadtxt(Fieldz, usecols=(0,), skiprows= 1, unpack =True )


	Ex = abs(Ex)*abs(Ex)*(dt/T)*(dt/T)
	Ey = abs(Ey)*abs(Ey)*(dt/T)*(dt/T)
	Ez = abs(Ez)*abs(Ez)*(dt/T)*(dt/T)

	for i in range (0, Ny):
		Full[i] = Ex[i*Nx: i*Nx + Nx] + Ey[i*Nx: i*Nx + Nx]+ Ez[i*Nx: i*Nx + Nx]
		Full[i] = np.sqrt(Full[i])

	print(min(1e20*Full[107]))
	print(max(1e20*Full[107]))
	if (z==2):
		ax3.plot(X, 1e20*(Full[107] - Full[107,0]) + 0.4*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])
	else:
		ax3.plot(X, 1e20*(Full[107] - Full[107,0]) + 0.35*z, linewidth=2, color = shades[z], label = r"$ \rm %s \ \mu m$" %freq[z])


ax3.axvline(x = -0.68, linestyle = "dashed", color = 'black')
ax3.axvline(x =  0.68, linestyle = "dashed", color = 'black')
ax3.set_xlim(-1.19, 1.19)
ax3.set_ylim(-0.1, 1.75)

# plt.setp(ax3.spines.values(), linewidth=1)

ax3.tick_params(direction = 'in', which= 'minor', length=10, width=2)  
ax3.tick_params(direction = 'in', which= 'major', length=15, width=3, labelsize=20)  

ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
ax3.yaxis.set_major_locator(MultipleLocator(0.2))

ax3.xaxis.set_major_locator(MultipleLocator(0.5))
ax3.xaxis.set_minor_locator(MultipleLocator(0.1))

for axis in ['top','bottom','left','right']:
  ax3.spines[axis].set_linewidth(2)

ax3.set_xlabel(r"$ \rm x \ (\mu m)$", fontsize = '35')
ax3.set_ylabel(r"$\rm \vert E \vert \ Field \ (arb. units)$", fontsize = '35')
#################################################################################3
patches = []

ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
for i in range(0, 5):
	base  = 6.0 + 0.1*sboxAli[i]
	temp = mpatches.Patch(facecolor=shades[i], label = r"$\rm \nu= %2.0f \ cm^{-1} $" %(1/(base*1e-4)), edgecolor='black')
	patches.append(temp) 
leg = ax0.legend(handles = patches, ncol = 5, loc = 'lower center', frameon = True,fancybox = False, 
fontsize = 20, bbox_to_anchor=(0.5, -0.25)) #bbox_to_anchor=(1.05, 0.5), (0.0, -0.25, 1.0, .05)
leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(2)
plt.tight_layout()
plt.savefig("Fig6PartII.png")
plt.savefig("Fig6PartII.pdf")