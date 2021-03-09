##############################################################
##### To be run from /scratch/dermoth/AgPaper/FullDevice #####
##############################################################

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

# font = FontProperties()
# font.set_family('serif')
# font.set_name('Times New Roman')
# cols = ["black", 'red', 'limegreen', 'blue', 'cyan']
cols = ['deeppink','fuchsia', 'purple', 'darkviolet', 'blue', 'dodgerblue', 'deepskyblue', 'teal', 'springgreen', 'seagreen', 'limegreen',
 'forestgreen', 'greenyellow','gold', 'orange', 'orangered', 'salmon', 'red', 'darkred', 'lightcoral']

mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["font.family"] = 'STIXGeneral'

#all units are in m^-1
##############################################################################
fig, axs = plt.subplots(3, figsize=(18,33), sharex = True)
fig.subplots_adjust(hspace=0)
plt.gcf().subplots_adjust(bottom=0.24, top = 0.99)

# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax5 = fig.add_subplot(111, frameon=False)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_ylabel('Absorptance', rotation='vertical', fontsize = '85', labelpad = 65)

##############################################################################
model  = ['3Term', 'AnisoDead', 'FullDevice']
device = ['Bare', 'Uncoupled', 'Coupled']
pane   = ['(a)', '(b)', '(c)']
# dirk  = ['1.74um', '1.94um','2.14um','2.34um', '2.54um', '2.74um']#, '2.94um', '3.14um']#, '3.35um','3.55um', '3.75um', '3.95um']
# dirks = ['1.74', '1.94','2.14','2.34', '2.54', '2.74']#, '2.94', '3.14']# '3.35','3.55', '3.75', '3.95']
# spec =  ["R", "T", "A"]
##############################################################################
###########################################################################

###########################################################################

###########################################################################
i = 0
j = 0
q = 0
# X = np.arange(40, 80, 1)
Lfreq = 1000
Hfreq = 1800
X = np.arange(Lfreq, Hfreq, 100)
Y = np.arange(0, 1, 0.05)
# peakx = np.zeros((11,10), dtype = np.double)
# peaky = np.zeros((11,10), dtype = np.double)
# print(Y)
for ax0 in axs.flat:
	ax0.text(.7,.9,'%s Device' %device[q],
	        horizontalalignment='center',
	        transform=ax0.transAxes, fontsize = '60')
	ax0.text(.05,.9,'%s ' %pane[q],
	        horizontalalignment='center',
	        transform=ax0.transAxes, fontsize = '60', fontweight = 'bold')
	for i in range(0, 20):
		
		base = 2 + 4*i
		namy = "../%s/2_%.02dum/Ref/2_%.02dumPitchRTA.txt" %(model[q], base, base)
		lam, A = loadtxt(namy, usecols=(0,3), skiprows= 1, unpack =True)
		print(namy)
		plt.setp(ax0.spines.values(), linewidth=3)
		# plt.rcParams['font.family'] = 'serif'
		# plt.rcParams['font.serif'] = ['Times'] + plt.rcParams['font.serif']


		ax0.set_xlim(Lfreq, Hfreq)

		ax0.plot((1/(lam*1e-4)), A, color = cols[i], linewidth = 4)

		ax0.set_ylim(0,0.3)
		Y = np.arange(0,0.3, 0.05)
		# ax0.plot(1/(lam*1e-4), Tra, label = r'$\rm T_{%s}$' %dirk[q], color = "blue", linewidth = 3)
		# ax0.plot(1/(lam*1e-4), Abs, label = r'$\rm A_{%s}$' %dirk[q], color = "limegreen", linewidth = 3)

		ax0.tick_params(direction = 'in', which= 'minor', length=10, width=2)  
		ax0.tick_params(direction = 'in', which= 'major', length=15, width=3, labelsize=45)  
		ax0.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
		ax0.xaxis.set_major_locator(MultipleLocator(100))
		ax0.xaxis.set_minor_locator(MultipleLocator(50))
		
		ax0.yaxis.set_major_locator(MaxNLocator(nbins = 3,prune='lower'))
		ax0.yaxis.set_minor_locator(MultipleLocator(0.05))
		# ax0.set_xticks(X ,minor=True)
		ax0.set_yticks(Y ,minor=True) 
		ax0.axvline(x =1/(7.26*1e-4), color = 'black', linewidth = 2)
		# ax0.axvline(x =1370, ls = '--', color = 'black', linewidth = 2)
		# ax0.axvline(x =1610, ls = '--', color = 'black', linewidth = 2)
		# ax0.yaxis.set_major_locator(MaxNLocator(nbins = 5,prune='lower'))
		# ax0.legend(loc='upper right', fontsize='12', ncol = 4)
	q = q + 1
ax0.set_xlabel(r"$\rm Frequency\ (cm^{-1})$", fontsize = '85')

###############################################################################
patches = []
PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(20, dtype = np.double)
dcc = PitchLeng
for i in range(0, len(dcc)):
    temp = mpatches.Patch(facecolor=cols[i], label = r'$\mathrm{ d_{cc} = %2.2f \ \mu m}$' %dcc[i], edgecolor='black')
    patches.append(temp) 
leg = ax0.legend(handles = patches, ncol = 3, loc = 'lower center', frameon = True,fancybox = False, 
fontsize = 40, bbox_to_anchor=(-0.1, -0.95, 1.2, .175),mode="expand", borderaxespad=0.) #bbox_to_anchor=(1.05, 0.5),
leg.get_frame().set_edgecolor('black')

leg.get_frame().set_linewidth(3)
###########################################################################
###########################################################################
# cax = ax0.inset_axes([-0.1, -0.25, 1.2, .175])#-0.20,0.9, 0.15, 0.15
# cax.set_xticks([])
# cax.set_yticks([])
# Nx = 107
# Ny = 107
# ###########################################################################
# Field = "2.14um/Media/XY.dat"
# Full = np.zeros((107, 107), dtype = np.double)

# X = np.linspace(0,  2.14, Nx)
# Y = np.linspace(0,  2.14, Ny)


# E = np.loadtxt(Field, usecols=(0,), skiprows= 0, unpack =True )
# print(max(E))
# for i in range (0, Ny):
# 	for j in range (0, Nx):
#  		Full[i][j] = E[i*Nx + j]

# print(max(E))

# cols = ['tan', 'violet', 'black','gold', 'gold', 'gold', 'lightblue', 'darkgray']
# cMap = c.ListedColormap(cols)
# norm = mpl.colors.Normalize(vmin=0, vmax=len(cols))
# pc_kwargs = {'rasterized': True}

# cax.set_aspect('equal')

# im = cax.pcolormesh(X, Y, Full, norm=norm, cmap = cMap, **pc_kwargs)

###############################################################################


plt.savefig("Fig3.png" )

