#############################################################
##### To be run in the same directory as hBN_VacRTA.txt #####
#############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from pylab import *
import matplotlib.patches as mpatches

import math
import cmath
from ctypes import *
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["font.family"] = 'STIXGeneral'# all units are in m^-1
pane   = ['(a)', '(b)', '(c)']

################################################

#9.79125662e13
w = linspace (100e12, 3600e12,  100)
eps_inf_Ag = 1.0
eps_s_Ag   = 343.744
tau_Ag     = 7.90279e-15
sigma_Ag   = 3.32069e6
eps0 = 8.85418782e-12
c0 = 3e8

eps1_Ag = eps_inf_Ag + (eps_s_Ag-eps_inf_Ag)/(1+1j*w*tau_Ag) + sigma_Ag/(1j*w*eps0)

Re_eps_Ag = np.real(eps1_Ag)
Im_eps_Ag = abs(np.imag(eps1_Ag))
n1_Ag = np.sqrt(eps1_Ag)

# plot(1/((c0/w0)*1e6*1e-4), Re_eps_Ag)
# show()

d = 50e-9
imp0 = 376.730313
# k0 = k*1e12

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k0 = 0
# 
# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1_Ag*d*(w/c0)#*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1_Ag = (n1_Ag)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1_Ag*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1_Ag)*sin(delta1))

Rm_Ag = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))
plot(1/((c0/w)*1e6*1e-4), Rm_Ag)
show()
################################################


wp = 1.15136316e16
gamma = 9.79125662e13
eps_inf = 4.87
s = 1.83
omega_nu = 137200
gamma = 3198.7#/(2*math.pi)#700.01702
################################################
eps_infz = 2.95
sz = 0.61
omega_nuz = 74606.285
gammaz = 491.998
################################################
d = 80e-9
imp0 = 376.730313
k00 = linspace (50000, 2500000, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*k00 - k00*k00)
eps1z = eps_infz + (sz*(omega_nuz**2))/((omega_nuz**2) + 1j*gammaz*k00 - k00*k00)


n1 = np.sqrt(eps1)
n1z = np.sqrt(eps1z)

#using the equations in chapter 2.2.r of Macleod
#assuming the impedance of free space cancels out
#assuming the incident media is vacuum with k00 = 0

# unlabled equation on p 38 in Macleod after eqn 2.88 
delta1 = n1*d*k00*2*math.pi
delta1z = n1z*d*k00*2*math.pi


# eqn 2.93 in Macleod
#since we behin at normal incidence eta0 = y0
eta0 = imp0
eta1 = (n1)*imp0
eta2 = imp0
Y =  (eta2*cos(delta1) + 1j*eta1*sin(delta1))/(cos(delta1) + 1j*(eta2/eta1)*sin(delta1))

Rm = abs(((eta0 - Y)/(eta0 + Y))*conj((eta0 - Y)/(eta0 + Y)))

#########################################################
# eta0z = imp0
# eta1z = (n1z)*imp0
# eta2z = imp0
# Yz =  (eta2z*cos(delta1z) + 1j*eta1z*sin(delta1z))/(cos(delta1z) + 1j*(eta2z/eta1z)*sin(delta1z))

# Rmz = abs(((eta0z - Yz)/(eta0z + Yz))*conj((eta0z - Yz)/(eta0z + Yz)))
#########################################################

#Calculating the T

# Calculate (Power) Transmision from the result of problem 5.11 
# from: http://eceweb1.rutgers.edu/~orfanidi/ewa/ch05.pdf
# Note j --> -i Convention in formula below

B = cos(delta1) + 1j*(eta2/eta1)*sin(delta1)
C = eta2*cos(delta1) + 1j*eta1*sin(delta1)
Tm = 4*eta0*real(eta2)/((eta0*B + C)*conj(eta0*B + C))
#################################################################
mpl.rcParams['agg.path.chunksize'] = 10000
# all units are in THz

eps_inf = 4.87
s = 1.83
omega_nu = 41.14e12
gamma = 0.95895e12
w0 = linspace (15e12, 300e12, 20000)
c0 = 3e8
eps1 = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w0 - w0*w0)

Re_eps = np.real(eps1)
Im_eps = abs(np.imag(eps1))
freq    = 1/((c0/w0)*1e6*1e-4)

lam_res = 7.29e-6
w_res   = c0/lam_res
eps_res = np.real(eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w_res - w_res*w_res))
#################################################################

eps_inf = 2.95
s = 0.61
omega_nu = 22.36640e12
gamma = 0.06045e12

eps1z = eps_inf + (s*(omega_nu**2))/((omega_nu**2) + 1j*gamma*w0 - w0*w0)

Re_epsz = np.real(eps1z)
Im_epsz = abs(np.imag(eps1z))

###############################################################################

for i in range(0,len(Re_eps)-1):
    if ((Im_eps[i] > Im_eps[i-1]) & (Im_eps[i] > Im_eps[i+1]) ):
    	peak  = freq[i]
    	peakA = Im_eps[i]
###############################################################################
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize = (18,30), sharex = True)
fig.subplots_adjust(hspace=0)
###############################################################################
plt.setp(ax1.spines.values(), linewidth=4)
ax1.tick_params(direction = 'in', which= 'minor', length=10, width=2)  
ax1.tick_params(direction = 'in', which= 'major', length=15, width=3, labelsize=45) 
ax1.xaxis.set_major_locator(MultipleLocator(100))
ax1.xaxis.set_minor_locator(MultipleLocator(50))

ax1.text(.07,.9,'%s ' %pane[0],
    horizontalalignment='center',
    transform=ax1.transAxes, fontsize = '60', fontweight = 'bold')

ax1.plot(freq, Re_eps, label  = r'$\rm  Re \ \mathit{\epsilon_{x,y}}$', color = "black", linewidth = 4)
ax1.plot(freq, Re_epsz, label = r'$\rm  Re \ \mathit{\epsilon_{z}}$', color = "blue", linewidth = 4)
# ax1.plot(freq, Re_eps_Ag, label  = r'$\rm  Re \ \mathit{\epsilon_{Ag}}$', color = "silver", linewidth = 14)

ax1.axvline(x = peak, color = 'black', linewidth = 3)
# ax1.set_xlim(1000,1800)
# ax1.set_ylim(-50,50)
# ax1.set_ylabel(r'$\rm Re \ \mathit{\epsilon}$', fontsize = '85')   )
ax1.set_ylabel(r'$\rm Re \ \mathit{\epsilon}$', fontsize = '85', labelpad = -30) 

ax1.legend(loc='upper right', fontsize='45')
print(peak)
########################################################################################
plt.setp(ax2.spines.values(), linewidth=4)
ax2.tick_params(direction = 'in', which= 'minor', length=10, width=2)  
ax2.tick_params(direction = 'in', which= 'major', length=15, width=3, labelsize=45)
ax2.xaxis.set_major_locator(MultipleLocator(100))
ax2.xaxis.set_minor_locator(MultipleLocator(50))

ax2.text(.07,.9,'%s ' %pane[1],
    horizontalalignment='center',
    transform=ax2.transAxes, fontsize = '60', fontweight = 'bold')

ax2.plot(freq, Im_eps, label  = r'$\rm Im \ \mathit{\epsilon_{x,y}}$', color = "black", linewidth = 4)
ax2.plot(freq, Im_epsz, label = r'$\rm Im \ \mathit{\epsilon_{z}}$', color = "blue", linewidth = 4)
# ax2.plot(freq, Im_eps_Ag, label  = r'$\rm  Im \ \mathit{\epsilon_{Ag}}$', color = "silver", linewidth = 4)

ax2.legend(loc='upper right', fontsize='45')
ax2.set_ylabel(r'$\rm Im \ \mathit{\epsilon}$', fontsize = '85') 
ax2.axvline(x = peak, color = 'black', linewidth = 3)
# ax2.set_xlim(1000,1800)
# ax2.set_ylim(-10,90)

c0 = 3e8
ddx = 1e-9
dt = ddx/(2*c0)

########################################################################################

namy = "hBN_VacRTA.txt" 
lam, R, T, A = loadtxt(namy, usecols=(0,1,2,3), skiprows= 1, unpack =True)

ax3.plot(k00/100, Rm, label=r'$ R_{TM}$', color='darkred', linewidth = 12)
ax3.plot(k00/100, Tm, label=r'$ T_{TM}$', color='navy', linewidth = 12)
ax3.plot(k00/100, (1-(Rm + Tm)), label=r'$ A_{TM}$', color='darkgreen', linewidth = 12)

# ax3.plot((c0*2*math.pi/(w))*1e6, Rm_Ag, label=r'$ R_{Ag}$', color='silver', linewidth = 12)

ax3.text(.07,.9,'%s ' %pane[2],
    horizontalalignment='center',
    transform=ax3.transAxes, fontsize = '60', fontweight = 'bold')

ax3.plot(1/(lam*1e-4), R, label = r'$R_{FDTD}$', color = "red", linewidth = 4)
ax3.plot(1/(lam*1e-4), T, label = r'$ T_{FDTD}$', color = "cyan", linewidth = 4)
ax3.plot(1/(lam*1e-4), A, label = r'$ A_{FDTD}$', color = "limegreen", linewidth = 4)


plt.setp(ax3.spines.values(), linewidth=4)
ax3.tick_params(direction = 'in', which= 'minor', length=10, width=2)  
ax3.tick_params(direction = 'in', which= 'major', length=15, width=3, labelsize=45)
ax3.tick_params('x', pad = 10)

ax3.xaxis.set_major_locator(MultipleLocator(100))
ax3.xaxis.set_minor_locator(MultipleLocator(50))
# ax3.yaxis.set_major_locator(MaxNLocator(prune='lower'))

# plt.yaxis.set_label_position("right")
ax3.set_ylabel(r"$R,T,A$", fontsize = '85') 
ax3.set_xlabel(r"$\rm Frequency\ (cm^{-1})$", fontsize = '85')

# plt.set_yticks(np.arange(0.1, 0.6, 0.1))
ax3.yaxis.set_major_locator(MaxNLocator(nbins = 6, prune='both'))

ax3.legend(loc='center right', fontsize='45')
ax3.axvline(x = peak, color = 'black',linewidth = 3)
ax3.set_xlim(1000,1800)
ax3.set_ylim(0,1.1)

###############################################################################
# patches = []
# PitchLeng = np.linspace(2.02, 2.78, 20)
# dcc = np.zeros(20, dtype = np.double)
# dcc = PitchLeng

# for i in range(0, 20):
#     temp = mpatches.Patch(facecolor=ShadesRed[i], label = r'$\rm d_{cc} = %2.2f \ \mu m$' %dcc[i], edgecolor='black')
#     patches.append(temp) 
# leg = ax2.legend(handles = patches, ncol = 4, loc = 'upper center', frameon = True,fancybox = False, fontsize = 25, bbox_to_anchor=(0., 1.35, 1., .102),mode="expand", borderaxespad=0.) #bbox_to_anchor=(1.05, 0.5),
# leg.get_frame().set_edgecolor('black')

# leg.get_frame().set_linewidth(4)
###############################################################################  

plt.savefig("Fig2.png")
plt.savefig("Fig2.pdf")

# Peak.insert(i, min(RT[0:300]))