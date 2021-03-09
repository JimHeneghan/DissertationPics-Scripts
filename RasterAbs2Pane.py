#############################################################
##### To be run from /scratch/dermoth/AgPaper/AnisoDead #####
#############################################################
import numpy as np
from scipy import *
from pylab import *
import math
import cmath
from ctypes import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import UnivariateSpline
import matplotlib.gridspec as gridspec

mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams["mathtext.fontset"] = 'stix'
plt.rcParams["font.family"] = 'STIXGeneral'
pane   = ['(a)', '(b)', '(c)']
#all units are in cm^-1

# res = [6.598003, 6.724, 6.885993, 7.029992, 7.173996, 7.299998, 7.426, 7.570004, 7.677995, 7.821992]
# res = [5.94,  6.21,6.35, 6.48, 6.62, 6.75, 6.89, 7.03, 7.16, 7.30, 7.43, 7.57, 7.71, 7.84,  8.11]

AgRef    = [1650.7100777 , 1612.3833622 , 1584.78655619, 1549.42671212,
         1515.61010203, 1487.20999405, 1459.85337525, 1433.48664951,
         1404.49339572, 1376.653309  , 1356.48325747, 1333.69005097,
         1308.55659926, 1293.32660286, 1269.68004063, 1249.68664107,
         1227.59666771, 1214.18001991, 1193.31671043, 1175.64003313,  1155.]
resR = np.asarray(AgRef)
AgTra    = [1670.56325546, 1636.12672521, 1607.71678333, 1575.79677012,
       1545.11671271, 1519.75660794, 1495.21665241, 1467.56677429,
       1440.92343596, 1415.22664996, 1393.92327512, 1369.863389  ,
       1346.61998384, 1330.49335093, 1308.55659926, 1287.33330965,
       1266.78329518, 1249.68664107, 1230.31329559, 1214.18001991, 1214.0]
res = np.asarray(AgTra)
AgAbs    = [1670.56325546, 1631.32003971, 1607.71678333, 1575.79677012,
       1545.11671271, 1511.48661813, 1483.24005484, 1448.43673844,
       1418.84343261, 1390.43342869, 1366.49339075, 1343.36342199,
       1317.8699742 , 1299.37663705, 1275.51004139, 1255.33659277,
       1233.04668697, 1216.84004406, 1198.46668173, 1180.63670793,  1155.64003313]
resA = np.asarray(AgAbs)

DrDDots  = [1448.65803175, 1427.93418979, 1418.10712938, 1390.36337344,
        1366.98867418, 1364.99736296, 1366.75354535]

DeadRRes = [1640.95670401, 1603.07662466, 1571.34001519, 1536.56990271,
       1507.38664679, 1475.35993988, 1452.22337577, 1422.47672544,
       1393.92327512, 1369.863389  , 1349.89328419, 1327.31669524,
       1302.42335401, 1284.35670233, 1263.90341149, 1244.08995069,
       1222.19670547, 1208.89660863, 1190.75998827, 1170.68670726]

DeadTRes = [1670.56325546, 1636.12672521, 1607.71678333, 1575.79677012,
       1545.11671271, 1519.75660794, 1495.21665241, 1467.56677429,
       1440.92343596, 1415.22664996, 1393.92327512, 1369.863389  ,
       1346.61998384, 1330.49335093, 1308.55659926, 1287.33330965,
       1266.78329518, 1249.68664107, 1230.31329559, 1214.18001991]

DeadARes = [1670.56325546, 1631.32003971, 1607.71678333, 1575.79677012,
       1545.11671271, 1499.24992526, 1467.56677429, 1437.19335683,
       1408.05327623, 1380.07328741, 1356.48325747, 1333.69005097,
       1308.55659926, 1290.32341311, 1269.68004063, 1246.88325943,
       1227.59666771, 1211.53336145, 1193.31671043, 1175.64003313]

PolRes  = [1471.45336332, 1471.45336332, 1467.56677429, 1467.56677429,
       1467.56677429,    0.        , 1491.20324295, 1444.67000559,
       1426.12664005, 1463.70001995, 1467.56677429, 1467.56677429,
       1463.70001995, 1463.70001995, 1463.70001995, 1463.70001995,
       1463.70001995, 1463.70001995, 1463.70001995, 1463.70001995]

##############################################################################
PitchLeng = np.linspace(2.02, 2.78, 20)
# AgAbs = PitchLeng
##############################################################################
##############################################################################

c0       = 3e8
nref     = 1.0
ntra     = 3.42
NFREQs   = 500
numPlots = 20
FullDev  = []#np.zeros((NFREQs,10), dtype = np.double)
FullDead = []
###########################################################################
lam        = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(0), skiprows= 1, unpack =True)
lam = np.append(lam, 10.018)
###########################  Pitch = **** um ##############################
AFD2_02um  = np.loadtxt("../FullDevice/2_02um/Ref/2_02umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_02um    = np.loadtxt("2_02um/Ref/2_02umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_02um)
###########################  Pitch = **** um ##############################
AFD2_06um  = np.loadtxt("../FullDevice/2_06um/Ref/2_06umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_06um    = np.loadtxt("2_06um/Ref/2_06umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_06um)
###########################  Pitch = **** um ##############################
AFD2_10um  = np.loadtxt("../FullDevice/2_10um/Ref/2_10umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_10um    = np.loadtxt("2_10um/Ref/2_10umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_10um)

###########################  Pitch = **** um ##############################
AFD2_14um  = np.loadtxt("../FullDevice/2_14um/Ref/2_14umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_14um    = np.loadtxt("2_14um/Ref/2_14umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_14um)

###########################  Pitch = **** um ##############################
AFD2_18um  = np.loadtxt("../FullDevice/2_18um/Ref/2_18umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_18um    = np.loadtxt("2_18um/Ref/2_18umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_18um)

###########################  Pitch = **** um ##############################
AFD2_22um  = np.loadtxt("../FullDevice/2_22um/Ref/2_22umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_22um    = np.loadtxt("2_22um/Ref/2_22umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_22um)
###########################  Pitch = **** um ##############################
AFD2_26um  = np.loadtxt("../FullDevice/2_26um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_26um    = np.loadtxt("2_26um/Ref/2_26umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# Full[,1]    = AFD2_26um
FullDev.append(AFD2_26um)

###########################################################################
AFD2_30um  = np.loadtxt("../FullDevice/2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_30um    = np.loadtxt("2_30um/Ref/2_30umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
# Full[,2]    = AFD2_30um
FullDev.append(AFD2_30um)

###########################################################################
AFD2_34um  = np.loadtxt("../FullDevice/2_34um/Ref/2_34umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_34um    = np.loadtxt("2_34um/Ref/2_34umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_34um)

# Full[,3]    = AFD2_34um
###########################################################################
AFD2_38um = np.loadtxt("../FullDevice/2_38um/Ref/2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_38um   = np.loadtxt("2_38um/Ref/2_38umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_38um)

# Full[4]    = AFD2_38um
###########################################################################
AFD2_42um = np.loadtxt("../FullDevice/2_42um/Ref/2_42umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_42um   = np.loadtxt("2_42um/Ref/2_42umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_42um)

# Full[5]    = AFD2_42um
###########################################################################
AFD2_46um = np.loadtxt("../FullDevice/2_46um/Ref/2_46umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_46um   = np.loadtxt("2_46um/Ref/2_46umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_46um)
# Full[6]    = AFD2_46um
###########################################################################
AFD2_50um = np.loadtxt("../FullDevice/2_50um/Ref/2_50umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_50um   = np.loadtxt("2_50um/Ref/2_50umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_50um)
# Full[7]    = AFD2_50um
###########################################################################
AFD2_54um = np.loadtxt("../FullDevice/2_54um/Ref/2_54umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
A2_54um   = np.loadtxt("2_54um/Ref/2_54umPitchRTA.txt",  usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_54um)
# Full[8]    = AFD2_54um
###########################################################################
AFD2_58um = np.loadtxt("../FullDevice/2_58um/Ref/2_58umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
A2_58um   = np.loadtxt("2_58um/Ref/2_58umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_58um)
###########################################################################
AFD2_62um = np.loadtxt("../FullDevice/2_62um/Ref/2_62umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
A2_62um   = np.loadtxt("2_62um/Ref/2_62umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_62um)
###########################################################################
AFD2_66um = np.loadtxt("../FullDevice/2_66um/Ref/2_66umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
A2_66um   = np.loadtxt("2_66um/Ref/2_66umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_66um)
###########################################################################
AFD2_70um = np.loadtxt("../FullDevice/2_70um/Ref/2_70umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
A2_70um   = np.loadtxt("2_70um/Ref/2_70umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_70um)
###########################################################################
AFD2_74um = np.loadtxt("../FullDevice/2_74um/Ref/2_74umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
A2_74um   = np.loadtxt("2_74um/Ref/2_74umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_74um)
###########################################################################
AFD2_78um = np.loadtxt("../FullDevice/2_78um/Ref/2_78umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
A2_78um   = np.loadtxt("2_78um/Ref/2_78umPitchRTA.txt",    usecols=(3), skiprows= 1, unpack =True)
FullDev.append(AFD2_78um)

FullDead.append(A2_02um)

FullDead.append(A2_06um)


FullDead.append(A2_10um)
FullDead.append(A2_14um)

FullDead.append(A2_18um)

FullDead.append(A2_22um)

FullDead.append(A2_26um)

FullDead.append(A2_30um)


FullDead.append(A2_34um)

FullDead.append(A2_38um)


FullDead.append(A2_42um)

FullDead.append(A2_46um)
# Full[6]    = A2_46um

FullDead.append(A2_50um)
# Full[7]    = A2_50um

FullDead.append(A2_54um)
# Full[8]    = A2_54um

FullDead.append(A2_58um)

FullDead.append(A2_62um)

FullDead.append(A2_66um)

FullDead.append(A2_70um)

FullDead.append(A2_74um)
FullDead.append(A2_78um)

pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)

pitchThou[0]   = A2_02um
pitchThou[1]   = A2_06um
pitchThou[2]   = A2_10um
pitchThou[3]   = A2_14um
pitchThou[4]   = A2_18um
pitchThou[5]   = A2_22um
pitchThou[6]   = A2_26um
pitchThou[7]   = A2_30um
pitchThou[8]   = A2_34um
pitchThou[9]   = A2_38um
pitchThou[10]  = A2_42um
pitchThou[11]  = A2_46um
pitchThou[12]  = A2_50um
pitchThou[13]  = A2_54um
pitchThou[14]  = A2_58um
pitchThou[15]  = A2_62um
pitchThou[16]  = A2_66um
pitchThou[17]  = A2_70um
pitchThou[18]  = A2_74um
pitchThou[19]  = A2_78um
peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)

peak2  = np.zeros(numPlots, dtype = np.double)
peakA2 = np.zeros(numPlots, dtype = np.double)
###############################################################################
for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & ((1/(lam[j]*1e-4) < 2000)) & ((1/(lam[j]*1e-4) > 1000))):
            peak[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA[i] = pitchThou[i,j]


for i in range(0, numPlots):
    temp = 0.0
    for j in range(250,NFREQs-1):
        if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & (temp < pitchThou[i,j]) & ((1/(lam[j]*1e-4) > peak[i]))):
            temp = pitchThou[i,j]
            peak2[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA2[i] = pitchThou[i,j]


###############################################################################



dcc = np.zeros(numPlots, dtype = np.double)
dcc = PitchLeng
dcc1 = dcc
###############################################################################
FullDead = np.transpose(FullDead)


###############################################################################

X = 1/(lam*1e-4)
Y = resA
print(len(res))
print(len(X))
print(len(FullDead))

minni = 1/(10.0*1e-4)
maxi  = 1/(4.5*1e-4)
dcc = 1/(dcc*1e-4)
# peak = 1/(peak*1e-4)
# peak2 = 1/(peak2*1e-4)
# Full = np.reshape(Full, (10,NFREQs), order='C')

fig, (ax0, ax1) = plt.subplots(1,2, figsize = (24,12), sharey=True)#,constrained_layout=True)
# fig.subplots_adjust( wspace= None, hspace = None)
gs1 = gridspec.GridSpec(24, 12)
gs1.update(wspace=0.025)

# plt.tight_layout()
plt.setp(ax0.spines.values(), linewidth=2)
ax0.tick_params(direction = 'in', width=2, labelsize=30)
ax0.set_ylabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '60')   
ax0.set_xlabel(r"$\rm \nu_{Bare}  \ (cm^{-1})$", fontsize = '60')
ax0.set_xlim(min(Y),max(Y))
print(min(Y),max(Y))
ax0.set_ylim(1100,1700)
ax0.set_title(r"$\rm Uncoupled \ Device  $", fontsize = '50', pad = -45, color = 'white')
ax0.text(.07,.95,'%s ' %pane[0],
    horizontalalignment='center', color = 'white',
    transform=ax0.transAxes, fontsize = '40', fontweight = 'bold')
norm = mpl.colors.Normalize(vmin=0.00, vmax=0.30)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

im = ax0.pcolormesh(Y, X, FullDead, norm=norm, **pc_kwargs)
ax0.set_aspect('equal')

# plt.scatter(AgAbs[0:numPlots], DeadARes, linewidth = 2, s=150,edgecolors = 'white', c='limegreen', label = "Dead Resonance")       

# plt.scatter(AgAbs[0:numPlots], peak2,  linewidth = 2, s=55,edgecolors = 'white', c='blue', label = "Polariton")       
ax0.scatter(AgAbs[0:numPlots], peak, linewidth = 5, s=150,edgecolors = 'black', c='purple', label = "Plasmon")       

ax0.scatter(AgAbs[0:numPlots], peak, linewidth = 2, s=125,edgecolors = 'white', c='purple', label = "Plasmon")       

temper = peak[10]

ax0.tick_params(left = False, bottom = False)

peak  = np.zeros(numPlots, dtype = np.double)
peakA = np.zeros(numPlots, dtype = np.double)

peak2  = np.zeros(numPlots, dtype = np.double)
peakA2 = np.zeros(numPlots, dtype = np.double)
###########################################################################
pitchThou = np.zeros((numPlots,NFREQs), dtype = np.double)

pitchThou[0]   = AFD2_02um
pitchThou[1]   = AFD2_06um
pitchThou[2]   = AFD2_10um
pitchThou[3]   = AFD2_14um
pitchThou[4]   = AFD2_18um
pitchThou[5]   = AFD2_22um
pitchThou[6]   = AFD2_26um
pitchThou[7]   = AFD2_30um
pitchThou[8]   = AFD2_34um
pitchThou[9]   = AFD2_38um
pitchThou[10]  = AFD2_42um
pitchThou[11]  = AFD2_46um
pitchThou[12]  = AFD2_50um
pitchThou[13]  = AFD2_54um
pitchThou[14]  = AFD2_58um
pitchThou[15]  = AFD2_62um
pitchThou[16]  = AFD2_66um
pitchThou[17]  = AFD2_70um
pitchThou[18]  = AFD2_74um
pitchThou[19]  = AFD2_78um

###############################################################################
for i in range(0, numPlots):
    for j in range(0,NFREQs-1):
        if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & ((1/(lam[j]*1e-4) < 1400)) & ((1/(lam[j]*1e-4) > 1000))):
            peak[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA[i] = pitchThou[i,j]


for i in range(0, numPlots):
    temp = 0.0
    for j in range(250,NFREQs-1):
        if ((pitchThou[i,j] > pitchThou[i,j-1]) & (pitchThou[i,j+1] < pitchThou[i,j]) & (temp < pitchThou[i,j]) & ((1/(lam[j]*1e-4) > peak[i]))):
            temp = pitchThou[i,j]
            peak2[i]  = (1/(lam[j]*1e-4))
            # print(j)
            peakA2[i] = pitchThou[i,j]
for i in range(10, 13):
    y_spl = UnivariateSpline(lam[334:355],pitchThou[i,334:355],s=0,k=5)
    # plt.plot(lam[334:360],pitchThou[i,334:360], 'ro')
    x_range = np.linspace(lam[334],lam[355],1000)
    # plt.semilogy(x_range,y_spl(x_range))
    y_spl_2d = y_spl.derivative(n=2)

    plt.plot(x_range,y_spl_2d(x_range))
    # print(y_spl_2d.roots())
    # print(y_spl_2d(x_range))
    # rootys = y_spl_2d.roots()
    # X0s = np.zeros(len(rootys), dtype = np.double)
    # plt.semilogy(rootys,y_spl(rootys), 'bo')
    y_spl = y_spl(x_range)

    y_spl_2d = y_spl_2d(x_range)
    temp1 = 0.0
    for j in range(0,998):
      if((y_spl_2d[j-1] > y_spl_2d[j]) & (y_spl_2d[j] < y_spl_2d[j+1]) & (y_spl_2d[j] < temp1)):
        peak2[i]  = (1/(x_range[j]*1e-4))
        peakA2[i] = y_spl[j] 
        temp1 = y_spl_2d[j]
        temp2 = y_spl_2d[j]
###############################################################################

PitchLeng = np.linspace(2.02, 2.78, 20)
dcc = np.zeros(numPlots, dtype = np.double)
dcc = PitchLeng
dcc1 = dcc
###############################################################################
FullDev = np.transpose(FullDev)


###############################################################################

X = 1/(lam*1e-4)
Y = resA
print(len(res))
print(len(X))
print(len(FullDev))

minni = 1/(10.0*1e-4)
maxi  = 1/(4.5*1e-4)
dcc = 1/(dcc*1e-4)
# peak = 1/(peak*1e-4)
# peak2 = 1/(peak2*1e-4)
# Full = np.reshape(Full, (10,NFREQs), order='C')

plt.setp(ax1.spines.values(), linewidth=2)

ax1.tick_params(direction = 'in', width=2, labelsize=30)
# ax1.set_ylabel(r"$\rm Frequency \ (cm^{-1})$", fontsize = '30')   
ax1.set_xlabel(r"$\rm \nu_{Bare}  \ (cm^{-1})$", fontsize = '60')
ax1.set_xlim(min(Y),max(Y))
ax1.set_ylim(1100,1700)
ax1.set_title(r"$\rm Coupled \ Device $", fontsize = '50', pad = -45, color = 'white')
ax1.text(.07,.95,'%s ' %pane[1],
    horizontalalignment='center', color = 'white',
    transform=ax1.transAxes, fontsize = '40', fontweight = 'bold')

norm = mpl.colors.Normalize(vmin=0.00, vmax=0.30)
pc_kwargs = {'rasterized': True, 'cmap': 'jet'}

im = ax1.pcolormesh(Y, X, FullDev, norm=norm, **pc_kwargs)
ax1.set_aspect('equal')
# plt.scatter(AgAbs[0:numPlots], DeadARes, linewidth = 2, s=150,edgecolors = 'white', c='limegreen', label = "Dead Resonance")       
funDots =  np.zeros(7, dtype = np.double)
print(AgAbs[0:numPlots])
for i in range (0,7):
  funDots[i] = (peak2[i+7] + DrDDots[i])/2

# ax1.scatter(AgAbs[0:numPlots], PolRes,  linewidth = 5, s=150,edgecolors = 'black', c='green')       

# ax1.scatter(AgAbs[0:numPlots], PolRes,  linewidth = 2, s=125,edgecolors = 'white', c='green')  

ax1.scatter(AgAbs[0:numPlots], peak2,  linewidth = 5, s=150,edgecolors = 'black', c='blue')       

ax1.scatter(AgAbs[0:numPlots], peak2,  linewidth = 2, s=125,edgecolors = 'white', c='blue')       

ax1.scatter(AgAbs[0:numPlots], peak,  linewidth = 5, s=150,edgecolors = 'black', c='blue')       

ax1.scatter(AgAbs[0:numPlots], peak,  linewidth = 2, s=125,edgecolors = 'white', c='red')  



ax1.tick_params(left = False, bottom = False)


ax0 = fig.add_subplot(111, frameon=False)
ax0.set_xticks([])
ax0.set_yticks([])
cbaxes = fig.add_axes([0.88, 0.12, 0.04, 0.75]) 

cbar = fig.colorbar(im, ax = ax0, cax = cbaxes)
# cbar = plt.colorbar(im, fraction=0.046, pad = 0.04)

cbar.set_label(label = r"$\rm Absorptance$", size = '60')
cbar.ax.tick_params(width=2,labelsize=30) 
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig("Fig4.pdf")
plt.savefig("Fig4.png")






  # plt.savefig("AbsDeadRasterHM.pdf")
  # plt.savefig("AbsDeadRasterHM.png")


