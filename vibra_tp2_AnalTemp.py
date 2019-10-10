import numpy as np
import matplotlib.pyplot as plt

import peakutils



projectPath = "/Users/maximeglomot/Cours/L3_2020/Vibra/tp2_Syst2DDL/"
data = np.loadtxt(projectPath + "tp2temporal.txt", skiprow = 1)

timeAxis = data[0]
accelerance = data[1]



#%%
timeAxis = np.arange(0,100,0.1)
accelerance = np.exp(-0.1*timeAxis)*np.sin(timeAxis)
indexes = peakutils.indexes(accelerance, thres=0.5, min_dist=30)
plt.figure(figsize=(10,5))
plt.plot(timeAxis, accelerance)
plt.plot(timeAxis[indexes],accelerance[indexes],'.')
plt.plot([ timeAxis[indexes[0]] ,timeAxis[indexes[1]]],[0,0],'r', label = " Psoeudo-période $T_{d}$")

a=np.arange( timeAxis[indexes[0]] , timeAxis[indexes[1]]+1, 0.1)
array = np.argmax(accelerance)*np.exp(- delta * a)

plt.plot(a,array)
plt.plot([ timeAxis[indexes[0]] ,timeAxis[indexes[1]]],[0,0],'r', label = " Psoeudo-période $T_{d}$")

delta = ( np.log(accelerance[indexes[0]]) / np.log(accelerance[indexes[1]]) )
xi =   #( timeAxis[indexes[0]] -  timeAxis[indexes[1]])
plt.title(" First Estimate of Max ")

plt.grid(True)
plt.show()
#%%
m=2e-3
#Td=
#omegaD= 2 * np.pi / Td
omega0=

xi=
#%%
# firstmax_arg = np.argmax(accelerance)
# firstmin_arg = np.argmin(accelerance)
# halfperiode = np.abs(firstmax_arg - firstmin_arg)
# i=0
# while np.argmax(accelerance[0:firstmax_arg+1]) == np.argmax(accelerance[0:firstmax_arg+i]):
#     i+=1
# secondmax = i
