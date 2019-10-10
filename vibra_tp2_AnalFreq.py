import os

import numpy as np
import matplotlib.pyplot as plt

import peakutils



projectPath = "/Users/maximeglomot/Cours/L3_2020/Vibra/tp2_Syst2DDL/"

dataFRF_1mean = np.loadtxt(projectPath + "tp2frequencial_1mean.txt", skiprow = 1)

dataFRF_5mean = np.loadtxt(projectPath + "tp2frequencial_5mean.txt", skiprow = 1)


# FRF
frequency = dataFRF_1mean[0] # 0 à 2000Hz
accelerance_inout = dataFRF_5mean[1]

#%%
