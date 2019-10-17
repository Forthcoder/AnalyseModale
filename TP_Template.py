import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
from scipy.interpolate import interp2d

parentDirPath = "/Users/maximeglomot/Cours/L3_2020/Vibra/tp3_meuleuse"

for folder in os.listdir(parentDirPath):
    if os.path.isdir(parentDirPath + "/" + folder) == True:
        print ("yeah")

os.path.isdir(parentDirPath) == True]


if os.path.isdir(parentDirPath + "/" + folder) == True:
    folder.append(os.listdir(parentDirPath)[folder])
# %%

# POUR UNE FACE, c'est un dossier qui contient une lettre
# CLASSE ARGUMENT: FACE, donc la lettre
# Trouver tous les fichiers qui contiennent
# loadtxt
# init : chercher toutes les références nécessaires au traitement
# Classe sert a traiter une surface: colonne de fréquence pour tout,
# %%
np.loadtxt("/Users/maximeglomot/Cours/L3_2020/Vibra/1_DDL")
# %%

for i in range(1,9):
    print(i)
class TP(object):
    ""
    def __init__(self,parentDirPath):
        self.parentDirPath=parentDirPath
        self.measureDict={}
    def getMeasurementData(self):
        for folder in os.listdir(self.parentDirPath):
            if os.path.isdir(self.parentDirPath + "/" + folder) == True:
                self.measureDict[folder]={}
                for file in os.listdir(self.parentDirPath + "/" + folder):
                    if file[0] != ".":
                        line=open(self.parentDirPath + "/" + folder + "/" + file,'r').readline()
                        self.measureDict[folder][file]={"labels":line,"data" : np.loadtxt(self.parentDirPath + "/" + folder + "/" + file , skiprows=1)}
    def plotTemporal(self,measurename):
        n_channels=self.measureDict[measurename]["TemporalData.txt"]["data"].shape[1]-1
        x=self.measureDict[measurename]["TemporalData.txt"]["data"][:,0]
        fig, ax = plt.subplots()
        ax.grid()
        # labels=self.measureDict[measurename]["TemporalData.txt"]["labels"]
        # mots = labels.split()
        # for i in range(2,len(mots),2):
        #     label=mots[i] # marteau
        #     unity=mots[i+1]
        # ax.labels=(label + r '$ \si{'+unity+r'} $')
        ax.set_xlim(0.5,0.8)
        for n in range(1,n_channels+1):

            y=self.measureDict[measurename]["TemporalData.txt"]["data"][:,n]

            ax.plot(x,y,'r')
            print('bonjour')
        plt.show()
        return(n_channels)

test=TP(parentDirPath)
test.getMeasurementData()
test.measureDict
test.plotTemporal("Mesure1")
#
#
#             self.measurementPointName = measurementPointName
#         # Register Every directory
#         self.dictionnary={Temporal}
#         self.numFiles = np.array([folders for folders in os.listdir(parentDirPath) if len(folders) == 2 and folders[1] == "b" or folders[1] =="l" or folders[1] =="t" or folders[1] =="r" or folders[1] =="f"])
#         self.measuredSurface = face  # La surface gauche : 'l'
#         self.numb = []
#         self.pointName = []
#         # self.numb = np.array([], dtype=int)
#         # self.pointNumberBelongingToSurface = np.array([], dtype=int)
#         for folders in os.listdir(parentDirPath):  # Parcours le dossier
#             # in front of the speaker
#             if len(folders) == 2 and "{}".format(self.measuredSurface) in folders:
#                 # self.numb=int(folders[0]) # 1 2 3 4 num measuredPoint, pas très utile...
#                 self.pointName.append(folders)      # Folders strings
#                 # self.numb.append(int(folders[0])) # 1 2 3 4
#
#         frequency = np.arange(20, 4001, 1)
#         dx = 20e-3  # écartement des deux microphones (m)
#         rho = 1.2  # densité volumique de l'air (kg/m3)
#         c = 340  # célérité de l'air (m/s)
#
#         RealImag = []
#         PowerSpectrum = []
#         self.datas = [frequency, [RealImag], [PowerSpectrum]] # DATA ARRAY
#         self.datasFRF = []
#         self.datasPowerSpec = []
#         self.Iac_mes = np.empty(self.datas[0].size) #INTENSITY
#         self.mod=[]
#         self.phase=[]
#         for points in range(len(self.pointName)):
#             # self.datas=np.append(np.loadtxt(os.path.join(parentDirPath),self.pointNumberBelongingToSurface+face,'/FRF_RealImag.txt'))
#             # Ajoute les 4 set de données relatives à la face demandée.
#             # FRF [:,3] MODULE /// [:,4] PHASE
#             self.datasFRF.append(np.loadtxt(parentDirPath + '{}'.format(
#                 self.pointName[points]) + '/FRF_RealImag.txt', skiprows=1, delimiter="\t"))
#             RealImag.append(self.datasFRF[points][:, 3] + 1j * self.datasFRF[points][:, 4])
#             self.mod.append(np.abs(RealImag)[points][:])
#             self.phase.append(np.angle(RealImag[points][:]))
#
#             # self.datas.append(
#                 # self.datasFRF[points][:, 3] + 1j * self.datasFRF[points][:, 4])
#
#             # POWER SPECTRUM
#             self.datasPowerSpec.append(np.loadtxt(parentDirPath + '{}'.format(
#                 self.pointName[points]) + '/PowerSpectrum.txt', skiprows=1, delimiter="\t"))
#             PowerSpectrum.append((self.datasPowerSpec[points][:, 2] + self.datasPowerSpec[points][:, 3]) / 2)
#             # PowerSpec_chpLibre = (
#             #     self.datasPowerSpec[points][:, 2] + self.datasPowerSpec[points][:, 3]) / 2
#         #self.Iac_mes=np.array( -np.imag(np.array(PowerSpectrum , dtype= float )*np.array(RealImag, dtype= complex))/ (2*np.pi*frequency*rho*dx))
#
#         ## DATAS BEING FANCY LISTS
#         #self.Iac_mes=[( -np.imag(np.array(PowerSpectrum , dtype= float )*np.array(RealImag, dtype= complex))/ (2*np.pi*frequency*rho*dx))]
#         #self.datas = [frequency, RealImag, PowerSpectrum, self.Iac_mes]
#         ## DATAS BEING ARRAYS
#         self.Iac_mes=np.array( -np.imag(np.array(PowerSpectrum , dtype= float )*np.array(RealImag, dtype= complex))/ (2*np.pi*frequency*rho*dx))
#         self.datas = np.array([np.array(frequency), np.array(RealImag), np.array(PowerSpectrum),np.array(self.Iac_mes)])
#
#
#             # for freqs in range(len(self.datas[0])):
#             #     self.Iac_mes.append( -np.imag((self.datas[1][points][freqs]*self.datas[2][points][freqs]) / (2*np.pi*frequency*rho*dx) ) )
#
#         #     self.pointName=np.array(self.pointName)
#         #     Iac_mes=np.empty([self.pointName.shape[0],self.datas[0].size])
#         #     for freqs in range(len(self.datas[0])):
#         #         Iac_mes[points][freqs]= -np.imag( (self.datas[1][points][freqs]*self.datas[2][points][freqs]) / (2*np.pi*frequency*rho*dx) )
#
#
#             #self.PowerSpec_chpLibre = (temp[:,2]+temp[:,3])/2
#
#
#             #self.FRF_chpLibre= self.datas[:,3]+1j*self.datas[:,4]
#     def getFreqIndex(self, freqs):
#         "Cette fonction retourne l'index correspondant à la fréquence d'interet "
#         #angleIndex = np.where(self.measuredPoint >= angle)[0][0]
#         freqEtudeIndex = np.where(self.datas[0] >= freqs)[0][0]
#         return freqEtudeIndex
#
#     def getIntensityvalueAtFreq(self, freqs):
#         "Cette fonction retourne la valeur de l'intensité pour la fréquence d'interet (non utiliséen)"
#         #angleIndex = np.where(self.measuredPoint >= angle)[0][0]
#         freqEtudeIndex = np.where(self.datas[0] >= freqs)[0]
#         return self.datas[3][:,freqEtudeIndex]
#
#     # def acousticPower(self):
# #
# #
# #
# #     def cartographyAtFreq(self,freqs,face=None,Lowres=None,save=None,dispIac=None):
# #         """ Cette fonction plot la valeur de l'intensité pour la surface mesurée entrée:
# #         args: 'l'= left, 'r' = right, 't'=top 'b'=back 'f'=front """
# #         # pour le prochain plot
# #         a = np.linspace(0,1,6)
# #         b = np.linspace(0,1,6)
# #         aa, bb = np.meshgrid(a,b)
# #         self.zz = np.zeros((6, 6))
# #         self.freqEtude=int(freqs)
# #         if face is None:
# #             numFaces=self.numFiles.size
# #             freqEtudeIndex=self.getFreqIndex(freqs=freqs)
# #
# #             # FILL IN THE MATRIX TO PLOT
# #             #BACK
# #             for ii in range(1,aa.shape[1]-1):
# #                 for jj in range(0,1):
# #                     self.zz[jj][ii] = Intensimetry('b').datas[3][:,freqEtudeIndex][jj]
# #
# #             #FRONT
# #             for ii in range(1,aa.shape[1]-1):
# #                 for jj in range(aa.shape[1]-1,aa.shape[1]):
# #                     self.zz[jj][ii] = Intensimetry('f').datas[3][:,freqEtudeIndex][ii-1]
# #             #LEFT
# #             for ii in range(0,1):
# #                 for jj in range(1,aa.shape[1]-1):
# #                     self.zz[jj][ii] = 0  Intensimetry('l').datas[3][:,freqEtudeIndex][ii]
# #             #RIGHT
# #             for ii in range(aa.shape[1]-1,aa.shape[1]):
# #                 for jj in range(1,aa.shape[1]-1):
# #                     self.zz[jj][ii] = Intensimetry('r').datas[3][:,freqEtudeIndex][jj-1]
# #
# #         fig = plt.figure(figsize=(15,6))
# #         ax = fig.gca(projection='3d')
# #         X,Y=aa,bb
# #         if Lowres is None:
# #             Z=self.zz
# #             ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# #             cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
# #             cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
# #             cset = ax.contourf(X, Y, Z, zdir='y', offset=0, cmap=cm.coolwarm)
# #
# #         elif Lowres is "HD":
# #             Xplus=np.linspace(0,1,100)
# #             f = interp2d(Xplus, Xplus, self.zz, kind='cubic')
# #             HDdata = f(Y[:,0],X[0,:])
# #             Z=HDdata
# #             ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
# #             cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.coolwarm)
# #             cset = ax.contourf(X, Y, Z, zdir='x', offset=0, cmap=cm.coolwarm)
# #             cset = ax.contourf(X, Y, Z, zdir='y', offset=0, cmap=cm.coolwarm)
# #
# #         ax.set_xlabel('X')
# #         ax.set_xlim(0, 1)
# #         ax.set_ylabel('Y')
# #         ax.set_ylim(ax.get_ylim()[::-1])
# #         ax.set_zlabel('Z')
# #         ax.set_zlim(-np.max(abs(self.zz)), np.max(abs(self.zz)))
# #         if save is not None:
# #             plt.savefig("CartogIntensity"+"{}".format(self.freqEtude))
# #         if dispIac is None:
# #             return (fig)
# #         else:
# #             return (plt.show(), Z, aa,bb)
# #
# # #Intensimetry.acousticPower(Intensimetry('l'))
# #         # elif face == "l" or "left":
# #         #     self.dataleft=np.array(Intensimetry('l').datas)
# #         #
# #         # elif face == "t" or "top":
# #         #     self.datatop=np.array(Intensimetry('t').datas)
# #         # elif face == "r" or "right":
# #         #     self.dataright=np.array(Intensimetry('r').datas)
# #         # elif face == "b" or "back" :
# #         #     self.databack=np.array(Intensimetry('b').datas)
# #         # elif face == "f" or "front":
# #         #     self.datafront=np.array(Intensimetry('f').datas)
# #         # else : print("La face que vous avez commandée n'est pas sur le menu.")
# #         ##########
# # #%%
# # intensityLeft=Intensimetry('l')
# # intensityLeft.aa
# # Intensimetry('l').cartographyAtFreq(freqs=(4000),dispIac=None,save=None)
# # Intensimetry('l').cartographyAtFreq(freqs=(300),dispIac=None,save=True)
# # Intensimetry('l').cartographyAtFreq(freqs=(100),dispIac=None)
# #
# #
# # Intensimetry('l').getFreqIndex(900)
# # Intensimetry('b').datas[3][:,781]
# #
# # logInt,logIndex=[],[]
# # logspace=2*np.logspace(1,4,num=31,base=10)
# # for freqs in range(0,np.where(logspace>=4000)[0][0]):
# #     logInt.append(int(logspace[freqs]))
# #     #logIndex.append(Intensimetry('l').getFreqIndex(logInt[freqs]))
# #
# # fig, ax = plt.subplots(figsize=(10, 4))
# # # ax.bar(levelDifference[1, 7:-6], Dnt, width=levelDifference[1, 7:-6] / 4, color='darkgrey',
# # #        ec="k"*31, zorder=2, label=r'Level Difference unbiased $D_{n,T}$')
# # ax.bar(logInt, Lw, width=logInt[1, 7:-6] / 4, color='lightsteelblue',
# #        ec="k"*31, zorder=2, label=r'Sound Intensity Level')
# # #ax.semilogx(levelDifference[1, 7:-6], interpGaugeLine,
# #             #'--+', label='Adjusted Gauge Line')
# # ax.set_xscale("log")
# # plt.ylabel(r'$dB SWL$')
# # plt.xlabel(r'Frequency $Hz$')
# # # ax.text(460, interpGaugeLine[1]-8.6, r'$500$', color='black', fontsize=11)
# # # ax.text(450, interpGaugeLine[7] +2, r'$23.48$', color='r', fontsize=11)
# #
# # plt.legend(loc="best")
# # plt.grid(which='minor', linestyle=':', zorder=0)
# # plt.grid(which='major', linestyle='-', zorder=1)
# # plt.tight_layout()
# # plt.savefig(os.path.join(parentDirPath, 'intensimetrySWL.pdf'))
# # plt.show()
# # #%%
# # #for i in range(0,5):
# #
# # #%%
# # for freq in range(3981):
# #     print(AllIntensim[i][0][:,freq].shape)
# # Wac=sum(AllIntensim[:][0][:,freq])
# #
# # #%%
# #
# # Wac.shape
# # #Wac=[]
# # AllIntensim=np.array([[Intensimetry('b').datas[3][:,:]],[Intensimetry('f').datas[3][:,:]],[Intensimetry('l').datas[3][:,:]],[Intensimetry('r').datas[3][:,:]],[Intensimetry('t').datas[3][:,:]]])
# #
# # AllIntensim.shape==Intensimetry('b').datas[3][:,:].shape
# # Wac=np.empty((len(Intensimetry('b').datas[0]),1))
# # Lw=np.empty((len(Intensimetry('b').datas[0]),1))
# # for freqz in range(len(Intensimetry('b').datas[0])):
# #     Lw[freqz]=10*np.log10(Wac[freqz] / 10**(-12))
# # #plt.plot(Intensimetry('b').datas[0] , Wac)
# # plt.plot(Intensimetry('b').datas[0] , Lw ,'-')
# # plt.show()
# #
# #
# # #%%
# #
# # Wac = np.sum(np.array([[Intensimetry('b').datas[3][:,freqz]],[Intensimetry('f').datas[3][:,freqz]],[Intensimetry('l').datas[3][:,freqz]],[Intensimetry('r').datas[3][:,freqz]],[Intensimetry('t').datas[3][:,freqz]]]))
# # np.array([[Intensimetry('b').datas[3][:,freqz]],[Intensimetry('f').datas[3][:,freqz]],[Intensimetry('l').datas[3][:,freqz]],[Intensimetry('r').datas[3][:,freqz]],[Intensimetry('t').datas[3][:,freqz]]]).shape
# #
# # W = np.zeros(freq.shape)
# # for i in range(freq.shape[0]):
# #     W += I[:, i] * sphericalSurface
# #
# #
# #
# # plt.plot(Intensimetry('l').datas[0],Intensimetry('l').datasFRF)
# #
# # np.array(Intensimetry('l').datas[1]).shape
# #
# # mod = np.abs(Intensimetry('l').datasFRF[:,3]+1j*Intensimetry('l').datasFRF[:,4])
# # phase = np.angle(FRF_chpLibre)
# #
# # fig, ax = plt.subplots(2,1) #création figure et axes
# #
# # # plot
# # ax[0].plot(freq, mod)
# # ax[1].plot(freq, phase)
# # #ax[1].plot(freq, np.unwrap(2*phase)/2) #phase déroulée...
# #
# #
# # # mises en forme
# # ax[0].set_xlim([Intensimetry('f').datas[0][0], Intensimetry('l').datas[0][-1]])
# # ax[0].set_xlabel("Fréquence (Hz)")
# # ax[0].set_ylabel("Module (-)")
# # ax[0].grid()
# # ax[0].set_title(""" Dans l'axe du HP +r' : FRF de $P_2/P_1$'""")
# #
# # ax[1].set_xlim([freq[0], freq[-1]])
# # ax[1].set_xlabel("Fréquence (Hz)")
# # ax[1].set_ylabel("Phase (rad)")
# # ax[1].grid()
# # plt.tight_layout()
# #
# #
# # # affichage
# # plt.show()
# #
# # #%%
# #
# # dx = 20e-3 #écartement des deux microphones (m)
# # rho = 1.2; #densité volumique de l'air (kg/m3)
# # c   = 340; #célérité de l'air (m/s)
# # freq= posaRI[:,0]
# # posaRI=np.loadtxt(parentDirPath+"7-a/FRF_RealImag.txt",skiprows=1)
# # posbRI=np.loadtxt(parentDirPath+"7-b/FRF_RealImag.txt",skiprows=1)
# # poscRI=np.loadtxt(parentDirPath+"7-c/FRF_RealImag.txt",skiprows=1)
# # posaRI.shape
# #
# # posaPS=np.loadtxt(parentDirPath+"7-a/PowerSpectrum.txt",skiprows=1)
# # posbPS=np.loadtxt(parentDirPath+"7-b/PowerSpectrum.txt",skiprows=1)
# # poscPS=np.loadtxt(parentDirPath+"7-c/PowerSpectrum.txt",skiprows=1)
# #
# # FRF_champlibre_a=posaRI[:,3]+1j*posaRI[:,4]
# # FRF_champlibre_b=posbRI[:,3]+1j*posbRI[:,4]
# # FRF_champlibre_c=poscRI[:,3]+1j*poscRI[:,4]
# #
# # PowerSpec_chpLibre_a= (posaPS[:,2]+posaPS[:,3])/2
# # PowerSpec_chpLibre_b= (posaPS[:,2]+posaPS[:,3])/2
# # PowerSpec_chpLibre_c= (posaPS[:,2]+posaPS[:,3])/2
# #
# #
# # Iaca= -np.imag((FRF_champlibre_a*PowerSpec_chpLibre_a)/(2*np.pi*freq*rho*dx));
# # Iacb= -np.imag((FRF_champlibre_b*PowerSpec_chpLibre_b)/(2*np.pi*freq*rho*dx));
# # Iacc= -np.imag((FRF_champlibre_c*PowerSpec_chpLibre_c)/(2*np.pi*freq*rho*dx));
# #
# # moda=np.abs(FRF_champlibre_a)
# # modb=np.abs(FRF_champlibre_b)
# # modc=np.abs(FRF_champlibre_c)
# #
# # phasa=np.angle(FRF_champlibre_a)
# # phasb=np.angle(FRF_champlibre_b)
# # phasc=np.angle(FRF_champlibre_c)
# #
# # plt.figure()
# # plt.plot(moda,'b')
# # plt.plot(modb,'g')
# # plt.plot(modc,'orange')
# # plt.legend()
# # plt.show()
# #
# # plt.figure()
# # plt.plot(phasa,'b')
# # plt.plot(phasb,'g')
# # plt.plot(phasc,'orange')
# # plt.legend()
# # plt.show()
# #
# # plt.figure()
# # plt.plot(Iaca,'b')
# # plt.plot(Iacb,'g')
# # plt.plot(Iacc,'orange')
# # plt.legend()
# # plt.show()
# #
# # #%%
# #
# # fig, ax = plt.subplots(2,1) #création figure et axes
# #
# # # plot
# # ax[0].plot(freq, moda)
# #
# #
# # ax[1].plot(freq, phasa)
# #
# # #ax[1].plot(freq, np.unwrap(2*phase)/2) #phase déroulée...
# #
# #
# # # mises en forme
# # ax[0].set_xlim([freq[0], freq[-1]])
# # ax[0].set_xlabel("Fréquence (Hz)")
# # ax[0].set_ylabel("Module (-)")
# # ax[0].grid()
# # #ax[0].set_title(r' : FRF de $P_2/P_1$')
# #
# # ax[1].set_xlim([freq[0], freq[-1]])
# # ax[1].set_xlabel("Fréquence (Hz)")
# # ax[1].set_ylabel("Phase (rad)")
# # ax[1].grid()
# # plt.tight_layout()
# # plt.savefig("7a_FRF.png")
# # # affichage
# # plt.show()
# #
# # #%%
# #
# # fig, ax = plt.subplots(2,1) #création figure et axes
# #
# # # plot
# # ax[0].plot(freq, modb,"brown")
# #
# #
# # ax[1].plot(freq, phasb,"brown")
# #
# # #ax[1].plot(freq, np.unwrap(2*phasb)/2) #phase déroulée...
# #
# #
# # # mises en forme
# # ax[0].set_xlim([freq[0], freq[-1]])
# # ax[0].set_xlabel("Fréquence (Hz)")
# # ax[0].set_ylabel("Module (-)")
# # ax[0].grid()
# # #ax[0].set_title(r' : FRF de $P_2/P_1$')
# #
# # ax[1].set_xlim([freq[0], freq[-1]])
# # ax[1].set_xlabel("Fréquence (Hz)")
# # ax[1].set_ylabel("Phase (rad)")
# # ax[1].grid()
# # plt.tight_layout()
# # plt.savefig("7b_FRF.png")
# #
# #
# # # affichage
# # plt.show()
# #
# # #%%
# # fig, ax = plt.subplots(2,1) #création figure et axes
# #
# # # plot
# # ax[0].plot(freq, modc,"darkorange")
# #
# #
# # ax[1].plot(freq, phasc,"darkorange")
# #
# # #ax[1].plot(freq, np.unwrap(2*phasb)/2) #phase déroulée...
# #
# #
# # # mises en forme
# # ax[0].set_xlim([freq[0], freq[-1]])
# # ax[0].set_xlabel("Fréquence (Hz)")
# # ax[0].set_ylabel("Module (-)")
# # ax[0].grid()
# # #ax[0].set_title(r' : FRF de $P_2/P_1$')
# #
# # ax[1].set_xlim([freq[0], freq[-1]])
# # ax[1].set_xlabel("Fréquence (Hz)")
# # ax[1].set_ylabel("Phase (rad)")
# # ax[1].grid()
# # plt.tight_layout()
# # plt.savefig("7c_FRF.png")
# #
# # # affichage
# # plt.show()
# #
# # #%%
# #
# # ##########################################
# # # Visuallisation du spectre en puissance #
# # ##########################################
# # fig2, ax2 = plt.subplots() #création figure et axes
# #
# # # plot
# # ax2.plot(freq[::2], PowerSpec_chpLibre_b[::2],"brown")
# #
# # # mises en forme
# # ax2.set_xlim([freq[0], freq[-1]])
# # ax2.set_xlabel("Fréquence (Hz)")
# # ax2.set_ylabel("Module (-)")
# # ax2.grid()
# # #ax2.set_title(expName+r' : $|P_1|^2$')
# # plt.tight_layout()
# # plt.savefig("7b-PowerSpec.png")
# # # affichage
# # plt.show()
# # freq
# # #%%
# # #################################
# # # Visuallisation de l'intensité #
# # #################################
# # fig3, ax3 = plt.subplots() #création figure et axes
# #
# # # plot
# # ax3.plot(freq, Iaca, label="position a")
# # ax3.plot(freq, Iacb,"brown" , label="position b")
# # ax3.plot(freq, Iacc,"darkorange",label="position c")
# # # mises en forme
# # ax3.set_xlim([freq[0], freq[-1]])
# # ax3.set_xlabel("Fréquence (Hz)")
# # ax3.set_ylabel(r"Amplitude (W.m$^{-2}$)")
# # ax3.grid()
# # #ax3.set_title(expName+r' : $I_{ac}$')
# # plt.legend()
# # plt.tight_layout()
# #
# # plt.savefig("7abc-Iac")
# #
# # # affichage
# # plt.show()
# #
# # #%%
# # ###################
# # # Calcul de rho*c #
# # ###################
# #
# # # Impédance
# # rhoc_mes = -(2*np.pi*freq*rho*dx)/np.imag(FRF_champlibre_a);
# #
# #
# # ###########################
# # # Visuallisation de rho*c #
# # ###########################
# # fig3, ax3 = plt.subplots() #création figure et axes
# #
# # # plot
# # ax3.plot(freq, rhoc_mes,label='mes.')
# # ax3.plot(freq, np.ones(len(freq))*rho*c,'--k',label='th.')
# #
# # # mises en forme
# # ax3.set_xlim([freq[0], freq[-1]])
# # ax3.set_ylim([-1000, 1000])
# # ax3.set_xlabel("Fréquence (Hz)")
# # ax3.set_ylabel(r"Amplitude (kg.m$^{-2}$.s$^{-1}$)")
# # ax3.grid()
# # #ax3.set_title(Mesure ()r' : $\rho \cdot c$')
# # ax3.legend()
# # plt.tight_layout()
# # plt.savefig("InverseIntensiteReactive")
# # # affichage
# # plt.show()
# # np.imag(FRF_champlibre_a)
