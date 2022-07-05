#This is just a bunch of libraries
import time
start_time = time.time()
import numpy as np
import itertools
from numpy import linalg as LA
from numpy import matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
#from joblib import Parallel, delayed
import multiprocessing
from itertools import product
import pylab
from joblib import Parallel, delayed
import multiprocessing
import os

def vects_Mullen0D(width, long):
    """This function defines the vectors of the 0D graphene armchair ribbon with defined zigzag edges in function of its width and long.
    """
    cellpos = [] #These are the positions of the unit cell
    vI = np.array([0.0, 0.0, 0.0])
    vII = np.array([0.7, 1.212435565, 0.0])
    vIII = np.array([2.1, 1.212435565, 0.0])
    vIV = np.array([2.8, 0.0, 0.0])
    cellpos.append(vI)
    cellpos.append(vII)
    cellpos.append(vIII)
    cellpos.append(vIV)

    pos = [] #This keeps the positions of every atom
    pos.append(vI)
    pos.append(vII)
    pos.append(vIII)
    pos.append(vIV)

    xcord = [] #This keeps the x position of every atom of the first line
    xcord.append(vI[0])
    xcord.append(vII[0])
    xcord.append(vIII[0])
    xcord.append(vIV[0])

    ycord = [] #This keeps the y position of every atom of the first column
    ycord.append(vI[1])
    ycord.append(vII[1])
    ycord.append(vIII[1])
    ycord.append(vIV[1])

    countloopi = 1
    countx = 1
    while True: #This while loop creates the vectors of the cells of the first line,
        for i in cellpos: #at the end the number of cells created + the unit cell will be equal to the width
            if countx == width:
                break
            vnx = np.array([i[0] + countx * 4.2, i[1], i[2]])
            pos.append(vnx)
            xcord.append(vnx[0])
            if countloopi % 4 == 0:
                countx += 1
            countloopi += 1
        if countx == width:
            break

    countloopj = 1
    county = 1
    countwidth = 1
    while True: #This while loop creates the vectors of the cells of the first column (except those of the unit cell)
        for j in cellpos: #and also for every cell created fills that line with a number of cells equal to the width
            if county == long:
                break
            if county < long:	#This creates the first column
                vny = np.array([j[0], j[1] - county * 2.424871131, j[2]])
                pos.append(vny)
                ycord.append(vny[1])
            if width > 1 and county < long: #This fills the lines
                for widthpos in range(1, width):
                    vnxy = np.array([xcord[4 * widthpos + countwidth - 1], j[1] - county * 2.424871131, j[2]])
                    pos.append(vnxy)
                countwidth += 1
                if countwidth == 5:
                    countwidth = countwidth - 4
            if countloopj % 4 == 0:
                county += 1
            countloopj += 1
        if county == long:
            break
    longminusone = long - 1
    for edgexpos in range(0, width): #This creates the last line of atoms for closing the last line of cells
        vnclose1 = np.array([xcord[1 + 4 * edgexpos], ycord[1 + 4 * longminusone] - 2.424871131, 0.0])
        vnclose2 = np.array([xcord[2 + 4 * edgexpos], ycord[2 + 4 * longminusone] - 2.424871131, 0.0])
        pos.append(vnclose2)
        pos.append(vnclose1)
    return pos

def save_representation(y, x, name):
    fil= str(name) + '.dat'
    """saves 2 lists and leaves them for veusz
    """
    for i in list(range(len(y))):
        yi = y[i]
        xi = x[i]
        with open(fil, 'a') as f:
            f.write(str(np.real(yi)) + ' ' + str(xi) + '\n')

def plot_wavefunction_sign(pos, vecsup, vecsdown, bondpos):
    Xup, Xdown, Yup, Ydown, Zup, Zdown, Sup, Sdown, Signup, Signdown = [], [], [], [], [], [], [], [], [], []
    #print (vecsup)
    for i in range(len(pos)):
        r = pos[i]     # positions
        v = vecsup[i]    # eigvector component corresponding to atom i
        Xup.append(r[0])
        Yup.append(r[1])
        Zup.append(r[2])
        #print (v)
        conjvect = (50000*(abs(v)))
        Sup.append(conjvect)
        """
        if abs(v) > 0.0001:
            Sup.append(2000*abs(v))
        if abs(v) < 0.0001:
            Sup.append(0.0)
        if v <= 0:
            #print (r,i,'minus',v)
            Signup.append('b')
        if v > 0:
            #print (r,i,'plus',v)
            Signup.append('r')
        """
        if abs(v.imag) == 0:
            if v.real <= 0:
                #print (r,i,'minus',v)
                Signup.append('b')
            if v.real > 0:
                #print (r,i,'plus',v)
                Signup.append('r')
        if abs(v.imag) != 0:
            if v.imag <= 0:
                #print (r,i,'minus',v)
                Signup.append('b')
            if v.imag > 0:
                #print (r,i,'plus',v)
                Signup.append('r')
    """
    for j in range(len(pos)):
        r = pos[j]     # positions
        v = vecsdown[j]    # eigvector component corresponding to atom j
        Xdown.append(r[0])
        Ydown.append(r[1])
        Zdown.append(r[2])
        conjvect = (v*np.conj(v))
        if abs(v) > 0.0001:
            Sdown.append(500)
        if abs(v) < 0.0001:
            Sdown.append(0.0)
        if v < 0:
            Signdown.append('b')
        if v > 0:
            Signdown.append('r')
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x($\AA$)")
    ax.set_ylabel("y($\AA$)")
    font = {'size'   : 20}
    plt.rc('font', **font)
    #ax.set_title("spin up and down wavefunctions")

    for bond in bondpos:
        ax.plot(bond[0:2], bond[-2:], 'k-')

    ax.scatter(Xup,Yup,c=Signup,s=Sup,edgecolors='none', alpha = 1) # Plots the up wavefunction
    #fig.patch.set_visible(False)
    ax.axis('off')
    #plt.xlim(-100.0,100.0)
    #plt.ylim(-6, 3)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

def transpentachain_FM(numberpentagones):

    pos = []
    cellpos = []
    pos1 = np.array([0.0,0.0,0.0])
    pos2 = np.array([-1.42*np.sin(54*(np.pi/180)),1.42*np.cos(54*(np.pi/180)),0.0])
    pos3 = np.array([1.42*np.sin(54*(np.pi/180)),1.42*np.cos(54*(np.pi/180)),0.0])
    pos4 = np.array([-1.42/2,(1.42/(2*np.cos(54*(np.pi/180))))+(1.42*np.sin(54*(np.pi/180))/(2*np.cos(54*(np.pi/180)))),0.0])
    pos5 = np.array([1.42/2,(1.42/(2*np.cos(54*(np.pi/180))))+(1.42*np.sin(54*(np.pi/180))/(2*np.cos(54*(np.pi/180)))),0.0])

    pos6 = np.array([pos4[0]-1.42*np.cos(48*(np.pi/180)),pos4[1]+1.42*np.sin(48*(np.pi/180)),0.0])
    pos7 = np.array([pos4[0]-1.42*np.sqrt(3)*np.cos(18*(np.pi/180)),pos4[1]+1.42*np.sqrt(3)*np.sin(18*(np.pi/180)),0.0])
    pos8 = np.array([pos2[0]-1.42*np.sqrt(3)*np.cos(18*(np.pi/180)),pos2[1]+1.42*np.sqrt(3)*np.sin(18*(np.pi/180)),0.0])
    pos9 = np.array([pos2[0]-1.42*np.sin(78*(np.pi/180)),pos2[1]-1.42*np.cos(78*(np.pi/180)),0.0])

    pos10 = np.array([pos5[0]+1.42*np.cos(48*(np.pi/180)),pos5[1]+1.42*np.sin(48*(np.pi/180)),0.0])
    pos11 = np.array([pos5[0]+1.42*np.sqrt(3)*np.cos(18*(np.pi/180)),pos5[1]+1.42*np.sqrt(3)*np.sin(18*(np.pi/180)),0.0])
    pos12 = np.array([pos3[0]+1.42*np.sqrt(3)*np.cos(18*(np.pi/180)),pos3[1]+1.42*np.sqrt(3)*np.sin(18*(np.pi/180)),0.0])
    pos13 = np.array([pos3[0]+1.42*np.sin(78*(np.pi/180)),pos3[1]-1.42*np.cos(78*(np.pi/180)),0.0])

    pos14 = np.array([pos11[0]+1.42*np.sin(54*(np.pi/180)), pos11[1]+ 1.42*np.cos(54*(np.pi/180)), 0.0])
    pos15 = np.array([pos11[0]+2*1.42*np.sin(54*(np.pi/180)), pos11[1], 0.0])
    pos16 = np.array([pos12[0]+2*1.42/2,pos12[1],0.0])
    pos17 = np.array([pos15[0]-pos7[0]+pos6[0],pos6[1],0.0])
    pos18 = np.array([pos17[0]-pos6[0]+pos4[0],pos4[1],0.0])
    pos19 = np.array([pos18[0]-pos4[0]+pos2[0],pos3[1],0.0])
    pos20 = np.array([pos19[0]-pos2[0]+pos9[0],pos9[1],0.0])

    cellpos5 = []

    pos.append(np.array([pos3[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos3[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos3[2]]))
    pos.append(np.array([pos5[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos5[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos5[2]]))
    pos.append(np.array([pos10[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos10[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos10[2]]))
    pos.append(np.array([pos11[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos11[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos11[2]]))
    pos.append(np.array([pos12[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos12[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos12[2]]))
    pos.append(np.array([pos13[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos13[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos13[2]]))
    pos.append(np.array([pos14[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos14[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos14[2]]))
    pos.append(np.array([pos15[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos15[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos15[2]]))
    pos.append(np.array([pos16[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos16[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos16[2]]))
    pos.append(np.array([pos17[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos17[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos17[2]]))
    pos.append(np.array([pos18[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos18[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos18[2]]))
    pos.append(np.array([pos19[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos19[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos19[2]]))
    pos.append(np.array([pos20[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos20[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos20[2]]))


    cellpos5.append(np.array([pos14[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos14[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos14[2]]))
    cellpos5.append(np.array([pos15[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos15[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos15[2]]))
    cellpos5.append(np.array([pos16[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos16[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos16[2]]))
    cellpos5.append(np.array([pos17[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos17[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos17[2]]))
    cellpos5.append(np.array([pos18[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos18[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos18[2]]))
    cellpos5.append(np.array([pos19[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos19[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos19[2]]))
    cellpos5.append(np.array([pos20[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos20[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos20[2]]))

    cellpos.append(np.array([pos14[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos14[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos14[2]]))
    cellpos.append(np.array([pos3[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos3[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos3[2]]))
    cellpos.append(np.array([pos5[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos5[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos5[2]]))
    cellpos.append(np.array([pos10[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos10[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos10[2]]))
    cellpos.append(np.array([pos11[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos11[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos11[2]]))
    cellpos.append(np.array([pos12[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos12[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos12[2]]))
    cellpos.append(np.array([pos13[0]-((1.42/2)+1.42*np.sqrt(3)*np.cos(18*(np.pi/180))+1.42*np.sin(54*(np.pi/180))),pos13[1]-(1.42*np.cos(54*(np.pi/180))+1.42*np.sqrt(3)*np.sin(18*(np.pi/180))+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),pos13[2]]))

    #cellpos2 = rotational_matrix3DZaxis(cellpos, (-180-72-18)*(np.pi/180))
    #np.array([firstrmod*np.cos(theta*(np.pi/180)), firstrmod*np.sin(theta*(np.pi/180)),0.0])
    #for atom in cellpos2:
        #pos.append(np.array([atom[0]+np.cos((-72-18)*(np.pi/180))*(1.42*0.5*np.sqrt(3)+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),atom[1]+np.sin((-72-18)*(np.pi/180))*(1.42*0.5*np.sqrt(3)+((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180))))),atom[2]]))


    cellpos3 = []
    cellpos4 = []
    transvectlist = []

    cellpos2=[]
    cellpos2 = rotational_matrix3DZaxis(cellpos, (-180-60)*(np.pi/180))
    firstrmod = ((1.42*np.sin(54*(np.pi/180)))/(2*np.cos(54*(np.pi/180)))) + 1.42*np.sqrt(3)/2
    theta= -18
    firsttranslationvect = np.array([firstrmod*np.cos(theta*(np.pi/180)), firstrmod*np.sin(theta*(np.pi/180)),0.0])
    if len(transvectlist) == 0:
        transvectlist.append(firsttranslationvect)

    translationvect1 = np.array([firstrmod*np.cos((-60-18)*(np.pi/180))+transvectlist[len(transvectlist)-1][0],firstrmod*np.sin((-60-18)*(np.pi/180))+transvectlist[len(transvectlist)-1][1],0.0])

    translationvect2 = np.array([firstrmod*np.cos((-60-18+36)*(np.pi/180))+translationvect1[0],firstrmod*np.sin((-60-18+36)*(np.pi/180))+translationvect1[1],0.0])

    translationvect3 = np.array([firstrmod*np.cos((-60-18+36+60)*(np.pi/180))+translationvect2[0],firstrmod*np.sin((-60-18+36+60)*(np.pi/180))+translationvect2[1],0.0])

    transvectlist.append(translationvect1)
    for i in cellpos5:
        cellpos3.append(i)
    for i in cellpos2:
        cellpos3.append(np.array([i[0]+translationvect1[0],i[1]+translationvect1[1],i[2]+translationvect1[2]]))
        cellpos4.append(np.array([i[0]+translationvect1[0],i[1]+translationvect1[1],i[2]+translationvect1[2]]))
    for atom in cellpos4:
        pos.append(atom)
    print(translationvect3)

    if numberpentagones > 1:
        if numberpentagones % 2 == 0:
            number = int(numberpentagones/2)
            for n in range(1,number):
                for atom in cellpos3:
                    pos.append(np.array([atom[0]+(n)*translationvect3[0],atom[1]+(n)*translationvect3[1],atom[2]+(n)*translationvect3[2]]))

        if numberpentagones % 2 != 0:
            number = int((numberpentagones+1)/2)
            for n in range(1,number):
                for atom in cellpos3:
                    pos.append(np.array([atom[0]+n*translationvect3[0],atom[1]+n*translationvect3[1],atom[2]+n*translationvect3[2]]))
            del pos[len(pos)-1]
            del pos[len(pos)-1]
            del pos[len(pos)-1]
            del pos[len(pos)-1]
            del pos[len(pos)-1]
            del pos[len(pos)-1]
            del pos[len(pos)-1]

    return pos

def read_and_write_eigvalues():
    """This function reads a text document with some eigenvalues and create a list that stores them
    """
    eig_vals = []
    file = open('map.dat', 'r')
    lines = file.readlines()
    for line in lines:
        x = ''
        for i in range(len(line)):
            x += line[i]
            if line[i] == ' ':
                break
        eig_vals.append(float(x))
    return eig_vals

def correlator_map_improved(pos, eig_vects, Psi, configurationslist, i, zeromodes, Nbot, Ntop,j):
    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot
    Nbot = 0
    Ntop = int(len(configurationslist[0])/2)
    maplist = []

    print (j,'map')
    jlist = []
    for jconf in range(len(configurationslist)):
        print(str(jconf) + ','+str(len(configurationslist)))
        for n in range(0, Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime])[0]
            for m in range(0, Numberbotstates, 1):
                vectm = np.array(eig_vects[m])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)

                for iconf in range(len(configurationslist)):
                    if statej == list(configurationslist[iconf]):
                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                        jlist.append(0.25*sum(Unnmmlist))





        for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates])[0]
            for nprime in range(Nbot, Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
                for m in range(Nbot, Ntop, 1):
                    vectm = np.array(eig_vects[m+Numberbotstates])[0]
                    for mprime in range(Nbot, Ntop, 1):
                        vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                        statej = []
                        for obj in configurationslist[jconf]:
                            statej.append(obj)
                        if statej[mprime] == 1:
                            epsilonanihidownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < mprime:
                                    epsilonanihidownlist.append(statej[fockindex])
                                #if fockindex == 0 and mprime == 0:
                                    #epsilonanihidownlist.append(0.0)
                            epsilonanihidown = sum(epsilonanihidownlist)
                            statej[mprime] = 0
                            if statej[m] == 0:
                                epsiloncreatdownlist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)/2), 1):
                                    if fockindex < m:
                                        epsiloncreatdownlist.append(statej[fockindex])
                                    #if fockindex == 0 and m == 0:
                                        #epsiloncreatdownlist.append(0.0)
                                epsiloncreatdown = sum(epsiloncreatdownlist)
                                statej[m] = 1
                                if statej[nprime] == 1:
                                    epsilonanihiuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)/2), 1):
                                        if fockindex < nprime:
                                            epsilonanihiuplist.append(statej[fockindex])
                                        #if fockindex == 0 and nprime == 0:
                                            #epsilonanihiuplist.append(0.0)
                                    epsilonanihiup = sum(epsilonanihiuplist)
                                    statej[nprime] = 0
                                    if statej[n] == 0:
                                        epsiloncreatuplist = [Numberbotstates]
                                        for fockindex in range(0, int(len(statej)/2), 1):
                                            if fockindex < n:
                                                epsiloncreatuplist.append(statej[fockindex])
                                            #if fockindex == 0 and n == 0:
                                                #epsiloncreatuplist.append(0.0)
                                        epsiloncreatup = sum(epsiloncreatuplist)
                                        statej[n] = 1
                                        for iconf in range(len(configurationslist)):
                                            if statej == list(configurationslist[iconf]):
                                                Unnmmlist = []

                                                gncompl = np.conj(vectn[i])
                                                gn = vectnprime[i]
                                                gmcompl = np.conj(vectm[j])
                                                gm = vectmprime[j]
                                                tau = np.conj(np.array(Psi)[0][iconf])
                                                tauprime = np.array(Psi)[0][jconf]
                                                Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                                fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                                jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)


        for n in range(0,Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime])[0]
            for m in range(Nbot, Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates])[0]
                for mprime in range(Nbot, Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1

                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)




        for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates])[0]
            for nprime in range(Nbot, Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
                for m in range(0,Numberbotstates, 1):
                    vectm = np.array(eig_vects[m])[0]
                    mprime = m
                    vectmprime = np.array(eig_vects[mprime])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)

                    if statej[nprime] == 1:
                        epsilonanihiuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < nprime:
                                epsilonanihiuplist.append(statej[fockindex])
                            #if fockindex == 0 and nprime == 0:
                                #epsilonanihiuplist.append(0.0)
                        epsilonanihiup = sum(epsilonanihiuplist)
                        statej[nprime] = 0
                        if statej[n] == 0:
                            epsiloncreatuplist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < n:
                                    epsiloncreatuplist.append(statej[fockindex])
                                #if fockindex == 0 and n == 0:
                                    #epsiloncreatuplist.append(0.0)
                            epsiloncreatup = sum(epsiloncreatuplist)
                            statej[n] = 1
                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)



        for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
                vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                for iconf in range(len(configurationslist)):
                    if statej == list(configurationslist[iconf]):
                        Unnmmlist = []
                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                        jlist.append(0.25*sum(Unnmmlist))


        for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                    vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                    for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                        vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                        statej = []
                        for obj in configurationslist[jconf]:
                            statej.append(obj)
                        if statej[mprime] == 1:
                            epsilonanihidownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < mprime:
                                    epsilonanihidownlist.append(statej[fockindex])
                                #if fockindex == 0 and mprime == 0:
                                    #epsilonanihidownlist.append(0.0)
                            epsilonanihidown = sum(epsilonanihidownlist)
                            statej[mprime] = 0
                            if statej[m] == 0:
                                epsiloncreatdownlist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)), 1):
                                    if fockindex < m:
                                        epsiloncreatdownlist.append(statej[fockindex])
                                    #if fockindex == 0 and m == 0:
                                        #epsiloncreatdownlist.append(0.0)
                                epsiloncreatdown = sum(epsiloncreatdownlist)
                                statej[m] = 1
                                if statej[nprime] == 1:
                                    epsilonanihiuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)), 1):
                                        if fockindex < nprime:
                                            epsilonanihiuplist.append(statej[fockindex])
                                        #if fockindex == 0 and nprime == 0:
                                            #epsilonanihiuplist.append(0.0)
                                    epsilonanihiup = sum(epsilonanihiuplist)
                                    statej[nprime] = 0
                                    if statej[n] == 0:
                                        epsiloncreatuplist = [Numberbotstates]
                                        for fockindex in range(0, int(len(statej)), 1):
                                            if fockindex < n:
                                                epsiloncreatuplist.append(statej[fockindex])
                                            #if fockindex == 0 and n == 0:
                                                #epsiloncreatuplist.append(0.0)
                                        epsiloncreatup = sum(epsiloncreatuplist)
                                        statej[n] = 1
                                        for iconf in range(len(configurationslist)):
                                            if statej == list(configurationslist[iconf]):
                                                Unnmmlist = []

                                                gncompl = np.conj(vectn[i])
                                                gn = vectnprime[i]
                                                gmcompl = np.conj(vectm[j])
                                                gm = vectmprime[j]
                                                tau = np.conj(np.array(Psi)[0][iconf])
                                                tauprime = np.array(Psi)[0][jconf]
                                                Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                                fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                                jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

        for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
            for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1

                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)


        for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
                    vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
                    mprime = m
                    vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)

                    if statej[nprime] == 1:
                        epsilonanihiuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < nprime:
                                epsilonanihiuplist.append(statej[fockindex])
                            #if fockindex == 0 and nprime == 0:
                                #epsilonanihiuplist.append(0.0)
                        epsilonanihiup = sum(epsilonanihiuplist)
                        statej[nprime] = 0
                        if statej[n] == 0:
                            epsiloncreatuplist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < n:
                                    epsiloncreatuplist.append(statej[fockindex])
                                #if fockindex == 0 and n == 0:
                                    #epsiloncreatuplist.append(0.0)
                            epsiloncreatup = sum(epsiloncreatuplist)
                            statej[n] = 1
                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)




        for n in range(0, Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime])[0]
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
                vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                for iconf in range(len(configurationslist)):
                    if statej == list(configurationslist[iconf]):
                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                        jlist.append(0.25*sum(Unnmmlist))




        for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates])[0]
            for nprime in range(Nbot, Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
                for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                    vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                    for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                        vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                        statej = []
                        for obj in configurationslist[jconf]:
                            statej.append(obj)
                        if statej[mprime] == 1:
                            epsilonanihidownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < mprime:
                                    epsilonanihidownlist.append(statej[fockindex])
                                #if fockindex == 0 and mprime == 0:
                                    #epsilonanihidownlist.append(0.0)
                            epsilonanihidown = sum(epsilonanihidownlist)
                            statej[mprime] = 0
                            if statej[m] == 0:
                                epsiloncreatdownlist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)), 1):
                                    if fockindex < m:
                                        epsiloncreatdownlist.append(statej[fockindex])
                                    #if fockindex == 0 and m == 0:
                                        #epsiloncreatdownlist.append(0.0)
                                epsiloncreatdown = sum(epsiloncreatdownlist)
                                statej[m] = 1
                                if statej[nprime] == 1:
                                    epsilonanihiuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)/2), 1):
                                        if fockindex < nprime:
                                            epsilonanihiuplist.append(statej[fockindex])
                                        #if fockindex == 0 and nprime == 0:
                                            #epsilonanihiuplist.append(0.0)
                                    epsilonanihiup = sum(epsilonanihiuplist)
                                    statej[nprime] = 0
                                    if statej[n] == 0:
                                        epsiloncreatuplist = [Numberbotstates]
                                        for fockindex in range(0, int(len(statej)/2), 1):
                                            if fockindex < n:
                                                epsiloncreatuplist.append(statej[fockindex])
                                            #if fockindex == 0 and n == 0:
                                                #epsiloncreatuplist.append(0.0)
                                        epsiloncreatup = sum(epsiloncreatuplist)
                                        statej[n] = 1
                                        for iconf in range(len(configurationslist)):
                                            if statej == list(configurationslist[iconf]):
                                                Unnmmlist = []

                                                gncompl = np.conj(vectn[i])
                                                gn = vectnprime[i]
                                                gmcompl = np.conj(vectm[j])
                                                gm = vectmprime[j]
                                                tau = np.conj(np.array(Psi)[0][iconf])
                                                tauprime = np.array(Psi)[0][jconf]
                                                Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                                fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                                jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

        for n in range(0, Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime])[0]
            for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1
                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)



        for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates])[0]
            for nprime in range(Nbot, Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
                for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
                    vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
                    mprime = m
                    vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)

                    if statej[nprime] == 1:
                        epsilonanihiuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < nprime:
                                epsilonanihiuplist.append(statej[fockindex])
                            #if fockindex == 0 and nprime == 0:
                                #epsilonanihiuplist.append(0.0)
                        epsilonanihiup = sum(epsilonanihiuplist)
                        statej[nprime] = 0
                        if statej[n] == 0:
                            epsiloncreatuplist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < n:
                                    epsiloncreatuplist.append(statej[fockindex])
                                #if fockindex == 0 and n == 0:
                                    #epsiloncreatuplist.append(0.0)
                            epsiloncreatup = sum(epsiloncreatuplist)
                            statej[n] = 1
                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)




        for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
            for m in range(0, Numberbotstates, 1):
                vectm = np.array(eig_vects[m])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                for iconf in range(len(configurationslist)):
                    if statej == list(configurationslist[iconf]):
                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                        jlist.append(0.25*sum(Unnmmlist))





        for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for m in range(Nbot, Ntop, 1):
                    vectm = np.array(eig_vects[m+Numberbotstates])[0]
                    for mprime in range(Nbot, Ntop, 1):
                        vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                        statej = []
                        for obj in configurationslist[jconf]:
                            statej.append(obj)
                        if statej[mprime] == 1:
                            epsilonanihidownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < mprime:
                                    epsilonanihidownlist.append(statej[fockindex])
                                #if fockindex == 0 and mprime == 0:
                                    #epsilonanihidownlist.append(0.0)
                            epsilonanihidown = sum(epsilonanihidownlist)
                            statej[mprime] = 0
                            if statej[m] == 0:
                                epsiloncreatdownlist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)/2), 1):
                                    if fockindex < m:
                                        epsiloncreatdownlist.append(statej[fockindex])
                                    #if fockindex == 0 and m == 0:
                                        #epsiloncreatdownlist.append(0.0)
                                epsiloncreatdown = sum(epsiloncreatdownlist)
                                statej[m] = 1
                                if statej[nprime] == 1:
                                    epsilonanihiuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)), 1):
                                        if fockindex < nprime:
                                            epsilonanihiuplist.append(statej[fockindex])
                                        #if fockindex == 0 and nprime == 0:
                                            #epsilonanihiuplist.append(0.0)
                                    epsilonanihiup = sum(epsilonanihiuplist)
                                    statej[nprime] = 0
                                    if statej[n] == 0:
                                        epsiloncreatuplist = [Numberbotstates]
                                        for fockindex in range(0, int(len(statej)), 1):
                                            if fockindex < n:
                                                epsiloncreatuplist.append(statej[fockindex])
                                            #if fockindex == 0 and n == 0:
                                                #epsiloncreatuplist.append(0.0)
                                        epsiloncreatup = sum(epsiloncreatuplist)
                                        statej[n] = 1
                                        for iconf in range(len(configurationslist)):
                                            if statej == list(configurationslist[iconf]):
                                                Unnmmlist = []

                                                gncompl = np.conj(vectn[i])
                                                gn = vectnprime[i]
                                                gmcompl = np.conj(vectm[j])
                                                gm = vectmprime[j]
                                                tau = np.conj(np.array(Psi)[0][iconf])
                                                tauprime = np.array(Psi)[0][jconf]
                                                Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                                fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                                jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

        for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for m in range(0, Numberbotstates, 1):
                    vectm = np.array(eig_vects[m])[0]
                    mprime = m
                    vectmprime = np.array(eig_vects[mprime])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)

                    if statej[nprime] == 1:
                        epsilonanihiuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < nprime:
                                epsilonanihiuplist.append(statej[fockindex])
                            #if fockindex == 0 and nprime == 0:
                                #epsilonanihiuplist.append(0.0)
                        epsilonanihiup = sum(epsilonanihiuplist)
                        statej[nprime] = 0
                        if statej[n] == 0:
                            epsiloncreatuplist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < n:
                                    epsiloncreatuplist.append(statej[fockindex])
                                #if fockindex == 0 and n == 0:
                                    #epsiloncreatuplist.append(0.0)
                            epsiloncreatup = sum(epsiloncreatuplist)
                            statej[n] = 1
                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

        for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
            vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
            nprime = n
            vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
            for m in range(Nbot, Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates])[0]
                for mprime in range(Nbot, Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1
                            for iconf in range(len(configurationslist)):
                                if statej == list(configurationslist[iconf]):
                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)


        #print (jlist, 'jlist')
        maplist.append(sum(jlist))
        #save_representation([sum(jlist)], [''], 'corrmap')
        #print(maplist)

    return maplist


def correlator_map_improved_paral(pos, eig_vects, Psi, configurationslist, i, zeromodes, Nbot, Ntop,j,jconf):
    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot
    Nbot = 0
    Ntop = int(len(configurationslist[0])/2)
    maplist = []
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations

    print (j,'map')
    jlist = []


    print(str(jconf) + ','+str(len(configurationslist)))
    save_representation([str(jconf) + ','+str(len(configurationslist))], [''], 'checks')
    for n in range(0, Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime])[0]
        for m in range(0, Numberbotstates, 1):
            vectm = np.array(eig_vects[m])[0]
            mprime = m
            vectmprime = np.array(eig_vects[mprime])[0]
            statej = []
            for obj in configurationslist[jconf]:
                statej.append(obj)

            counter = 1
            dec = 0.0
            conf = statej
            for y in conf:
                dec = dec + y/(10**counter)
                counter += 1
            iconf = confdic[str(dec)]

            Unnmmlist = []

            gncompl = np.conj(vectn[i])
            gn = vectnprime[i]
            gmcompl = np.conj(vectm[j])
            gm = vectmprime[j]
            tau = np.conj(np.array(Psi)[0][iconf])
            tauprime = np.array(Psi)[0][jconf]
            Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
            jlist.append(0.25*sum(Unnmmlist))





    for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates])[0]
        for nprime in range(Nbot, Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
            for m in range(Nbot, Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates])[0]
                for mprime in range(Nbot, Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1
                            if statej[nprime] == 1:
                                epsilonanihiuplist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)/2), 1):
                                    if fockindex < nprime:
                                        epsilonanihiuplist.append(statej[fockindex])
                                    #if fockindex == 0 and nprime == 0:
                                        #epsilonanihiuplist.append(0.0)
                                epsilonanihiup = sum(epsilonanihiuplist)
                                statej[nprime] = 0
                                if statej[n] == 0:
                                    epsiloncreatuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)/2), 1):
                                        if fockindex < n:
                                            epsiloncreatuplist.append(statej[fockindex])
                                        #if fockindex == 0 and n == 0:
                                            #epsiloncreatuplist.append(0.0)
                                    epsiloncreatup = sum(epsiloncreatuplist)
                                    statej[n] = 1

                                    counter = 1
                                    dec = 0.0
                                    conf = statej
                                    for y in conf:
                                        dec = dec + y/(10**counter)
                                        counter += 1
                                    iconf = confdic[str(dec)]

                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)


    for n in range(0,Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime])[0]
        for m in range(Nbot, Ntop, 1):
            vectm = np.array(eig_vects[m+Numberbotstates])[0]
            for mprime in range(Nbot, Ntop, 1):
                vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                if statej[mprime] == 1:
                    epsilonanihidownlist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)/2), 1):
                        if fockindex < mprime:
                            epsilonanihidownlist.append(statej[fockindex])
                        #if fockindex == 0 and mprime == 0:
                            #epsilonanihidownlist.append(0.0)
                    epsilonanihidown = sum(epsilonanihidownlist)
                    statej[mprime] = 0
                    if statej[m] == 0:
                        epsiloncreatdownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < m:
                                epsiloncreatdownlist.append(statej[fockindex])
                            #if fockindex == 0 and m == 0:
                                #epsiloncreatdownlist.append(0.0)
                        epsiloncreatdown = sum(epsiloncreatdownlist)
                        statej[m] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)




    for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates])[0]
        for nprime in range(Nbot, Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
            for m in range(0,Numberbotstates, 1):
                vectm = np.array(eig_vects[m])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)

                if statej[nprime] == 1:
                    epsilonanihiuplist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)/2), 1):
                        if fockindex < nprime:
                            epsilonanihiuplist.append(statej[fockindex])
                        #if fockindex == 0 and nprime == 0:
                            #epsilonanihiuplist.append(0.0)
                    epsilonanihiup = sum(epsilonanihiuplist)
                    statej[nprime] = 0
                    if statej[n] == 0:
                        epsiloncreatuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < n:
                                epsiloncreatuplist.append(statej[fockindex])
                            #if fockindex == 0 and n == 0:
                                #epsiloncreatuplist.append(0.0)
                        epsiloncreatup = sum(epsiloncreatuplist)
                        statej[n] = 1


                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)



    for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
        for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
            vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
            mprime = m
            vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
            statej = []
            for obj in configurationslist[jconf]:
                statej.append(obj)

            counter = 1
            dec = 0.0
            conf = statej
            for y in conf:
                dec = dec + y/(10**counter)
                counter += 1
            iconf = confdic[str(dec)]

            Unnmmlist = []
            gncompl = np.conj(vectn[i])
            gn = vectnprime[i]
            gmcompl = np.conj(vectm[j])
            gm = vectmprime[j]
            tau = np.conj(np.array(Psi)[0][iconf])
            tauprime = np.array(Psi)[0][jconf]
            Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
            jlist.append(0.25*sum(Unnmmlist))


    for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
        for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1
                            if statej[nprime] == 1:
                                epsilonanihiuplist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)), 1):
                                    if fockindex < nprime:
                                        epsilonanihiuplist.append(statej[fockindex])
                                    #if fockindex == 0 and nprime == 0:
                                        #epsilonanihiuplist.append(0.0)
                                epsilonanihiup = sum(epsilonanihiuplist)
                                statej[nprime] = 0
                                if statej[n] == 0:
                                    epsiloncreatuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)), 1):
                                        if fockindex < n:
                                            epsiloncreatuplist.append(statej[fockindex])
                                        #if fockindex == 0 and n == 0:
                                            #epsiloncreatuplist.append(0.0)
                                    epsiloncreatup = sum(epsiloncreatuplist)
                                    statej[n] = 1


                                    counter = 1
                                    dec = 0.0
                                    conf = statej
                                    for y in conf:
                                        dec = dec + y/(10**counter)
                                        counter += 1
                                    iconf = confdic[str(dec)]

                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

    for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
        for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
            vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                if statej[mprime] == 1:
                    epsilonanihidownlist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)), 1):
                        if fockindex < mprime:
                            epsilonanihidownlist.append(statej[fockindex])
                        #if fockindex == 0 and mprime == 0:
                            #epsilonanihidownlist.append(0.0)
                    epsilonanihidown = sum(epsilonanihidownlist)
                    statej[mprime] = 0
                    if statej[m] == 0:
                        epsiloncreatdownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < m:
                                epsiloncreatdownlist.append(statej[fockindex])
                            #if fockindex == 0 and m == 0:
                                #epsiloncreatdownlist.append(0.0)
                        epsiloncreatdown = sum(epsiloncreatdownlist)
                        statej[m] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)


    for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
        for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
                vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)

                if statej[nprime] == 1:
                    epsilonanihiuplist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)), 1):
                        if fockindex < nprime:
                            epsilonanihiuplist.append(statej[fockindex])
                        #if fockindex == 0 and nprime == 0:
                            #epsilonanihiuplist.append(0.0)
                    epsilonanihiup = sum(epsilonanihiuplist)
                    statej[nprime] = 0
                    if statej[n] == 0:
                        epsiloncreatuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < n:
                                epsiloncreatuplist.append(statej[fockindex])
                            #if fockindex == 0 and n == 0:
                                #epsiloncreatuplist.append(0.0)
                        epsiloncreatup = sum(epsiloncreatuplist)
                        statej[n] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)




    for n in range(0, Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime])[0]
        for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
            vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
            mprime = m
            vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
            statej = []
            for obj in configurationslist[jconf]:
                statej.append(obj)

            counter = 1
            dec = 0.0
            conf = statej
            for y in conf:
                dec = dec + y/(10**counter)
                counter += 1
            iconf = confdic[str(dec)]

            Unnmmlist = []

            gncompl = np.conj(vectn[i])
            gn = vectnprime[i]
            gmcompl = np.conj(vectm[j])
            gm = vectmprime[j]
            tau = np.conj(np.array(Psi)[0][iconf])
            tauprime = np.array(Psi)[0][jconf]
            Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
            jlist.append(0.25*sum(Unnmmlist))




    for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates])[0]
        for nprime in range(Nbot, Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
            for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1
                            if statej[nprime] == 1:
                                epsilonanihiuplist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)/2), 1):
                                    if fockindex < nprime:
                                        epsilonanihiuplist.append(statej[fockindex])
                                    #if fockindex == 0 and nprime == 0:
                                        #epsilonanihiuplist.append(0.0)
                                epsilonanihiup = sum(epsilonanihiuplist)
                                statej[nprime] = 0
                                if statej[n] == 0:
                                    epsiloncreatuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)/2), 1):
                                        if fockindex < n:
                                            epsiloncreatuplist.append(statej[fockindex])
                                        #if fockindex == 0 and n == 0:
                                            #epsiloncreatuplist.append(0.0)
                                    epsiloncreatup = sum(epsiloncreatuplist)
                                    statej[n] = 1

                                    counter = 1
                                    dec = 0.0
                                    conf = statej
                                    for y in conf:
                                        dec = dec + y/(10**counter)
                                        counter += 1
                                    iconf = confdic[str(dec)]

                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

    for n in range(0, Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime])[0]
        for m in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
            vectm = np.array(eig_vects[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for mprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
                vectmprime = np.array(eig_vects[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                if statej[mprime] == 1:
                    epsilonanihidownlist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)), 1):
                        if fockindex < mprime:
                            epsilonanihidownlist.append(statej[fockindex])
                        #if fockindex == 0 and mprime == 0:
                            #epsilonanihidownlist.append(0.0)
                    epsilonanihidown = sum(epsilonanihidownlist)
                    statej[mprime] = 0
                    if statej[m] == 0:
                        epsiloncreatdownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < m:
                                epsiloncreatdownlist.append(statej[fockindex])
                            #if fockindex == 0 and m == 0:
                                #epsiloncreatdownlist.append(0.0)
                        epsiloncreatdown = sum(epsiloncreatdownlist)
                        statej[m] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)



    for n in range(Nbot, Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates])[0]
        for nprime in range(Nbot, Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates])[0]
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):
                vectm = np.array(eig_vects[m-int(len(configurationslist[jconf])/2)])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime-int(len(configurationslist[jconf])/2)])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)

                if statej[nprime] == 1:
                    epsilonanihiuplist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)/2), 1):
                        if fockindex < nprime:
                            epsilonanihiuplist.append(statej[fockindex])
                        #if fockindex == 0 and nprime == 0:
                            #epsilonanihiuplist.append(0.0)
                    epsilonanihiup = sum(epsilonanihiuplist)
                    statej[nprime] = 0
                    if statej[n] == 0:
                        epsiloncreatuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < n:
                                epsiloncreatuplist.append(statej[fockindex])
                            #if fockindex == 0 and n == 0:
                                #epsiloncreatuplist.append(0.0)
                        epsiloncreatup = sum(epsiloncreatuplist)
                        statej[n] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)




    for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
        for m in range(0, Numberbotstates, 1):
            vectm = np.array(eig_vects[m])[0]
            mprime = m
            vectmprime = np.array(eig_vects[mprime])[0]
            statej = []
            for obj in configurationslist[jconf]:
                statej.append(obj)

            counter = 1
            dec = 0.0
            conf = statej
            for y in conf:
                dec = dec + y/(10**counter)
                counter += 1
            iconf = confdic[str(dec)]

            Unnmmlist = []

            gncompl = np.conj(vectn[i])
            gn = vectnprime[i]
            gmcompl = np.conj(vectm[j])
            gm = vectmprime[j]
            tau = np.conj(np.array(Psi)[0][iconf])
            tauprime = np.array(Psi)[0][jconf]
            Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
            jlist.append(0.25*sum(Unnmmlist))





    for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
        for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for m in range(Nbot, Ntop, 1):
                vectm = np.array(eig_vects[m+Numberbotstates])[0]
                for mprime in range(Nbot, Ntop, 1):
                    vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1:
                        epsilonanihidownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            #if fockindex == 0 and mprime == 0:
                                #epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0
                        if statej[m] == 0:
                            epsiloncreatdownlist = [Numberbotstates]
                            for fockindex in range(0, int(len(statej)/2), 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                #if fockindex == 0 and m == 0:
                                    #epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1
                            if statej[nprime] == 1:
                                epsilonanihiuplist = [Numberbotstates]
                                for fockindex in range(0, int(len(statej)), 1):
                                    if fockindex < nprime:
                                        epsilonanihiuplist.append(statej[fockindex])
                                    #if fockindex == 0 and nprime == 0:
                                        #epsilonanihiuplist.append(0.0)
                                epsilonanihiup = sum(epsilonanihiuplist)
                                statej[nprime] = 0
                                if statej[n] == 0:
                                    epsiloncreatuplist = [Numberbotstates]
                                    for fockindex in range(0, int(len(statej)), 1):
                                        if fockindex < n:
                                            epsiloncreatuplist.append(statej[fockindex])
                                        #if fockindex == 0 and n == 0:
                                            #epsiloncreatuplist.append(0.0)
                                    epsiloncreatup = sum(epsiloncreatuplist)
                                    statej[n] = 1

                                    counter = 1
                                    dec = 0.0
                                    conf = statej
                                    for y in conf:
                                        dec = dec + y/(10**counter)
                                        counter += 1
                                    iconf = confdic[str(dec)]

                                    Unnmmlist = []

                                    gncompl = np.conj(vectn[i])
                                    gn = vectnprime[i]
                                    gmcompl = np.conj(vectm[j])
                                    gm = vectmprime[j]
                                    tau = np.conj(np.array(Psi)[0][iconf])
                                    tauprime = np.array(Psi)[0][jconf]
                                    Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                                    jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

    for n in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
        for nprime in range(int(len(configurationslist[jconf])/2)+Nbot, int(len(configurationslist[jconf])/2) + Ntop, 1):
            vectnprime = np.array(eig_vects[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]
            for m in range(0, Numberbotstates, 1):
                vectm = np.array(eig_vects[m])[0]
                mprime = m
                vectmprime = np.array(eig_vects[mprime])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)

                if statej[nprime] == 1:
                    epsilonanihiuplist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)), 1):
                        if fockindex < nprime:
                            epsilonanihiuplist.append(statej[fockindex])
                        #if fockindex == 0 and nprime == 0:
                            #epsilonanihiuplist.append(0.0)
                    epsilonanihiup = sum(epsilonanihiuplist)
                    statej[nprime] = 0
                    if statej[n] == 0:
                        epsiloncreatuplist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)), 1):
                            if fockindex < n:
                                epsiloncreatuplist.append(statej[fockindex])
                            #if fockindex == 0 and n == 0:
                                #epsiloncreatuplist.append(0.0)
                        epsiloncreatup = sum(epsiloncreatuplist)
                        statej[n] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)

    for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + Numberbotstates, 1):               #This is for the Active Space states
        vectn = np.array(eig_vects[n-int(len(configurationslist[jconf])/2)])[0]
        nprime = n
        vectnprime = np.array(eig_vects[nprime-int(len(configurationslist[jconf])/2)])[0]
        for m in range(Nbot, Ntop, 1):
            vectm = np.array(eig_vects[m+Numberbotstates])[0]
            for mprime in range(Nbot, Ntop, 1):
                vectmprime = np.array(eig_vects[mprime+Numberbotstates])[0]
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                if statej[mprime] == 1:
                    epsilonanihidownlist = [Numberbotstates]
                    for fockindex in range(0, int(len(statej)/2), 1):
                        if fockindex < mprime:
                            epsilonanihidownlist.append(statej[fockindex])
                        #if fockindex == 0 and mprime == 0:
                            #epsilonanihidownlist.append(0.0)
                    epsilonanihidown = sum(epsilonanihidownlist)
                    statej[mprime] = 0
                    if statej[m] == 0:
                        epsiloncreatdownlist = [Numberbotstates]
                        for fockindex in range(0, int(len(statej)/2), 1):
                            if fockindex < m:
                                epsiloncreatdownlist.append(statej[fockindex])
                            #if fockindex == 0 and m == 0:
                                #epsiloncreatdownlist.append(0.0)
                        epsiloncreatdown = sum(epsiloncreatdownlist)
                        statej[m] = 1

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = []

                        gncompl = np.conj(vectn[i])
                        gn = vectnprime[i]
                        gmcompl = np.conj(vectm[j])
                        gm = vectmprime[j]
                        tau = np.conj(np.array(Psi)[0][iconf])
                        tauprime = np.array(Psi)[0][jconf]
                        Unnmmlist.append(-1*gncompl*gn*gmcompl*gm*tau*tauprime)
                        fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown)
                        jlist.append(0.25*sum(Unnmmlist)*fermionic_sign)


    #print (jlist, 'jlist')
    maplist.append(sum(jlist))
    #save_representation([sum(jlist)], [''], 'corrmap')
    #print(maplist)

    return sum(jlist)



def calcvec(n):
    """
    this function calculates the Bravais vectors for the fused triangulene
    n: integer number that sets how much larger is the unit cell.
    n=7 for the S=1 triangulenes
    """
    a=2.42
    x1=a*np.cos(np.pi/6.)
    y1=a*np.sin(np.pi/6.)
    x2=x1
    y2=-y1

    vec1=np.array([n*x1,n*y1,0])
    vec2=np.array([n*x2,n*y2,0])


    return vec1,vec2

def fused(nrows):
        """this function generates 2 face to face triangulene
           generated with triangulene function
        """
        aCC=2.42/np.sqrt(3.)
        list1=triangulene(nrows)
        nsites1=len(list1)
        nsites=2*nsites1
        ymin=list1[nsites1-1][1]

#       We replicate the triangle:
#       shit all y to y-ymin+aCC/2 Therefore, the lowest tip at y=aCC/2
#       reversing y to -y
        list2=[]

        for k in range(nsites1):
                x,y,z=list1[k]
                yp=y-ymin+0.5*aCC
                list2.append((x,yp,z))
                list2.append((x,-yp,z))

        return list2

def vects_triangullenes(edgelength,a):
    """This function gives the vectors of the atoms for triangular graphene flakes.
    Inputs:
    Edgelength is an integer. It is the number of atoms of the edge.
    a is the bondlength
    Returns a list with the vectors for the positions of the atoms as a list of 3D arrays.
    """
    #The next 27 lines of code create the S=1/2 triangulene position vectors
    pos=[]
    vI = np.array([0.0, 0.0, 0.0])
    vII = np.array([np.sqrt(3)*a/2, a/2, 0.0])
    vIII = np.array([np.sqrt(3)*a, 0.0, 0.0])
    vIV = np.array([np.sqrt(3)*a, -a, 0.0])
    vV = np.array([np.sqrt(3)*a/2, -3*a/2, 0.0])
    vVI = np.array([0.0, -a, 0.0])
    vVII = np.array([-np.sqrt(3)*a/2, a/2, 0.0])
    vVIII = np.array([-np.sqrt(3)*a, 0.0, 0.0])
    vIX = np.array([-np.sqrt(3)*a, -a, 0.0])
    vX = np.array([-np.sqrt(3)*a/2, -3*a/2, 0.0])
    vXI = np.array([np.sqrt(3)*a/2, 3*a/2, 0.0])
    vXII = np.array([0.0, 2.8, 0.0])
    vXIII = np.array([-np.sqrt(3)*a/2, 3*a/2, 0.0])

    pos.append(vI)
    pos.append(vII)
    pos.append(vIII)
    pos.append(vIV)
    pos.append(vV)
    pos.append(vVI)
    pos.append(vVII)
    pos.append(vVIII)
    pos.append(vIX)
    pos.append(vX)
    pos.append(vXI)
    pos.append(vXII)
    pos.append(vXIII)

    #The remaining lines of code make a repeating unit and scalates the triangulene
    cellpos = []

    cellpos.append(vII)
    cellpos.append(vIII)
    cellpos.append(vIV)
    cellpos.append(vV)
    cellpos.append(vXI)
    cellpos.append(vXII)

    countloopi = 1
    countx = 1
    while True:
        if edgelength == 2:
            break
        for i in cellpos:

            vn = np.array([i[0] + (countx * np.sqrt(3)*a), i[1], i[2]])
            pos.append(vn)
            if countloopi % (6+2*(countx-1)) == 0:

                vn = pos[len(pos)-1]
                pos.append(np.array([vn[0], vn[1]+a, vn[2]]))
                pos.append(np.array([vn[0]-np.sqrt(3)*a/2, vn[1]+3*a/2, vn[2]]))
                pos.append(np.array([vn[0]-np.sqrt(3)*a, vn[1]+a, vn[2]]))
                cellpos.append(np.array([vn[0]-(np.sqrt(3)*a*countx), vn[1]+a, vn[2]]))
                cellpos.append(np.array([vn[0]-np.sqrt(3)*a/2-(np.sqrt(3)*a*countx), vn[1]+3*a/2, vn[2]]))

                countx += 1
                countloopi = 1
                break
            countloopi += 1
        if countx == edgelength-1:
            break

    return pos

def triangulene(nrows):
    a=2.42
    a1=(0.5*a,0.5*a*np.sqrt(3.0))
    a2=(-0.5*a,0.5*a*np.sqrt(3.0))

    t=(0.5*a,0.5*a/np.sqrt(3.0))
    print('a1=',a1)
    print('a2=',a2)
    print('t=',t)
    natom_row=nrows

    xi=0.0
    yi=0.0
    r0=[]
    for i in range(1,nrows+1):
        x=xi
        y=yi+t[1]-a1[1]
        r0.append((x,y,0))
        for j in range(1,natom_row+1):
            x=xi+(j-1)*a
            y=yi
            r0.append((x,y,0))
            r0.append((x+t[0],y+t[1],0.))

        x=x+a
        y=y
        r0.append((x,y,0))
        y=y-2.0*t[1]
        r0.append((x,y,0))
        natom_row=natom_row-1
        xi=xi+a1[0]
        yi=yi-a1[1]


    x=x-t[0]
    y=y-t[1]
    r0.append((x,y,0))
    print(' number of atoms=',len(r0))
    return r0

def shift(atoms,vec):
    """ inputs:
        list of atoms
        displacement vector

        output: displaced list
    """
    list1=[]
    n=len(atoms)
    for k in range(n):
                x=vec[0]+atoms[k][0]
                y=vec[1]+atoms[k][1]
                list1.append((x,y,0))

    return list1

def chain(nt,nreplic):
    """ generates cluster with several fused triangulenes
    nt: size of triagulene
    nd: size of unit cell replication vector (7 for nt=2)
    nreplic: number of triangulenes in the chain
    it returns a list of position vectors (here 3D arrays)
    """
    if nreplic%2 == 0:
        newnreplic = int(nreplic/2)
    else:
        newnreplic = nreplic
    pos = []
    nd=2*nt+1
    vec1,vec2=calcvec(nd)

    atom0=fused(nt)
#    rotate
    atom1=[]
    cluster=[]
    unit=[]
    for k in range(len(atom0)):
        x=atom0[k][0]
        y=atom0[k][1]
        atom1.append((y,x,0))  # this rotates the fused trianguelen

    for n1 in range(newnreplic):
        vec=n1*vec1
        dis=shift(atom1,vec)
        for k in range(len(atom1)):
            x=dis[k][0]
            y=dis[k][1]
            cluster.append((x,y,0))

    if nreplic%2 != 0:
        for i in range(len(cluster)):
            if cluster[i][0]<21*0.7*(nreplic-1)*0.5:
                pos.append(np.array([cluster[i][0],cluster[i][1],cluster[i][2]]))
    if len(pos) == 0:
        for i in range(len(cluster)):
            pos.append(np.array([cluster[i][0],cluster[i][1],cluster[i][2]]))

    return pos

def bondpos_graph(pos, a):
    """Gives a list with the position of the bonds in planar systems.
    Inputs:
    pos is a list with the vectors of the lattice
    a is the bondlength
    Returns a list with the position of the bonds for those sites at a distance a
    """
    bondpos = [] #empty list
    for i in range(len(pos)): #loops over sites
        ri = pos[i] #this is a position vector
        for j in range(len(pos)): #loops over sites again
            rj = pos[j] #this is a position vector
            dist = np.linalg.norm(rj - ri) #distance between 2 vectors
            if (abs(dist - a) < 0.1): #if that distance is equal to the bondlenght, then
                bondpos.append([ri[0], rj[0], ri[1], rj[1]]) #append a bond in bondpos
    return bondpos

def plot_lattice(pos, bondpos, fname='lattice.png',show=True):
    """This function plots the lattice.
    Inputs:
    pos is a list with the vectors of the lattice
    bondpos is a list with the positions of the bonds
    """
    X,Y,Z = [],[],[] #empty lists
    for i in range(len(pos)): #loops over the sites
        r = pos[i]     # positions
        X.append(r[0])  #these 3 lines are the cartesian coordinates of r
        Y.append(r[1])
        Z.append(r[2])
    fig = plt.figure() #it creates the figure
    ax = fig.add_subplot(111) #it creates the subplot
    ax.set_xlabel("x($\AA$)") #label x
    ax.set_ylabel("y($\AA$)") #label y
    #Next 2 lines plot the bonds
    for bond in bondpos:
        ax.plot(bond[0:2], bond[-2:], 'k-')
    for i in range(len(pos)):
        plt.text(X[i],Y[i],i, ha="center", va="center", weight='bold') # Plots the lattice with numbers
    ax.scatter(X,Y,c='b',s=20.0,edgecolors='none') # Plots the lattice. c is the colour of the dots, s is the size.
    plt.gca().set_aspect('equal', adjustable='box') #this avoids a distortion in the picture

    plt.show()

def rotational_matrix3DZaxis(vects, degree):
    """This function gets some vects and it returns them rotated by a given degree in the Z axis
    """
    Rmatrix = np.matrix(np.zeros(((3,3)),dtype=np.complex_))
    Rmatrix[0,0] = np.cos(degree)
    Rmatrix[0,1] = -np.sin(degree)
    Rmatrix[1,0] = np.sin(degree)
    Rmatrix[1,1] = np.cos(degree)
    Rmatrix[2,2] = 1
    rotvects = []
    for atom in vects:
        rotvect =(np.dot(Rmatrix,np.array([atom[0], atom[1], atom[2]]).transpose()))
        rotvect = np.array(rotvect)[0]

        rotvects.append(np.array([rotvect[0], rotvect[1], rotvect[2]]))
    #print (rotvects)
    return rotvects

def plot_espectrum(t, eigvals):
    """plots an eigenvalue list (the espectrum of energies of the system)
    Inputs:
    t is the hopping parameter
    eigvals is the list of eigenvalues you want to plot
    """
    n = len(eigvals) #n is the length of the eigenvalue list
    eigenvalue_numberlist = [] #this is a list for the state number
    for i in range(len(eigvals)):  #loops over the eigenvalues
        eigenvalue_numberlist.append(i) #it appends the state number to the list

    fig = plt.figure() #it creates the figure
    ax = fig.add_subplot(111) #it creates the subplot
    if t == 1 or t == -1: #if the values are in units of t
        ax.set_ylabel("E(t)") #the y label is E(t)
    if t != 1 and t != -1: #otherwise
        ax.set_ylabel("E(eV)") #the y label is E(eV)
    ax.set_xlabel("State number") #the x label is "State number"
    ax.scatter(eigenvalue_numberlist, eigvals, c = 'b',s=50., edgecolors='none') #here it goes the values that will be plot, colour, size of dots.

    plt.show()

def plot_wavefunction(pos, eigvects,  bondpos, fname='lattice.png',show=True):
    """plots the lattice and the wavefunction of a selected eigenvalue
    inputs:
    pos is the list of position vectors
    eigvects is a list with the eigenvectors you want toi plot
    bondpos is the list the bonds
    """
    eigvects = np.array(eigvects)[0]
    X, Y, Z, S = [], [], [], [] #these are empty lists
    for i in range(len(pos)): #it loops through the sites
        r = pos[i]     # this is a position
        v = eigvects[i]    # eigvector component corresponding to atom i
        X.append(r[0]) #x coordinate
        Y.append(r[1]) #y coordinate
        Z.append(r[2]) #z coordinate
        conjvect = (abs(v*np.conj(v))) #square of the coefficient
        S.append(conjvect*6000) #it appends the coefficient to the list S, it is multiplied by a factor that you should choose in order for the circles to be visible in the plot

    fig = plt.figure(figsize = (20,10)) #a figure
    ax = fig.add_subplot(111) #the subplot
    ax.axis('off')
    ax.set_xlabel("x($\AA$)") #x label
    ax.set_ylabel("y($\AA$)") #y label
    font = {'size'   : 20} #size of the font
    plt.rc('font', **font)

    for bond in bondpos:    #this plots the bonds
        ax.plot(bond[0:2], bond[-2:], 'k-')

    ax.scatter(X,Y,c='r',s=S,edgecolors='none', alpha = 1) #set the values to plot, colour, etc
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def build_hamtnospin(pos, t, a):
    """builds a first neighbours tight-binding hamiltonian, with just one orbital per site, spinless, and the same hopping for all nearest-neighbours.
    Inputs:
    pos is a list with the vectors of the lattice
    t is the hopping parameter
    a is the bondlength
    returns a nxn matrix
    """
    n = len(pos)         #This is the dimension of the Hilbert Space
    ht = np.matrix(np.zeros((n,n)))        #This is the Hamiltonian with just 0's
    for i in range(n):         #A loop that runs over all the positions
        ri = pos[i]        #This is the vector for position i
        for j in range(n):     #A loop that runs again over all the positions
            rj = pos[j]      #This is the vector for position j
            dist = np.linalg.norm(rj - ri)   #This is the distance between ri and rj
            if (abs(dist - a) < 0.1):    #If the distance between ri and rj is close to a with a tolerance of 0.1 then...
                ht[i,j] = t         #The entrace in the row i and column j gets the value t
    return ht

def solve_h(h):
    """diagonalizes the matrix h
    Inputs:
    h is a matrix, here presumably it'll be a hamiltonian
    returns the eigenvalues in a list (eig_vals) and the eigenvectors in a matrix (eig_vects). In eig_vects rows label the orbital, columns the sites.
    """
    eig_vals, eig_vects = LA.linalg.eigh(h)     #Gets the eigenvalues and eigenvectors
    eig_vects = eig_vects.transpose()     #It transposes the eigenvectors
    return eig_vals, eig_vects


def CAS_better(No,Ne):
    """gets a list of the Fock states that conform the many body Hilbert Space in a molecular basis
    Inputs:
    No is the number of molecular orbitals in the CAS(Ne,No) notation, in the following it is multiplied by 2 for spin degeneracy
    Ne is the number of electrons in the system
    It returns a list with the Fock states
    """

    AllFockStatesList = list(itertools.product([0, 1], repeat=2*No)) #It gives back a list of tuples with the Fock states from  0 electrons to 2xNo electrons, in the following we will just keep those with the desired number of electrons
    configurationslist = [] #It creates an empty list that will gather the Fock states
    #The following piece of code will filter the states with just Ne electrons
    for i in range(len(AllFockStatesList)): #It runs over all the Fock states
        Fockstate1 = AllFockStatesList[i] #It gets an i Fock state
        Fockstate2 = []   #It creates an empty Fock state 2
        for l in range(2*No):   #This loop creates a copy out of it
            Fockstate2.append(Fockstate1[l])
        if sum(Fockstate2) == Ne:   #If that state has a number of electrons equal to Ne then it goes in the returned list
            configurationslist.append(Fockstate2)
    return configurationslist

def loopUHam(configurationslistfirst, Hamtdic, HamUdic, U):
    #print (U)

    Sztotallist = MB_Sz(len(configurationslistfirst[0]), configurationslistfirst)
    spin_ordered_state_lists = Spin_ordered(configurationslistfirst, Sztotallist)

    eig_vals0list = []
    eig_vects0mat = np.matrix(np.zeros(((len(configurationslistfirst), len(configurationslistfirst))),dtype=np.complex_))

    countern = 0

    for x in spin_ordered_state_lists:
    #for x in ['1.0']:
        configurationslist0 = spin_ordered_state_lists[x]
        tHamil0 = Hamtdic[x]
        UHamil0 = HamUdic[x]*U

        Hamil0 = tHamil0 + UHamil0
        eig_vals0, eig_vects0 = LA.linalg.eigh(Hamil0)
        eig_vects0 = eig_vects0.transpose()

        #if x == '0.0':
            #corrfunct = correlation_function_2zeromodes(np.array(eig_vects0[0])[0], eig_valsSP, eig_vals0, configurationslist0, zeromodes, Nbot, Ntop, 'yes')
            #corrfunct = correlation_function(np.array(eig_vects0[0])[0], eig_valsSP, eig_vals0, configurationslist0, zeromodes, Nbot, Ntop, 'yes')
            #corrfunct = correlation_function_1zeromode_1_quasi(np.array(eig_vects0[0])[0], eig_valsSP, eig_vals0, configurationslist0, zeromodes, Nbot, Ntop, 'yes', 'B')
            #save_representation([corrfunct], [U], 'corrfunct')


        for l in eig_vals0:
            eig_vals0list.append(l)


        counternlist = []
        for n in range(len(eig_vals0)):
            for c in range(len(eig_vals0)):
                eig_vects0mat[n+countern,c+countern] = np.array(eig_vects0[n])[0][c]
            counternlist.append(1)
        countern += sum(counternlist)





    eig_vals0list.sort()
    eig_vals0listordered = eig_vals0list
    #print (eig_vals0listordered)

    #eig_vects0matordered = eig_vects0matç
    eig_vects0matordered = 0
    #eig_vals0listordered, eig_vects0matordered = order_eigvals_eigvects(eig_vals0list, eig_vects0mat)

    return eig_vals0listordered, eig_vects0matordered


def solve_Ham_bySz(configurationslistfull, eig_valsSP, eig_vectsSP, pos, zeromodes, Nbot, Ntop,U,paral,loopU):
    """This function diagonalizes a Hamiltonian by parts labeled with Sz subspaces
    Inputs:
    configurationslistfull is a list with all the Fock states that form part of the Hilbert Space, in this function it will be divided in different lists labeled by Sz
    eig_valsSP are the single-particle eigenvalues
    eig_vectsSP are the single-particle eigenvectors
    pos is the list of vectors position of the atoms
    zeromodes is the number of zero modes in the system
    Nbot is the number of valence orbitals in the Active Space without counting the spin, for instance CAS(2,2) is Nbot = 1, CAS(4,4) is Nbot = 2
    Ntop is the number of conduction orbitals in the Active Space without counting the spin, for instance CAS(2,2) is Ntop = 1, CAS(4,4) is Ntop = 2
    U is the Hubbard integral, it should be a positive number
    returns the eigenvalues, eigenvectors and fock states in the correct order
    """
    Hamtdic = {}
    HamUdic = {}
    if loopU == 'Yes':
        U = 1
    Sztotallist = MB_Sz(len(configurationslistfull[0]), configurationslistfull) #This gets a list with the Sz numbers of the Fock states in configurationslistfull
    print('done')
    spin_ordered_state_lists = Spin_ordered(configurationslistfull, Sztotallist) #This gets a dictionary, classifying the Fock states by the Sz number
    print('done')

    #eig_vects0mat = np.matrix(np.zeros(((len(configurationslistfull), len(configurationslistfull))),dtype=np.complex_))
    countern = 0
    configurationslistnew = []

    eig_valslist = []  #This is the list that will gather the eigenvalues
    for Sz in spin_ordered_state_lists: #It runs over the Sz
        configurationslist = spin_ordered_state_lists[Sz] #It gets the Fock states labeled by Sz
        if paral == 'no':
            tHamil = build_hamt_confbasis_SP(eig_valsSP, pos, configurationslist, zeromodes, Nbot, Ntop) #It gets the hopping many-body Hamiltonian
            UHamil = build_hamU_SPbasis_confbasis_SP(U, eig_vectsSP, pos, configurationslist, zeromodes, Nbot, Ntop) #It gets the Hubbard Hamiltonian
        if paral == 'yes':
            num_cores = 4
            #num_cores = int(os.getenv('OMP_NUM_THREADS'))
            #save_representation([num_cores+'num_cores'], [''], 'checks')
            #UHamilzeros = np.matrix(np.zeros(((len(configurationslist),len(configurationslist))),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
            tHamil = build_hamt_confbasis_SP(eig_valsSP, pos, configurationslist, zeromodes, Nbot, Ntop) #It gets the hopping many-body Hamiltonian
            UHamilold = Parallel(n_jobs=num_cores)(delayed(build_hamU_SPbasis_confbasis_SP_paral)(U, eig_vectsSP, pos, configurationslist, zeromodes, Nbot, Ntop, jconf) for jconf in range(len(configurationslist)))
            print('hey')
            #print(UHamilold)

            UHamil = build_HamUparal(UHamilold,configurationslist)

        if loopU == 'Yes':
            Hamtdic[Sz] = tHamil
            HamUdic[Sz] = UHamil
        Hamil = tHamil + UHamil #The separated Hamiltonians are joined is one matrix
        eig_vals, eig_vects = solve_h(Hamil) #It diagonalizes Hamil

        print('done')
        for l in eig_vals: #It gathers the eigenvalues in a list, so the Sz loop can go on
            eig_valslist.append(l)
        for conf in configurationslist:
            configurationslistnew.append(conf)
        """
        counternlist = []
        for n in range(len(eig_vals)):
            print(n,'n')
            for c in range(len(eig_vals)):
                eig_vects0mat[n+countern,c+countern] = np.array(eig_vects[n])[0][c]
            counternlist.append(1)
        countern += sum(counternlist)
        print('done')
        """
    eig_valslist.sort()
    eig_vals0listordered = eig_valslist
    eig_vects0matordered = eig_vects

    #eig_vals0listordered, eig_vects0matordered = order_eigvals_eigvects(eig_valslist, eig_vects0mat)
    return eig_vals0listordered, eig_vects0matordered, Hamtdic, HamUdic, configurationslistnew

def order_eigvals_eigvects(eig_vals0list, eig_vects0mat):
    eig_vals = []
    eig_vects = np.matrix(np.zeros(((len(eig_vals0list), len(eig_vals0list))),dtype=np.complex_))
    countervect = 0
    while True:
        print (countervect)
        save_representation([str(countervect)], [''], 'checks')
        eig_valsi = min(eig_vals0list)
        i = eig_vals0list.index(eig_valsi)
        for e in range(len(eig_vals0list)):
            eig_vects[countervect,e] = np.array(eig_vects0mat[i])[0][e]
        countervect += 1
        eig_vals.append(eig_valsi)
        eig_vals0list[i] = 100000000000000000000000
        """

        for i in range(len(eig_vals0list)):
            eig_valsi = eig_vals0list[i]
            if eig_valsi != 'None':
                counter = 0
                for j in range(len(eig_vals0list)):
                    eig_valsj = eig_vals0list[j]
                    if eig_valsj != 'None':
                        if eig_valsi > eig_valsj:
                            counter += 1
                            break
                if counter == 0:
                    for e in range(len(eig_vals0list)):
                        eig_vects[countervect,e] = np.array(eig_vects0mat[i])[0][e]
                    countervect += 1
                    eig_vals.append(eig_valsi)
                    eig_vals0list[i] = 'None'
        """

        counterbreak = 0
        for m in eig_vals0list:
            if m != 100000000000000000000000:
                counterbreak += 1
        if counterbreak == 0:
            break

    return eig_vals, eig_vects

def MB_Sz(N, Focklist):
    """This function gets a list with the Sz numbers of the Fock states in configurationslistfull
    Inputs:
    N is the length of a Fock state, which is 2xNo
    Focklist is a list of Fock states
    returns a list with the Sz numbers of the states in Focklist
    """
    middle = int(N/2)  #it is an integer with a value equal to half the length of a configuration
    Sztotallist = [] #this is the list that will gather all the possible Sz numbers in the Fock state list
    for state in Focklist: #it loops over all the Focklist
        Szuplist = [] #a list for the up spins
        Szdownlist = [] #a list for the down spins
        for i in list(range(0, middle, 1)):  #it loops till the middle of the Focklist               #This is for a base in spin blocks |1up,2up....,1down,2down....>
            Szuplist.append(state[i]) #it gathers 1 or 0 in Szuplist, so up spins are counted
        for j in list(range(middle, N, 1)): #it loops the rest of the Focklist
            Szdownlist.append(state[j]) #it gathers 1 or 0 in Szdownlist, so down spins are counted

        #for i in list(range(0,middle)):   #it loops till the middle of the Focklist                  #This is for a Fock base |1up,1down,2up,2down....>
            #Szuplist.append(state[2*i])   #up electrons are gathered
            #Szdownlist.append(state[2*i +1]) #down electrons are gathered

        Szup = sum(Szuplist)  #the sum of Szuplist is the total number of up electrons
        Szdown = sum(Szdownlist) #the sum of Szdownlist is the total number of down electrons
        Sztotal = (Szup - Szdown)/2 #the total Sz number is the difference between up and down electrons divided by 2
        Sztotallist.append(Sztotal) #the Sz number of a certain configuration in Focklist is gathered in Sztotallist
    return Sztotallist

def Spin_ordered(Focklist, Sztotallist):
    """This function gets a dictionary classifying the Fock states by the Sz number
    Inputs:
    Focklist is a list of Fock states
    Sztotallist is a list with the Sz numbers of the states in Focklist
    returns a dictionary that gathers the Fock states classified by their Sz number
    """

    spin_ordered_state_lists = {} #creates the dictionary

    #This creates a list with the possible spins in the system [N/2,N/2 - 1...,-N/2]
    Sztotallist = []
    for i in range(0,sum(Focklist[0])+1):
        spin = 0.5*sum(Focklist[0]) - i
        spin_ordered_state_lists[str(spin)] = []
        Sztotallist.append(str(spin))
    print(Sztotallist)

    #This orders the states in the dictionary according to the spin
    for state_label in list(range(len(Focklist))): #it loops over the Fock states in Focklist
        conf = Focklist[state_label]
        spin = (sum(conf[0:int(len(conf)/2)]) - sum(conf[int(len(conf)/2):len(conf)]))/2
        spin_ordered_state_lists[str(spin)].append(conf) #then it is gathered in the proper dictionary entry

    return spin_ordered_state_lists

def build_hamt_confbasis_SP(eig_valsSP, pos, configurationslist, zeromodes, Nbot, Ntop):
    """This function builds the tight-binding hamiltonian with a configuration (many-body) basis from the single-particle eigenvalues (I mention this in order to distinguish from a spin basis),
    employing the equation H = \sum_{n,\sigma} \epsilon_n c^dagger_{n,\sigma} c_{n,\sigma}
    Inputs:
    eig_valsSP is the list of single-particle eigenvalues
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    zeromodes is the number of zero modes in the system
    Nbot is the number of valence states in the Active Space
    Ntop is the number of conduction states in the Active Space
    returns a matrix which is the tight-binding hamiltonian in the many-body basis
    """

    tHamil = np.matrix(np.zeros(((len(configurationslist),len(configurationslist))),dtype=np.complex_)) #this is the empty (all zeros) tight-binding hamiltonian
    limitup = int(len(configurationslist[0])/2) #this is a label for the last up electron
    limitdown = len(configurationslist[0]) #this is a label for the last down electron

    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot #This is the number of valence states out of the Active Space

    #H can be divided into different chunks. This one corresponds to the frozen valence states, that will correspond to a constant for every configuration
    constantlist= [] #This is just an empty list
    for i in range(0, Numberbotstates, 1): #loops over all the frozen valence states
        constantlist.append(eig_valsSP[i]) #append in the previous list the single-particle eigenvalue
    constant = sum(constantlist)*2 #The sum of that list, times 2 for the spin, will be the previously mentioned constant

    #This part corresponds to the Active Space
    for jconf in range(len(configurationslist)): #loops over every configuration
        print (jconf, 't')
        statej = []  #The next 3 lines makes a copy of the configuration labeled as jconf
        for obj in configurationslist[jconf]:
            statej.append(obj)
        #This part is for the Active Space spin up
        for n in range(0, limitup, 1): #loops over the up states
            if statej[n] == 1: #if it is occupied
                tHamil[jconf,jconf] = tHamil[jconf,jconf] + eig_valsSP[n+Numberbotstates] #then add to the corresponding matrix element the single particle eigenvalue. To understand the label in eig_valsSP it is important to notice that in the configuration there is just the Active Space
        #This part is for the Active Space spin down
        for n in range(limitup, limitdown, 1): #loops over the down states
            if statej[n] == 1: #if it is occupied
                tHamil[jconf,jconf] = tHamil[jconf,jconf] + eig_valsSP[n-limitup+Numberbotstates] #then add to the corresponding matrix element the single particle eigenvalue

        tHamil[jconf,jconf] = tHamil[jconf,jconf] + constant #adds the constant term to the hamiltonian corresponding to the frozen valence states

    return tHamil

def confdic_gen(configurationslist):
    """This function generates a dictionary with the label for all the configurations and labels them with a decimal
    Inputs:
    configurationslist is the list of all the configurations in the Hilbert Space
    Outputs:
    confdic is the dictionary with the labeled configurations
    """
    confdic = {}
    for jconf in range(len(configurationslist)):
        counter = 1
        dec = 0.0
        conf = configurationslist[jconf]
        for y in conf:
            dec = dec + y/(10**counter)
            counter += 1
        confdic[str(dec)] = jconf
    return confdic



def build_hamU_SPbasis_confbasis_SP(U, eig_vectsSP, pos, configurationslist, zeromodes, Nbot, Ntop):
    """This function builds the Hubbard hamiltonian with a configuration (many-body) basis from the single-particle eigenvectors (I mention this in order to distinguish from a spin basis),
    employing the equation H = U sum_{nmn'm',i} psi_{uparrow n}(i)^* psi_{uparrow n'}(i) psi_{downarrow m}(i)^* psi_{downarrow m'}(i) c^dagger_{n,uparrow} c_{n',uparrow} c^dagger_{m,downarrow} c^dagger_{m',downarrow}
    Note: the sum can be brokendown into different terms, each of them will be computed separately in the following code, this will allow us to avoid computing some parts of the summation that are always 0.
    For instance, terms that move electrons between frozen valence states and the active space will always be 0 since they are not in the Hilbert space.
    Inputs:
    eig_vectsSP is the matrix of single-particle eigenvector
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    zeromodes is the number of zero modes in the system
    Nbot is the number of valence states in the Active Space
    Ntop is the number of conduction states in the Active Space
    returns a matrix which is the Hubbard hamiltonian in the many-body basis
    """
    UHamil = np.matrix(np.zeros(((len(configurationslist),len(configurationslist))),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot #This is the number of valence states out of the Active Space
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations


    #the next chunk of code computes the summation term corresponding to the operators acting just on the active space
    for jconf in range(len(configurationslist)): #it loops over all the configurations in the Hilbert space
        print (jconf,'U') #This is just to monitor where we are
        #This is for the Active Space states
        for n in range(0, int(len(configurationslist[0])/2), 1):  #it loops over the up states of the active space
            vectn = np.array(eig_vectsSP[n+Numberbotstates])[0]   #it gets the eigenvector
            for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states of the active space
                vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0]  #it gets the eigenvector
                for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                    vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector
                    for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                        vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector

                        #the following 3 lines are to make a copy of the current configuration, named as statej
                        statej = []
                        for obj in configurationslist[jconf]:
                            statej.append(obj)
                        if statej[mprime] == 1: #if there is an electron in the orbital m', continue
                        #The next 7 lines are to calculate the fermionic sign for the acting on orbital m'
                            epsilonanihidownlist = [Numberbotstates*2]
                            for fockindex in range(0, mprime, 1):
                                if fockindex < mprime:
                                    epsilonanihidownlist.append(statej[fockindex])
                                if fockindex == 0 and mprime == 0:
                                    epsilonanihidownlist.append(0.0)
                            epsilonanihidown = sum(epsilonanihidownlist)
                            statej[mprime] = 0 #it destroys an electron in the orbital labeled as mprime
                            if statej[m] == 0: #if there is not an electron in orbital m, continue
                            #The next 7 lines are to calculate the fermionic sign for the acting on orbital m
                                epsiloncreatdownlist = [Numberbotstates*2]
                                for fockindex in range(0, m, 1):
                                    if fockindex < m:
                                        epsiloncreatdownlist.append(statej[fockindex])
                                    if fockindex == 0 and m == 0:
                                        epsiloncreatdownlist.append(0.0)
                                epsiloncreatdown = sum(epsiloncreatdownlist)
                                statej[m] = 1 #it creates an electron in the orbital labeled as m
                                if statej[nprime] == 1: #if there is an electron in the orbital n', continue
                                #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                                    epsilonanihiuplist = [Numberbotstates]
                                    for fockindex in range(0, nprime, 1):
                                        if fockindex < nprime:
                                            epsilonanihiuplist.append(statej[fockindex])
                                        if fockindex == 0 and nprime == 0:
                                            epsilonanihiuplist.append(0.0)
                                    epsilonanihiup = sum(epsilonanihiuplist)
                                    statej[nprime] = 0 #it destroys an electron in the orbital labeled as nprime
                                    if statej[n] == 0: #if there is not an electron in the orbital n, continue
                                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                                        epsiloncreatuplist = [Numberbotstates]
                                        for fockindex in range(0, n, 1):
                                            if fockindex < n:
                                                epsiloncreatuplist.append(statej[fockindex])
                                            if fockindex == 0 and n == 0:
                                                epsiloncreatuplist.append(0.0)
                                        epsiloncreatup = sum(epsiloncreatuplist)
                                        statej[n] = 1 #it creates an electron in the orbital labeled as n

                                        counter = 1
                                        dec = 0.0
                                        conf = statej
                                        for y in conf:
                                            dec = dec + y/(10**counter)
                                            counter += 1
                                        iconf = confdic[str(dec)]

                                        Unnmmlist = [] #create an empty list
                                        for i in range(len(pos)): #loop over all the sites
                                            gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                                            gn = vectnprime[i] #this is \psi_{\uparrow nprime}(i)
                                            gmcompl = np.conj(vectm[i]) #this is \psi_{\downarrow m}(i)^*
                                            gm = vectmprime[i] #this is \psi_{\downarrow m'}(i)
                                            Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append in the list
                                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown) #this is the fermionic sign
                                        Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                                        UHamil[iconf, jconf] = UHamil[iconf, jconf] + Unnmm #the matrix element is added in the matrix




        #the next chunk of code computes the summation term corresponding to the operators acting just on the valence states out of the active space. Here the electrons will just be counted, so n = n' and m = m'
        for n in range(0, Numberbotstates, 1): #it loops over the valence states
            vectn = np.array(eig_vectsSP[n])[0] #it gets the eigenvector
            nprime = n #this is to remark that c^{\dagger}_n c_{n'} turns into the number operator
            vectnprime = np.array(eig_vectsSP[nprime])[0] #it gets the eigenvector
            for m in range(0, Numberbotstates, 1): #it loops over the valence states again
                vectm = np.array(eig_vectsSP[m])[0] #it gets the eigenvector
                mprime = m #this is to remark that c^{\dagger}_m c_{m'} turns into the number operator
                vectmprime = np.array(eig_vectsSP[mprime])[0] #it gets the eigenvector
                Unnmmlist = [] #an empty list
                for i in range(len(pos)): #loops over the sites
                    gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                    gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                    gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                    gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                    Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append in the list
                Unnmm = U*sum(Unnmmlist) #this is the matrix element. The fermionic sign here always is positive, so it is ommited
                UHamil[jconf, jconf] = UHamil[jconf, jconf] + Unnmm #the matrix element is added in the matrix


        #The next chunk of code computes the summation term corresponding to the operators acting on the valence states out of the active space and in the up states of the active space
        for n in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states in the active space
            vectn = np.array(eig_vectsSP[n+Numberbotstates])[0] #it gets the eigenstate
            for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states in the active space again
                vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0] #it gets the eigenstate
                for m in range(0, Numberbotstates, 1): #it loops over the valence states out of the active space
                    vectm = np.array(eig_vectsSP[m])[0] #it gets the eigenstate
                    mprime = m  #this is to remark that c^{\dagger}_m c_{m'} turns into the number operator
                    vectmprime = np.array(eig_vectsSP[mprime])[0] #it gets the eigenstate
                    #The next 3 lines are to create a copy of the current Fock state in the loop
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[nprime] == 1: #if there is an electron in the orbital n', continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                        epsilonanihiuplist = [Numberbotstates]
                        for fockindex in range(0, nprime, 1):
                            if fockindex < nprime:
                                epsilonanihiuplist.append(statej[fockindex])
                            if fockindex == 0 and nprime == 0:
                                epsilonanihiuplist.append(0.0)
                        epsilonanihiup = sum(epsilonanihiuplist)
                        statej[nprime] = 0 #it destroys an electron in the orbital n'
                        if statej[n] == 0: #if there is not an electron in the orbital n, continue
                        #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                            epsiloncreatuplist = [Numberbotstates]
                            for fockindex in range(0, n, 1):
                                if fockindex < n:
                                    epsiloncreatuplist.append(statej[fockindex])
                                if fockindex == 0 and n == 0:
                                    epsiloncreatuplist.append(0.0)
                            epsiloncreatup = sum(epsiloncreatuplist)
                            statej[n] = 1 #it creates an electron in the orbital n

                            counter = 1
                            dec = 0.0
                            conf = statej
                            for y in conf:
                                dec = dec + y/(10**counter)
                                counter += 1
                            iconf = confdic[str(dec)]

                            Unnmmlist = [] #an empty list
                            for i in range(len(pos)): #loops over the sites
                                gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                                gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                                gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                                gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                                Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append into the list
                            fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup) #this is the fermionic sign
                            Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                            UHamil[iconf, jconf] = UHamil[iconf, jconf] + Unnmm #the matrix element is added in the matrix





        #The next chunk of code computes the summation term corresponding to the operators acting on the valence states out of the active space and in the down states of the active space
        for n in range(0, Numberbotstates, 1): #it loops over the valence states out of the Active Space
            vectn = np.array(eig_vectsSP[n])[0] #it gets the eigenstate
            nprime = n #this is to remark that c^{\dagger}_n c_{n'} turns into the number operator
            vectnprime = np.array(eig_vectsSP[nprime])[0] #it gets the eigenstate
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over down states in the active space
                vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenstate
                for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over down states in the active space
                    vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenstate
                    #The next 3 lines are to create a copy of the current Fock state in the loop
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1: #if there is an electron in the orbital m', continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital m'
                        epsilonanihidownlist = [Numberbotstates*2]
                        for fockindex in range(0, mprime, 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            if fockindex == 0 and mprime == 0:
                                epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0 #it destroys an electron in the orbital m'
                        if statej[m] == 0: #if there is an electron in the orbital m, continue
                        #The next 7 lines are to calculate the fermionic sign for the acting on orbital m
                            epsiloncreatdownlist = [Numberbotstates*2]
                            for fockindex in range(0, m, 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                if fockindex == 0 and m == 0:
                                    epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1 #it creates an electron in the orbital m

                            counter = 1
                            dec = 0.0
                            conf = statej
                            for y in conf:
                                dec = dec + y/(10**counter)
                                counter += 1
                            iconf = confdic[str(dec)]

                            Unnmmlist = [] #an empty list
                            for i in range(len(pos)): #it loops over the sites
                                gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                                gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                                gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                                gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                                Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append into the list
                            fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown) #this is the fermionic sign
                            Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                            UHamil[iconf, jconf] = UHamil[iconf, jconf] + Unnmm #the matrix element is added in the matrix

    return UHamil

def build_HamUparal(UHamilold,configurationslist):
    UHamil = np.matrix(np.zeros(((len(configurationslist),len(configurationslist))),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    for row in range(len(UHamilold)):
        for i in range(len(configurationslist)):
            UHamil[i,row] = UHamilold[row][i,0]
    return UHamil


def build_hamU_SPbasis_confbasis_SP_paral(U, eig_vectsSP, pos, configurationslist, zeromodes, Nbot, Ntop, jconf):
    """This function builds the Hubbard hamiltonian with a configuration (many-body) basis from the single-particle eigenvectors (I mention this in order to distinguish from a spin basis),
    employing the equation H = U sum_{nmn'm',i} psi_{uparrow n}(i)^* psi_{uparrow n'}(i) psi_{downarrow m}(i)^* psi_{downarrow m'}(i) c^dagger_{n,uparrow} c_{n',uparrow} c^dagger_{m,downarrow} c^dagger_{m',downarrow}
    Note: the sum can be brokendown into different terms, each of them will be computed separately in the following code, this will allow us to avoid computing some parts of the summation that are always 0.
    For instance, terms that move electrons between frozen valence states and the active space will always be 0 since they are not in the Hilbert space.
    Inputs:
    eig_vectsSP is the matrix of single-particle eigenvector
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    zeromodes is the number of zero modes in the system
    Nbot is the number of valence states in the Active Space
    Ntop is the number of conduction states in the Active Space
    returns a matrix which is the Hubbard hamiltonian in the many-body basis
    """
    UHamil = np.matrix(np.zeros(((len(configurationslist),1)),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot #This is the number of valence states out of the Active Space
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations


    #the next chunk of code computes the summation term corresponding to the operators acting just on the active space
    #num_cores = multiprocessing.cpu_count()

    print (jconf,'U') #This is just to monitor where we are
    save_representation([str(jconf)+' '+ str(len(configurationslist))], [''], 'checks')
    #This is for the Active Space states
    for n in range(0, int(len(configurationslist[0])/2), 1):  #it loops over the up states of the active space
        vectn = np.array(eig_vectsSP[n+Numberbotstates])[0]   #it gets the eigenvector
        for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states of the active space
            vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0]  #it gets the eigenvector
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector
                for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                    vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector

                    #the following 3 lines are to make a copy of the current configuration, named as statej
                    statej = []
                    for obj in configurationslist[jconf]:
                        statej.append(obj)
                    if statej[mprime] == 1: #if there is an electron in the orbital m', continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital m'
                        epsilonanihidownlist = [Numberbotstates*2]
                        for fockindex in range(0, mprime, 1):
                            if fockindex < mprime:
                                epsilonanihidownlist.append(statej[fockindex])
                            if fockindex == 0 and mprime == 0:
                                epsilonanihidownlist.append(0.0)
                        epsilonanihidown = sum(epsilonanihidownlist)
                        statej[mprime] = 0 #it destroys an electron in the orbital labeled as mprime
                        if statej[m] == 0: #if there is not an electron in orbital m, continue
                        #The next 7 lines are to calculate the fermionic sign for the acting on orbital m
                            epsiloncreatdownlist = [Numberbotstates*2]
                            for fockindex in range(0, m, 1):
                                if fockindex < m:
                                    epsiloncreatdownlist.append(statej[fockindex])
                                if fockindex == 0 and m == 0:
                                    epsiloncreatdownlist.append(0.0)
                            epsiloncreatdown = sum(epsiloncreatdownlist)
                            statej[m] = 1 #it creates an electron in the orbital labeled as m
                            if statej[nprime] == 1: #if there is an electron in the orbital n', continue
                            #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                                epsilonanihiuplist = [Numberbotstates]
                                for fockindex in range(0, nprime, 1):
                                    if fockindex < nprime:
                                        epsilonanihiuplist.append(statej[fockindex])
                                    if fockindex == 0 and nprime == 0:
                                        epsilonanihiuplist.append(0.0)
                                epsilonanihiup = sum(epsilonanihiuplist)
                                statej[nprime] = 0 #it destroys an electron in the orbital labeled as nprime
                                if statej[n] == 0: #if there is not an electron in the orbital n, continue
                                #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                                    epsiloncreatuplist = [Numberbotstates]
                                    for fockindex in range(0, n, 1):
                                        if fockindex < n:
                                            epsiloncreatuplist.append(statej[fockindex])
                                        if fockindex == 0 and n == 0:
                                            epsiloncreatuplist.append(0.0)
                                    epsiloncreatup = sum(epsiloncreatuplist)
                                    statej[n] = 1 #it creates an electron in the orbital labeled as n

                                    counter = 1
                                    dec = 0.0
                                    conf = statej
                                    for y in conf:
                                        dec = dec + y/(10**counter)
                                        counter += 1
                                    iconf = confdic[str(dec)]

                                    Unnmmlist = [] #create an empty list
                                    for i in range(len(pos)): #loop over all the sites
                                        gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                                        gn = vectnprime[i] #this is \psi_{\uparrow nprime}(i)
                                        gmcompl = np.conj(vectm[i]) #this is \psi_{\downarrow m}(i)^*
                                        gm = vectmprime[i] #this is \psi_{\downarrow m'}(i)
                                        Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append in the list
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown) #this is the fermionic sign
                                    Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                                    UHamil[iconf, 0] = UHamil[iconf, 0] + Unnmm #the matrix element is added in the matrix




    #the next chunk of code computes the summation term corresponding to the operators acting just on the valence states out of the active space. Here the electrons will just be counted, so n = n' and m = m'
    for n in range(0, Numberbotstates, 1): #it loops over the valence states
        vectn = np.array(eig_vectsSP[n])[0] #it gets the eigenvector
        nprime = n #this is to remark that c^{\dagger}_n c_{n'} turns into the number operator
        vectnprime = np.array(eig_vectsSP[nprime])[0] #it gets the eigenvector
        for m in range(0, Numberbotstates, 1): #it loops over the valence states again
            vectm = np.array(eig_vectsSP[m])[0] #it gets the eigenvector
            mprime = m #this is to remark that c^{\dagger}_m c_{m'} turns into the number operator
            vectmprime = np.array(eig_vectsSP[mprime])[0] #it gets the eigenvector
            Unnmmlist = [] #an empty list
            for i in range(len(pos)): #loops over the sites
                gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append in the list
            Unnmm = U*sum(Unnmmlist) #this is the matrix element. The fermionic sign here always is positive, so it is ommited
            UHamil[jconf, 0] = UHamil[jconf, 0] + Unnmm #the matrix element is added in the matrix


    #The next chunk of code computes the summation term corresponding to the operators acting on the valence states out of the active space and in the up states of the active space
    for n in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states in the active space
        vectn = np.array(eig_vectsSP[n+Numberbotstates])[0] #it gets the eigenstate
        for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states in the active space again
            vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0] #it gets the eigenstate
            for m in range(0, Numberbotstates, 1): #it loops over the valence states out of the active space
                vectm = np.array(eig_vectsSP[m])[0] #it gets the eigenstate
                mprime = m  #this is to remark that c^{\dagger}_m c_{m'} turns into the number operator
                vectmprime = np.array(eig_vectsSP[mprime])[0] #it gets the eigenstate
                #The next 3 lines are to create a copy of the current Fock state in the loop
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                if statej[nprime] == 1: #if there is an electron in the orbital n', continue
                #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                    epsilonanihiuplist = [Numberbotstates]
                    for fockindex in range(0, nprime, 1):
                        if fockindex < nprime:
                            epsilonanihiuplist.append(statej[fockindex])
                        if fockindex == 0 and nprime == 0:
                            epsilonanihiuplist.append(0.0)
                    epsilonanihiup = sum(epsilonanihiuplist)
                    statej[nprime] = 0 #it destroys an electron in the orbital n'
                    if statej[n] == 0: #if there is not an electron in the orbital n, continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                        epsiloncreatuplist = [Numberbotstates]
                        for fockindex in range(0, n, 1):
                            if fockindex < n:
                                epsiloncreatuplist.append(statej[fockindex])
                            if fockindex == 0 and n == 0:
                                epsiloncreatuplist.append(0.0)
                        epsiloncreatup = sum(epsiloncreatuplist)
                        statej[n] = 1 #it creates an electron in the orbital n

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = [] #an empty list
                        for i in range(len(pos)): #loops over the sites
                            gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                            gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                            gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                            gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                            Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append into the list
                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup) #this is the fermionic sign
                        Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                        UHamil[iconf, 0] = UHamil[iconf, 0] + Unnmm #the matrix element is added in the matrix





    #The next chunk of code computes the summation term corresponding to the operators acting on the valence states out of the active space and in the down states of the active space
    for n in range(0, Numberbotstates, 1): #it loops over the valence states out of the Active Space
        vectn = np.array(eig_vectsSP[n])[0] #it gets the eigenstate
        nprime = n #this is to remark that c^{\dagger}_n c_{n'} turns into the number operator
        vectnprime = np.array(eig_vectsSP[nprime])[0] #it gets the eigenstate
        for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over down states in the active space
            vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenstate
            for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over down states in the active space
                vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenstate
                #The next 3 lines are to create a copy of the current Fock state in the loop
                statej = []
                for obj in configurationslist[jconf]:
                    statej.append(obj)
                if statej[mprime] == 1: #if there is an electron in the orbital m', continue
                #The next 7 lines are to calculate the fermionic sign for the acting on orbital m'
                    epsilonanihidownlist = [Numberbotstates*2]
                    for fockindex in range(0, mprime, 1):
                        if fockindex < mprime:
                            epsilonanihidownlist.append(statej[fockindex])
                        if fockindex == 0 and mprime == 0:
                            epsilonanihidownlist.append(0.0)
                    epsilonanihidown = sum(epsilonanihidownlist)
                    statej[mprime] = 0 #it destroys an electron in the orbital m'
                    if statej[m] == 0: #if there is an electron in the orbital m, continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital m
                        epsiloncreatdownlist = [Numberbotstates*2]
                        for fockindex in range(0, m, 1):
                            if fockindex < m:
                                epsiloncreatdownlist.append(statej[fockindex])
                            if fockindex == 0 and m == 0:
                                epsiloncreatdownlist.append(0.0)
                        epsiloncreatdown = sum(epsiloncreatdownlist)
                        statej[m] = 1 #it creates an electron in the orbital m

                        counter = 1
                        dec = 0.0
                        conf = statej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = [] #an empty list
                        for i in range(len(pos)): #it loops over the sites
                            gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                            gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                            gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                            gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                            Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append into the list
                        fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown) #this is the fermionic sign
                        Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                        UHamil[iconf, 0] = UHamil[iconf, 0] + Unnmm #the matrix element is added in the matrix

    return UHamil

def build_HamU_binary(U, eig_vectsSP, pos, configurationslist, zeromodes, Nbot, Ntop, jconf):
    UHamil = np.matrix(np.zeros(((len(configurationslist),1)),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot #This is the number of valence states out of the Active Space
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations

    configurationsliststring = []
    for i in range(len(configurationslist)):
        confistring = ''
        confi = configurationslist[i]
        for j in confi:
            confistring = confistring + str(j)
        configurationsliststring.append(confistring)
    configurationslistint = []
    for i in configurationsliststring:
        configurationslistint.append(int(i,2))

    mprimelistabinary = []
    for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
        state = [0]*len(configurationslist[0])
        state[mprime] = 1
        confistring = ''
        for j in state:
            confistring = confistring + str(j)
        mprimelistabinary.append(confistring)
    mprimelistint = []
    for i in mprimelistabinary:
        mprimelistint.append(int(i,2))

    nprimelistabinary = []
    for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
        state = [0]*len(configurationslist[0])
        state[nprime] = 1
        confistring = ''
        for j in state:
            confistring = confistring + str(j)
        nprimelistabinary.append(confistring)
    nprimelistint = []
    for i in nprimelistabinary:
        nprimelistint.append(int(i,2))

    print(configurationslistint[0])
    print(mprimelistint[0])

    st


    #the next chunk of code computes the summation term corresponding to the operators acting just on the active space
    #num_cores = multiprocessing.cpu_count()

    print (jconf,'U') #This is just to monitor where we are
    save_representation([str(jconf)+' '+ str(len(configurationslist))], [''], 'checks')
    #This is for the Active Space states
    for n in range(0, int(len(configurationslist[0])/2), 1):  #it loops over the up states of the active space
        vectn = np.array(eig_vectsSP[n+Numberbotstates])[0]   #it gets the eigenvector
        for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states of the active space
            vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0]  #it gets the eigenvector
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector
                for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                    vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector


    return HamU

def build_hamU_SPbasis_confbasis_SP_paral_improved(U, eig_vectsSP, pos, configurationslist, zeromodes, Nbot, Ntop, jconf):
    """This function builds the Hubbard hamiltonian with a configuration (many-body) basis from the single-particle eigenvectors (I mention this in order to distinguish from a spin basis),
    employing the equation H = U sum_{nmn'm',i} psi_{uparrow n}(i)^* psi_{uparrow n'}(i) psi_{downarrow m}(i)^* psi_{downarrow m'}(i) c^dagger_{n,uparrow} c_{n',uparrow} c^dagger_{m,downarrow} c^dagger_{m',downarrow}
    Note: the sum can be brokendown into different terms, each of them will be computed separately in the following code, this will allow us to avoid computing some parts of the summation that are always 0.
    For instance, terms that move electrons between frozen valence states and the active space will always be 0 since they are not in the Hilbert space.
    Inputs:
    eig_vectsSP is the matrix of single-particle eigenvector
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    zeromodes is the number of zero modes in the system
    Nbot is the number of valence states in the Active Space
    Ntop is the number of conduction states in the Active Space
    returns a matrix which is the Hubbard hamiltonian in the many-body basis
    """
    UHamil = np.matrix(np.zeros(((len(configurationslist),1)),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    Numberbotstates = int((len(pos) - zeromodes)/2)-Nbot #This is the number of valence states out of the Active Space
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations


    #the next chunk of code computes the summation term corresponding to the operators acting just on the active space
    #num_cores = multiprocessing.cpu_count()

    print (jconf,'U') #This is just to monitor where we are
    save_representation([str(jconf)+' '+ str(len(configurationslist))], [''], 'checks')
    #This is for the Active Space states

    for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
        vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector

        statej = []
        for obj in configurationslist[jconf]:
            statej.append(obj)

        if statej[mprime] == 1: #if there is an electron in the orbital m', continue
        #The next 7 lines are to calculate the fermionic sign for the acting on orbital m'
            epsilonanihidownlist = [Numberbotstates*2]
            for fockindex in range(0, mprime, 1):
                if fockindex < mprime:
                    epsilonanihidownlist.append(statej[fockindex])
                if fockindex == 0 and mprime == 0:
                    epsilonanihidownlist.append(0.0)
            epsilonanihidown = sum(epsilonanihidownlist)
            statej[mprime] = 0 #it destroys an electron in the orbital labeled as mprime
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                newstatej = []
                for obj in statej:
                    newstatej.append(obj)
                vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenvector
                if newstatej[m] == 0: #if there is not an electron in orbital m, continue
                #The next 7 lines are to calculate the fermionic sign for the acting on orbital m
                    epsiloncreatdownlist = [Numberbotstates*2]
                    for fockindex in range(0, m, 1):
                        if fockindex < m:
                            epsiloncreatdownlist.append(newstatej[fockindex])
                        if fockindex == 0 and m == 0:
                            epsiloncreatdownlist.append(0.0)
                    epsiloncreatdown = sum(epsiloncreatdownlist)
                    newstatej[m] = 1 #it creates an electron in the orbital labeled as m
                    for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states of the active space
                        newnewstatej = []
                        for obj in newstatej:
                            newnewstatej.append(obj)
                        vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0]  #it gets the eigenvector
                        if newnewstatej[nprime] == 1: #if there is an electron in the orbital n', continue
                        #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                            epsilonanihiuplist = [Numberbotstates]
                            for fockindex in range(0, nprime, 1):
                                if fockindex < nprime:
                                    epsilonanihiuplist.append(newnewstatej[fockindex])
                                if fockindex == 0 and nprime == 0:
                                    epsilonanihiuplist.append(0.0)
                            epsilonanihiup = sum(epsilonanihiuplist)
                            newnewstatej[nprime] = 0 #it destroys an electron in the orbital labeled as nprime
                            for n in range(0, int(len(configurationslist[0])/2), 1):  #it loops over the up states of the active space
                                newnewnewstatej = []
                                for obj in newnewstatej:
                                    newnewnewstatej.append(obj)
                                vectn = np.array(eig_vectsSP[n+Numberbotstates])[0]   #it gets the eigenvector
                                if newnewnewstatej[n] == 0: #if there is not an electron in the orbital n, continue
                                #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                                    epsiloncreatuplist = [Numberbotstates]
                                    for fockindex in range(0, n, 1):
                                        if fockindex < n:
                                            epsiloncreatuplist.append(newnewnewstatej[fockindex])
                                        if fockindex == 0 and n == 0:
                                            epsiloncreatuplist.append(0.0)
                                    epsiloncreatup = sum(epsiloncreatuplist)
                                    newnewnewstatej[n] = 1 #it creates an electron in the orbital labeled as n

                                    counter = 1
                                    dec = 0.0
                                    conf = newnewnewstatej
                                    for y in conf:
                                        dec = dec + y/(10**counter)
                                        counter += 1
                                    iconf = confdic[str(dec)]

                                    Unnmmlist = [] #create an empty list
                                    for i in range(len(pos)): #loop over all the sites
                                        gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                                        gn = vectnprime[i] #this is \psi_{\uparrow nprime}(i)
                                        gmcompl = np.conj(vectm[i]) #this is \psi_{\downarrow m}(i)^*
                                        gm = vectmprime[i] #this is \psi_{\downarrow m'}(i)
                                        Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append in the list
                                    fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)*((-1)**epsilonanihidown)*((-1)**epsiloncreatdown) #this is the fermionic sign
                                    Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                                    UHamil[iconf, 0] = UHamil[iconf, 0] + Unnmm #the matrix element is added in the matrix





                    #the following 3 lines are to make a copy of the current configuration, named as statej









    #the next chunk of code computes the summation term corresponding to the operators acting just on the valence states out of the active space. Here the electrons will just be counted, so n = n' and m = m'
    for n in range(0, Numberbotstates, 1): #it loops over the valence states
        vectn = np.array(eig_vectsSP[n])[0] #it gets the eigenvector
        nprime = n #this is to remark that c^{\dagger}_n c_{n'} turns into the number operator
        vectnprime = np.array(eig_vectsSP[nprime])[0] #it gets the eigenvector
        for m in range(0, Numberbotstates, 1): #it loops over the valence states again
            vectm = np.array(eig_vectsSP[m])[0] #it gets the eigenvector
            mprime = m #this is to remark that c^{\dagger}_m c_{m'} turns into the number operator
            vectmprime = np.array(eig_vectsSP[mprime])[0] #it gets the eigenvector
            Unnmmlist = [] #an empty list
            for i in range(len(pos)): #loops over the sites
                gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append in the list
            Unnmm = U*sum(Unnmmlist) #this is the matrix element. The fermionic sign here always is positive, so it is ommited
            UHamil[jconf, 0] = UHamil[jconf, 0] + Unnmm #the matrix element is added in the matrix


    #The next chunk of code computes the summation term corresponding to the operators acting on the valence states out of the active space and in the up states of the active space
    for m in range(0, Numberbotstates, 1): #it loops over the valence states out of the active space
        statej = []
        for obj in configurationslist[jconf]:
            statej.append(obj)
        vectm = np.array(eig_vectsSP[m])[0] #it gets the eigenstate
        mprime = m  #this is to remark that c^{\dagger}_m c_{m'} turns into the number operator
        vectmprime = np.array(eig_vectsSP[mprime])[0] #it gets the eigenstate
        for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states in the active space again
            newstatej = []
            for obj in statej:
                newstatej.append(obj)
            vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0] #it gets the eigenstate
            if newstatej[nprime] == 1: #if there is an electron in the orbital n', continue
            #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                epsilonanihiuplist = [Numberbotstates]
                for fockindex in range(0, nprime, 1):
                    if fockindex < nprime:
                        epsilonanihiuplist.append(newstatej[fockindex])
                    if fockindex == 0 and nprime == 0:
                        epsilonanihiuplist.append(0.0)
                epsilonanihiup = sum(epsilonanihiuplist)
                newstatej[nprime] = 0 #it destroys an electron in the orbital n'
                for n in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states in the active space
                    newnewstatej = []
                    for obj in newstatej:
                        newnewstatej.append(obj)
                    vectn = np.array(eig_vectsSP[n+Numberbotstates])[0] #it gets the eigenstate
                    if newnewstatej[n] == 0: #if there is not an electron in the orbital n, continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                        epsiloncreatuplist = [Numberbotstates]
                        for fockindex in range(0, n, 1):
                            if fockindex < n:
                                epsiloncreatuplist.append(newnewstatej[fockindex])
                            if fockindex == 0 and n == 0:
                                epsiloncreatuplist.append(0.0)
                        epsiloncreatup = sum(epsiloncreatuplist)
                        newnewstatej[n] = 1 #it creates an electron in the orbital n

                        counter = 1
                        dec = 0.0
                        conf = newnewstatej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = [] #an empty list
                        for i in range(len(pos)): #loops over the sites
                            gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                            gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                            gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                            gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                            Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append into the list
                        fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup) #this is the fermionic sign
                        Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                        UHamil[iconf, 0] = UHamil[iconf, 0] + Unnmm #the matrix element is added in the matrix



                #The next 3 lines are to create a copy of the current Fock state in the loop








    #The next chunk of code computes the summation term corresponding to the operators acting on the valence states out of the active space and in the down states of the active space
    for mprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over down states in the active space
        statej = []
        for obj in configurationslist[jconf]:
            statej.append(obj)
        vectmprime = np.array(eig_vectsSP[mprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenstate
        if statej[mprime] == 1: #if there is an electron in the orbital m', continue
        #The next 7 lines are to calculate the fermionic sign for the acting on orbital m'
            epsilonanihidownlist = [Numberbotstates*2]
            for fockindex in range(0, mprime, 1):
                if fockindex < mprime:
                    epsilonanihidownlist.append(statej[fockindex])
                if fockindex == 0 and mprime == 0:
                    epsilonanihidownlist.append(0.0)
            epsilonanihidown = sum(epsilonanihidownlist)
            statej[mprime] = 0 #it destroys an electron in the orbital m'
            for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over down states in the active space
                newstatej = []
                for obj in statej:
                    newstatej.append(obj)
                vectm = np.array(eig_vectsSP[m+Numberbotstates-int(len(configurationslist[jconf])/2)])[0] #it gets the eigenstate
                if newstatej[m] == 0: #if there is an electron in the orbital m, continue
                #The next 7 lines are to calculate the fermionic sign for the acting on orbital m
                    epsiloncreatdownlist = [Numberbotstates*2]
                    for fockindex in range(0, m, 1):
                        if fockindex < m:
                            epsiloncreatdownlist.append(newstatej[fockindex])
                        if fockindex == 0 and m == 0:
                            epsiloncreatdownlist.append(0.0)
                    epsiloncreatdown = sum(epsiloncreatdownlist)
                    newstatej[m] = 1 #it creates an electron in the orbital m
                    for n in range(0, Numberbotstates, 1): #it loops over the valence states out of the Active Space
                        vectn = np.array(eig_vectsSP[n])[0] #it gets the eigenstate
                        nprime = n #this is to remark that c^{\dagger}_n c_{n'} turns into the number operator
                        vectnprime = np.array(eig_vectsSP[nprime])[0] #it gets the eigenstate

                        counter = 1
                        dec = 0.0
                        conf = newstatej
                        for y in conf:
                            dec = dec + y/(10**counter)
                            counter += 1
                        iconf = confdic[str(dec)]

                        Unnmmlist = [] #an empty list
                        for i in range(len(pos)): #it loops over the sites
                            gncompl = np.conj(vectn[i]) #this is \psi_{\uparrow n}(i)^*
                            gn = vectnprime[i] #this is \psi_{\uparrow n'}(i)
                            gmcompl = np.conj(vectm[i]) #this is \psi_{\uparrow m}(i)^*
                            gm = vectmprime[i] #this is \psi_{\uparrow m'}(i)
                            Unnmmlist.append(gncompl*gn*gmcompl*gm) #the multiplication is append into the list
                        fermionic_sign = ((-1)**epsilonanihidown)*((-1)**epsiloncreatdown) #this is the fermionic sign
                        Unnmm = U*sum(Unnmmlist)*fermionic_sign #this is the matrix element
                        UHamil[iconf, 0] = UHamil[iconf, 0] + Unnmm #the matrix element is added in the matrix



                    #The next 3 lines are to create a copy of the current Fock state in the loop

    return UHamil
