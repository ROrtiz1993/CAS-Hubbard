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

def get_eigvals_eigvects_Herm(h):
    """This function gets the eigenvalues and eigenvectors of a Hermitian matrix.
    Inputs:
    h is the matrix to diagonalize
    returns a list with the eigenvalues and a matrix with the eigenvectors
    """
    eigvals,eigvects = np.linalg.eigh(h)
    eigvects = eigvects.transpose() #Tranposes the eigenvectors, this way the rows are the eigenstates.
    return eigvals, eigvects

def get_eigvals_eigvects_nonHerm(h):
    """This function gets the eigenvalues and eigenvectors of a non-Hermitian matrix.
    Inputs:
    h is the matrix to diagonalize
    returns a list with the eigenvalues and a matrix with the eigenvectors
    """
    eigvals,eigvects = np.linalg.eig(h)
    eigvects = eigvects.transpose() #Tranposes the eigenvectors, this way the rows are the eigenstates.
    return eigvals, eigvects

def buildham_TB_SP_nospin(pos, t, a):
    """This function builds a spinless tight-binding hamiltonian. Single-particle basis, one orbital per atom. The hopping is considered for sites at a distance a.
    Inputs:
    pos is a list with the vectors of the lattice
    t is the hopping parameter
    a is the length at which the atoms should be for hopping to be considered
    returns a nxn matrix
    """
    n = len(pos)         #This is the dimension of the Hilbert Space
    ht = np.matrix(np.zeros((n,n),dtype=np.complex_))        #This is the Hamiltonian with just 0's
    for i in range(n):         #A loop that runs over all the positions
        ri = pos[i]        #This is the vector for position i
        for j in range(n):     #A loop that runs again over all the positions
            rj = pos[j]      #This is the vector for position j
            dist = np.linalg.norm(rj - ri)   #This is the distance between ri and rj
            if (abs(dist - a) < 0.1):    #If the distance between ri and rj is close to a with a tolerance of 0.1 then...
                ht[j,i] = t         #<j|Ht|i> = t
    return ht

def buildham_TB_SP_spin(pos, t, a):
    """This function builds the spinful tight-binding hamiltonian for a spin basis: |1up>, |1down>, |2up>, |2down> and so on. Single-particle basis, one orbital per atom.
    Inputs:
    pos is a list with the vectors of the lattice
    t is the hopping parameter
    a is the length at which the atoms should be for hopping to be considered
    returns a 2nx2n matrix
    """
    n = len(pos) #n is the number of sites
    ht = np.matrix(np.zeros((2*n,2*n),dtype=np.complex_)) #makes empty zero matrix with 2n,2n dimensions, in account of spin
    for i in range(len(pos)): #loops over sites
        ri = pos[i]     #ri is a position vector
        for j in range(len(pos)): #loops again
            rj = pos[j]      #rj is another position vector
            dist = np.linalg.norm(rj - ri) #dist is the distance between ri and rj
            if abs(dist - a) < 0.1 : #If the distance between ri and rj is close to a with a tolerance of 0.1 then...
                ht[2*j, 2*i] = t #<jup|Ht|iup> = t
                ht[(2*j)+1, (2*i)+1] = t #<jdown|Ht|idown> = t
    return ht

def build_hamRashba(pos, tR, sigmax, sigmay, sigmaz, a):
    """This function builds rashba hamiltonian for a spin basis |1up>, |1down>, |2up>, |2down> etc, with the Rashba term as function of an applied electric field that breaks the symmetry. Single-particle basis, one orbital per atom.
    Inputs:
    pos is a list with the atomic vectors
    tR is a list with the three components of an external electric field
    sigmax, sigmay and sigmaz are the Pauli matrixes with 1's
    a is the length at which the atoms should be for hopping to be considered
    returns a 2nx2n matrix
    """
    n = len(pos) #n is the number of sites
    hR = np.matrix(np.zeros((2*n,2*n),dtype=np.complex_)) #makes empty zero matrix with 2n,2n dimensions, in account of spin
    for i in range(len(pos)): #loops over sites
        ri = pos[i]     #ri is a position vector
        for j in range(len(pos)): #loops again
            rj = pos[j]      #rj is another position vector
            dist = np.linalg.norm(rj - ri)   #dist is the distance between ri and rj
            if abs (dist - a) < 0.1 : #if that distance is first-neighbours
                vecbond = (rj - ri)/a #vecbond is a vector that connects ri and rj
                vecbondx = vecbond[0] #component x of vecbond
                vecbondy = vecbond[1] #component y of vecbond
                vecbondz = vecbond[2] #component z of vecbond
                Rashbaterm = tR[0]*(sigmaz*vecbondy - sigmay*vecbondz) - tR[1]*(sigmaz*vecbondx - sigmax*vecbondz) + tR[2]*(sigmay*vecbondx - sigmax*vecbondy) #Rashba term is the term from the Rashba Hamiltonian
                hR[2*j, 2*i] = Rashbaterm[0,0] #The Rashba term goes to the matrix
                hR[2*j, 2*i+1] = Rashbaterm[0,1]
                hR[2*j+1, 2*i] = Rashbaterm[1,0]
                hR[2*j+1, 2*i+1] = Rashbaterm[1,1]
    return hR



def build_ham_TB_MB_SPCAS(eig_valsSP, pos, configurationslist, No, Ne, Extra_e):
    """This function builds the tight-binding hamiltonian with a configuration basis (many-body),
    employing the equation, running only over occupied states,
    H = \sum_{n,\sigma} \epsilon_n c^dagger_{n,\sigma} c_{n,\sigma}.
    The configurations consist in the electrons fluctuating through the SP states inside the CAS.
    Inputs:
    eig_valsSP is the list of single-particle eigenvalues
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    No is the number of orbitals in the Active Space
    Ne is the number of electrons in the Active Space
    Extra_e is the number of extra electrons with respect the neutral state (>0(<0) for negative(positive) charged)
    returns a matrix which is the tight-binding hamiltonian in the many-body basis
    """
    No_tot = len(pos) #Total single-particle states
    Ne_tot = len(pos) + Extra_e #Total of electrons
    Nconf = len(configurationslist) #Total of configurations
    Numberbotstates = int((Ne_tot - Ne)/2) #This is the number of valence states out of the Active Space
    tHamil = np.matrix(np.zeros((Nconf,Nconf),dtype=np.complex_)) #this is the empty (all zeros) tight-binding hamiltonian

    #H can be divided into different chunks. This one corresponds to the frozen valence states, that will correspond to a constant
    constantlist= [] #This is just an empty list
    for i in range(0, Numberbotstates, 1): #loops over all the frozen valence states
        constantlist.append(eig_valsSP[i]) #append in the previous list the single-particle eigenvalue
    constant = sum(constantlist)*2 #The sum of that list, times 2 for the spin, will be the previously mentioned constant

    #This part corresponds to the Active Space
    for jconf in range(Nconf): #loops over every configuration
        #print (jconf, 't')
        statej = []  #The next 3 lines makes a copy of the configuration labeled as jconf
        for obj in configurationslist[jconf]:
            statej.append(obj)
        #This part is for the Active Space spin up
        for n in range(0, No, 1): #loops over the up states
            if statej[n] == 1: #if it is occupied
                tHamil[jconf,jconf] = tHamil[jconf,jconf] + eig_valsSP[n+Numberbotstates] #then add to the corresponding matrix element the single particle eigenvalue. To understand the label in eig_valsSP it is important to notice that in the configuration there is just the Active Space
        #This part is for the Active Space spin down
        for n in range(No, 2*No, 1): #loops over the down states
            if statej[n] == 1: #if it is occupied
                tHamil[jconf,jconf] = tHamil[jconf,jconf] + eig_valsSP[n-No+Numberbotstates] #then add to the corresponding matrix element the single particle eigenvalue
        tHamil[jconf,jconf] = tHamil[jconf,jconf] + constant #adds the constant term to the hamiltonian corresponding to the frozen valence states
    return tHamil


def build_ham_Zeeman_MB_SPCAS(B, pos, configurationslist):
    """#This function builds the Zeeman hamiltonian with a configuration basis (many-body),
    it counts Zeeman by counting the difference between up and down electrons in a configuration
    and it applies edif times g\mu_B S_zB_z. This function is just for B=(0,0,Bz).
    Inputs:
    B is (Bx,By,Bz), in Teslas
    pos is a list of position vectors (np.arrays)
    configurationslist is the list of Fock configurations
    returns the many-body Zeeman Hamiltonian in eV
    """
    ZeeHamil = np.matrix(np.zeros((len(configurationslist),len(configurationslist)),dtype=np.complex_)) #This is the Zeeman Hamiltonian with zeros
    for jconf in range(len(configurationslist)):   #it loops over every configuration
        statej = [] #the next 3 lines create a copy of the configuration
        for obj in configurationslist[jconf]:
            statej.append(obj)
        uplist = [] #list of up electrons
        downlist = [] #list of down electrons
        for n in range(0, int(len(configurationslist[jconf])/2), 1):  #it loops over up electrons
            if statej[n] == 1: #if the occupation is 1
                uplist.append(1.0) #it counts the up electron
        for m in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])), 1): #it loops over down electrons
            if statej[m] == 1: #if the occupation is 1
                downlist.append(1.0) #it counts the down electron
        eup = sum(uplist) #total of up electrons in the configuration
        edown = sum(downlist) #total of down electrons in the configuration
        edifference = eup - edown #difference between up and down electrons
        Z = B[2]*2*0.000057883818066*0.5*edifference #zeeman term for this configuration, the bohr magneton is in eV/T
        ZeeHamil[jconf, jconf] = ZeeHamil[jconf, jconf] + Z #the zeeman term is added to the hamiltonian
    return ZeeHamil

def build_ham_Hubbard_MB_SPCAS(U, eig_vectsSP, pos, configurationslist, Ne, Extra_e):
    """This function builds the Hubbard hamiltonian with a configuration (many-body) basis. The configurations consist in the electrons fluctuating through the SP states inside the CAS.
    H = U sum_{nmn'm',i} psi_{uparrow n}(i)^* psi_{uparrow n'}(i) psi_{downarrow m}(i)^* psi_{downarrow m'}(i) c^dagger_{n,uparrow} c_{n',uparrow} c^dagger_{m,downarrow} c^dagger_{m',downarrow}
    Note: the sum can be brokendown into different terms, each of them will be computed separately in the following code, this will allow us to avoid computing some parts of the summation that are always 0.
    For instance, terms that move electrons between frozen valence states and the active space will always be 0 since they are not in the Hilbert space.
    Inputs:
    U is the Hubbard parameter
    eig_vectsSP is the matrix of single-particle eigenvector
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    Ne is the number of electrons in the CAS
    Extra_e is the number of extra electrons with respect the neutral state
    returns a matrix which is the Hubbard hamiltonian in the many-body basis
    """
    No_tot = len(pos) #Total single-particle states
    Ne_tot = len(pos) + Extra_e #Total of electrons
    Nconf = len(configurationslist) #Total of configurations
    Numberbotstates = int((Ne_tot - Ne)/2) #This is the number of valence states out of the Active Space
    UHamil = np.matrix(np.zeros(((Nconf,Nconf)),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations

    #the next chunk of code computes the summation term corresponding to the operators acting just on the active space
    for jconf in range(Nconf): #it loops over all the configurations in the Hilbert space
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
    """This function prepares the Hamiltonian in matrix form. This is done for the Hubbard Hamiltonian but actually works for any Hamiltonian that is output of the paralelization method
    Inputs:
    UHamilold is the Hamiltonian taken from the paralelization method, which is not a matrix
    configurationslist is the list of configurations of the Fock space
    Returns the Hamiltonian in a matrix form
    """
    UHamil = np.matrix(np.zeros(((len(configurationslist),len(configurationslist))),dtype=np.complex_)) #This is the empty matrix that will become the Hamiltonian
    for row in range(len(UHamilold)): #loops over the rows of UHamilold
        for i in range(len(configurationslist)): #loops over the configurations in configurationslist
            UHamil[i,row] = UHamilold[row][i,0] #it makes the Hamiltonian
    return UHamil


def molecularorbitals_fromMB(eig_vectsSP,vectn, pos, configurationslist,psi,spin,n):
    """This function make the molecular orbitals from a many-body wavefunction
    Inputs:
    eig_vectsSP are the matrix of single-particle eigenvectors
    vectn
    pos is the list of position vectors (np.arrays)
    configurationslist is the list of configurations of the Fock Space
    psi is the many-body wavefunction
    spin is 'up' or 'down'
    n is the label for the molecular orbital to be returned
    Returns a list with the probability distribution of the nth orbital
    """
    orbitallist = [] #this is a list for the g*g distribution
    if spin == 'up': #if the spin is up
        for i in range(len(pos)): #loops over the sites
            taulista=[] #a list for g*np.conj(taupsij)*taupsij
            for jconf in range(len(configurationslist)): #it loops over all the configurations in the Hilbert space
                statej = configurationslist[jconf] #this is a many-body state
                if statej[n] == 1: #if there is an electron in the orbital n, continue
                    taupsij = psi[jconf] #takes the many-body coefficient
                    gncompl = np.conj(vectn[i]) #this is the single-particle coefficient in site i of the n orbital
                    gn = vectn[i] #this is the single-particle coefficient in site i of the n orbital
                    g = gncompl*gn #both are multiplied
                    taulista.append(g*np.conj(taupsij)*taupsij) #it is appended to the taulista list
            orbitallist.append(sum(taulista)) #the sum is appended to the orbitallist
    #the same as before but for spin down
    if spin == 'down':
        for i in range(len(pos)):
            taulista=[]
            for jconf in range(len(configurationslist)):
                statej = configurationslist[jconf]
                if statej[n+int(len(configurationslist[0])/2)] == 1:
                    taupsij = psi[jconf]
                    gncompl = np.conj(vectn[i])
                    gn = vectn[i]
                    g = gncompl*gn
                    taulista.append(g*np.conj(taupsij)*taupsij)
            orbitallist.append(sum(taulista))

    return orbitallist


def build_hamU_SPbasis_confbasis_SP_paral(U, eig_vectsSP, pos, configurationslist, Ne, Extra_e,  jconf):
    """This function builds the Hubbard hamiltonian with a configuration (many-body) basis. The configurations consist in the electrons fluctuating through the SP states inside the CAS.
    H = U sum_{nmn'm',i} psi_{uparrow n}(i)^* psi_{uparrow n'}(i) psi_{downarrow m}(i)^* psi_{downarrow m'}(i) c^dagger_{n,uparrow} c_{n',uparrow} c^dagger_{m,downarrow} c^dagger_{m',downarrow}
    Note: the sum can be brokendown into different terms, each of them will be computed separately in the following code, this will allow us to avoid computing some parts of the summation that are always 0.
    For instance, terms that move electrons between frozen valence states and the active space will always be 0 since they are not in the Hilbert space.
    This function is for the paralelization option
    Inputs:
    U is the Hubbard parameter
    eig_vectsSP is the matrix of single-particle eigenvector
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    Ne is the number of electrons in the CAS
    Extra_e is the number of extra electrons with respect the neutral state
    jconf is the label of a configuration in the Fock Space
    returns a matrix which is the Hubbard hamiltonian in the many-body basis
    """
    No_tot = len(pos) #Total single-particle states
    Ne_tot = len(pos) + Extra_e #Total of electrons
    Nconf = len(configurationslist) #Total of configurations
    Numberbotstates = int((Ne_tot - Ne)/2) #This is the number of valence states out of the Active Space
    UHamil = np.matrix(np.zeros(((len(configurationslist),1)),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations

    #the next chunk of code computes the summation term corresponding to the operators acting just on the active space
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

def plot_bondorder(rho,ijlist,pos, bondpos):
    """This function plots the lattice with bond orders as coloured bonds
    Inputs:
    rho is the density matrix
    ijlist is a list of size 2 lists that have the position indexes of each pair of sites that is connected
    pos is a list with the vectors of the lattice
    bondpos is a list with the positions of the bonds
    """
    X,Y,Z,ce = [],[],[],[] #empty lists
    for i in range(len(pos)): #loops over the sites
        r = pos[i]     # positions
        X.append(r[0])  #these 3 lines are the cartesian coordinates of r
        Y.append(r[1])
        Z.append(r[2])
        ce.append(0.5)
    fig = plt.figure() #it creates the figure
    ax = fig.add_subplot(111) #it creates the subplot
    for b in range(len(bondpos)): #this plots the bonds with a color that represents the bond order
        bond = bondpos[b]
        i = ijlist[b][0]
        j = ijlist[b][1]
        ax.plot(bond[0:2], bond[-2:], c=cm.seismic(np.real(rho[i,j])),linewidth=5,solid_capstyle='round')
    cs = ax.scatter(X,Y,s=20.0,edgecolors='none', c=ce,cmap = 'seismic') # Plots the lattice. c is the colour of the dots, s is the size.
    plt.gca().set_aspect('equal', adjustable='box') #this avoids a distortion in the picture
    cbar = plt.colorbar(cs, ticks=[0, 1]) #this ads a colorbar
    plt.show()

def CAS_better(No,Ne):
    """This function gets a list of the Fock states that conform the many body Hilbert Space given the number of electrons and number of orbitals
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

def loopUHam(configurationslist, Hamtdic, HamUdic, U):
    """This function takes the TB and Hubbard Hamiltonian from a dictionary (with Sz as labels and U=1) and return the eigenvalues and eigenvectors for a given U.
    This function was prepared to do loop over a list of Us.
    Inputs:
    configurationslist is a list of configurations of the Fock Space. This needs to be the list that was entered as input in the function "solve_Ham_bySz", not its output
    Hamtdic is a dictionary of the Sz subspaces of the TB Hamiltonian
    HamUdic is a dictionary of the Sz subspaces of the Hubard Hamiltonian
    U is the Hubbard parameter
    returns the eigenvalues and eigenvectors
    """
    Sztotallist = MB_Sz(len(configurationslist[0]), configurationslist) #this gives a list of the Sz labels in the Hilbert space
    spin_ordered_state_lists = Spin_ordered(configurationslist, Sztotallist) #this gives a dictionary ordering the Fock states by the Sz label
    eig_vals0list = [] #this is an empty list for the eigenvalues
    eig_vects0mat = np.matrix(np.zeros(((len(configurationslist), len(configurationslist))),dtype=np.complex_)) #this is an empty matrix for the eigenvectors

    countern = 0
    for x in spin_ordered_state_lists: #this loops over all the Sz labels in the Hilbert space
    #for x in ['1.0']: #you can select specific Sz labels to loop, instead of all of them
        configurationslist0 = spin_ordered_state_lists[x] #This is the list of configurations for a given Sz subspace
        tHamil0 = Hamtdic[x] #this is the TB Hamiltonian subspace with an Sz label
        UHamil0 = HamUdic[x]*U #this is the Hubbard Hamiltonian subspace with an Sz label with U=1, which then is multiplied by U
        Hamil0 = tHamil0 + UHamil0 #The two are summed
        eig_vals0, eig_vects0 = LA.linalg.eigh(Hamil0) #it gets the eigenvalues and eigenvectors
        eig_vects0 = eig_vects0.transpose()

        for l in eig_vals0: #This appends the eigenvalues in a list
            eig_vals0list.append(l)

        counternlist = [] #This keeps the eigenvectors in a matrix
        for n in range(len(eig_vals0)):
            for c in range(len(eig_vals0)):
                eig_vects0mat[n+countern,c+countern] = np.array(eig_vects0[n])[0][c]
            counternlist.append(1)
        countern += sum(counternlist)

    eig_vals0listordered, eig_vects0matordered = order_eigvals_eigvects(eig_vals0list, eig_vects0mat) #This orders the eigenvalues and eigenvectors, from the lowest eigenvalue to the highest

    return eig_vals0listordered, eig_vects0matordered


def solve_Ham_bySz(configurationslistfull, eig_valsSP, eig_vectsSP, pos, U, paral, loopU, No, Ne, Extra_e):
    """This function diagonalizes a Hamiltonian by parts labeled with Sz subspaces
    Inputs:
    configurationslistfull is a list with all the Fock states that form part of the Hilbert Space, in this function it will be divided in different lists labeled by Sz
    eig_valsSP are the single-particle eigenvalues
    eig_vectsSP are the single-particle eigenvectors
    pos is the list of vectors position of the atoms
    zeromodes is the number of zero modes in the system
    U is the Hubbard integral, it should be a positive number
    paral = 'yes' or 'no' if you want paralelization
    Ne is the number of electrons in the CAS
    Extra_e is the number of extra electrons with respect the neutral state
    returns the eigenvalues, eigenvectors and fock states in the correct order
    """
    Hamtdic = {} #dictionary to store the Sz labeled subpsaces of the TB Hamiltonian
    HamUdic = {} #dictionary to store the Sz labeled subpsaces of the Hubbard Hamiltonian
    Sztotallist = MB_Sz(len(configurationslistfull[0]), configurationslistfull) #This gets a list with the Sz numbers of the Fock states in configurationslistfull
    print('done')
    spin_ordered_state_lists = Spin_ordered(configurationslistfull, Sztotallist) #This gets a dictionary, classifying the Fock states by the Sz number
    print('done')

    eig_vects0mat = np.matrix(np.zeros(((len(configurationslistfull), len(configurationslistfull))),dtype=np.complex_)) #an empty matrix for the eigenvectors
    countern = 0
    configurationslistnew = [] #an empty list for the new Fock Space. The order of the configurations will change with respect to configurationslistfull

    eig_valslist = []  #This is the list that will gather the eigenvalues
    for Sz in spin_ordered_state_lists: #It runs over the Sz
        configurationslist = spin_ordered_state_lists[Sz] #It gets the Fock states labeled by Sz
        if paral == 'no':
            tHamil = build_ham_TB_MB_SPCAS(eig_valsSP, pos, configurationslist, No, Ne, Extra_e) #It gets the hopping many-body Hamiltonian
            UHamil = build_ham_Hubbard_MB_SPCAS(U, eig_vectsSP, pos, configurationslist, Ne, Extra_e) #It gets the Hubbard Hamiltonian
            ZeeHamil = build_ham_Zeeman_MB_SPCAS([0,0,1], pos, configurationslist)
        if paral == 'yes':
            num_cores = 4 #This is the number of cores for the paralelization
            #num_cores = int(os.getenv('OMP_NUM_THREADS'))
            #ZeeHamil = build_ham_Zeeman_MB_SPCAS([0,0,1], pos, configurationslist)
            tHamil = build_ham_TB_MB_SPCAS(eig_valsSP, pos, configurationslist, No, Ne, Extra_e) #It gets the hopping many-body Hamiltonian
            UHamilold = Parallel(n_jobs=num_cores)(delayed(build_hamU_SPbasis_confbasis_SP_paral)(U, eig_vectsSP, pos, configurationslist, Ne, Extra_e, jconf) for jconf in range(len(configurationslist))) #this gets the Hubbard Hamiltonian using paralelization
            print('hey')
            UHamil = build_HamUparal(UHamilold,configurationslist) #this gets the Hubbard Hamiltonian in the proper format

            Hamtdic[Sz] = tHamil #it stores the TB Hamiltonian in the dictionary
            HamUdic[Sz] = UHamil #it stores the Hubbard Hamiltonian in the dictionary
        Hamil = tHamil + UHamil #The separated Hamiltonians are joined in one matrix
        eig_vals, eig_vects = get_eigvals_eigvects_Herm(Hamil) #It diagonalizes Hamil

        print('done')
        for l in eig_vals: #It stores the eigenvalues in a list
            eig_valslist.append(l)
        for conf in configurationslist: #It stores the Fock states in a list
            configurationslistnew.append(conf)

        counternlist = [] #It stores the eigenvectors in a matrix
        for n in range(len(eig_vals)):
            print(n,'n')
            for c in range(len(eig_vals)):
                eig_vects0mat[n+countern,c+countern] = np.array(eig_vects[n])[0][c]
            counternlist.append(1)
        countern += sum(counternlist)
        print('done')

    eig_vals0listordered, eig_vects0matordered = order_eigvals_eigvects(eig_valslist, eig_vects0mat) #it gets the eigenvalues and eigenvectors ordered, from the lowest eigenvalue to the highest
    return eig_vals0listordered, eig_vects0matordered, Hamtdic, HamUdic, configurationslistnew

def order_eigvals_eigvects(eig_vals0list, eig_vects0mat):
    """This function takes a list of eigenvalues and eigenvectors. It returns the eigenvalues ordered from the lowest (which corresponds to the first eigenvectors row) to the highest (last eigenvectors row)
    Inputs:
    eig_vals0list is the list of disordered eigenvalues
    eig_vects0mat is the matrix of disordered eigenvectors
    It returns the ordered eigenvalues and eigenvectors
    """
    eig_vals = [] #empty list for the eigenvalues
    eig_vects = np.matrix(np.zeros(((len(eig_vals0list), len(eig_vals0list))),dtype=np.complex_)) #empty matrix for the eigenvectors
    countervect = 0
    while True:
        eig_valsi = min(eig_vals0list) #it takes the smallest eigenvalue in eig_vals0list
        i = eig_vals0list.index(eig_valsi) #it takes the index of that eigenvalue
        for e in range(len(eig_vals0list)): #it loops over the eigenvalues
            eig_vects[countervect,e] = np.array(eig_vects0mat[i])[0][e] #it stores the eigenvector
        countervect += 1 #countervect counts the eigenvector row
        eig_vals.append(eig_valsi) #it stores the eigenvalue
        eig_vals0list[i] = 100000000000000000000000000000 #this is to replace that eigenvalue for something else that makes no physical sense

        counterbreak = 0
        for m in eig_vals0list:
            if m != 100000000000000000000000000000: #if there is any eigenvalue left
                counterbreak += 1 #counterbreak counts if there is any eigenvalue left
        if counterbreak == 0: #if there is no eigenvalue left, then break the while loop and finish
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

def manybody_densmatrix(psi, eig_vectsSP, pos, configurationslist, Ne, Extra_e):
    """
    This function calculates the density matrix given a many-body wavefunction. In this sense, rho_ii is the sum of up and down densities <niup> + <nidown>, and rho_ij is <c^dagger_iupc_jup> + <c^dagger_idownc_jdown>
    Inputs:
    psi is the many-body wavefunction
    eig_vectsSP is the matrix of single-particle eigenvector
    pos is the list with the position vectors of the molecule
    configurationslist is the list of Fock configurations in binary
    Ne is the number of electrons in the CAS
    Extra_e is the number of extra electrons with respect the neutral state
    returns the density matrix rhomatrix
    """
    No_tot = len(pos) #Total single-particle states
    Ne_tot = len(pos) + Extra_e #Total of electrons
    Nconf = len(configurationslist) #Total of configurations
    Numberbotstates = int((Ne_tot - Ne)/2) #This is the number of valence states out of the Active Space
    rhomatrix = np.matrix(np.zeros(((len(pos),len(pos))),dtype=np.complex_)) #This is the empty matrix that will become the Hubbard Hamiltonian
    confdic = confdic_gen(configurationslist) #calls for the dictionary of configurations

    for jconf in range(len(configurationslist)): #it loops over all the configurations in the Hilbert space
        print (jconf,'U') #This is just to monitor where we are
        tauj = psi[jconf]
        #This is for the Active Space states up electrons
        for n in range(0, int(len(configurationslist[0])/2), 1):  #it loops over the up states of the active space
            vectn = np.array(eig_vectsSP[n+Numberbotstates])[0]   #it gets the eigenvector
            for nprime in range(0, int(len(configurationslist[0])/2), 1): #it loops over the up states of the active space
                vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates])[0]  #it gets the eigenvector

                statej = [] #it makes a copy of the state
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
                        taui = psi[iconf]
                        #The next 9 lines are to calculate the density matrix
                        for i in range(len(pos)):
                            ri = pos[i]
                            for j in range(len(pos)):
                                rj = pos[j]
                                gncompl = np.conj(vectn[i])
                                gn = vectnprime[j]
                                fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                                rhonnprime = fermionic_sign*gn*gncompl*tauj*taui
                                rhomatrix[i, j] = rhomatrix[i, j] + rhonnprime

        #This is for the Active Space states down electrons
        for n in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1):  #it loops over the down states of the active space
            vectn = np.array(eig_vectsSP[n+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]   #it gets the eigenvector
            for nprime in range(int(len(configurationslist[jconf])/2), int(len(configurationslist[jconf])/2) + int(len(configurationslist[0])/2), 1): #it loops over the down states of the active space
                vectnprime = np.array(eig_vectsSP[nprime+Numberbotstates-int(len(configurationslist[jconf])/2)])[0]  #it gets the eigenvector
                statej = [] #it creates a copy of the state
                for obj in configurationslist[jconf]:
                    statej.append(obj)

                if statej[nprime] == 1: #if there is an electron in the orbital n', continue
                #The next 7 lines are to calculate the fermionic sign for the acting on orbital n'
                    epsilonanihiuplist = [Numberbotstates*2]
                    for fockindex in range(0, nprime, 1):
                        if fockindex < nprime:
                            epsilonanihiuplist.append(statej[fockindex])
                        if fockindex == 0 and nprime == 0:
                            epsilonanihiuplist.append(0.0)
                    epsilonanihiup = sum(epsilonanihiuplist)
                    statej[nprime] = 0 #it destroys an electron in the orbital labeled as nprime
                    if statej[n] == 0: #if there is not an electron in the orbital n, continue
                    #The next 7 lines are to calculate the fermionic sign for the acting on orbital n
                        epsiloncreatuplist = [Numberbotstates*2]
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
                        taui = psi[iconf]
                        #The next 9 lines are to create the density matrix
                        for i in range(len(pos)):
                            ri = pos[i]
                            for j in range(len(pos)):
                                rj = pos[j]
                                gncompl = np.conj(vectn[i])
                                gn = vectnprime[j]
                                fermionic_sign = ((-1)**epsilonanihiup)*((-1)**epsiloncreatup)
                                rhonnprime = fermionic_sign*gn*gncompl*tauj*taui
                                rhomatrix[i, j] = rhomatrix[i, j] + rhonnprime

        #This is for the electrons out of the Active Space
        for n in range(0, Numberbotstates, 1):  #it loops over the states out of the active space
            vectn = np.array(eig_vectsSP[n])[0]   #it gets the eigenvector
            nprime = n
            vectnprime = np.array(eig_vectsSP[nprime])[0]  #it gets the eigenvector
            for i in range(len(pos)):
                ri = pos[i]
                for j in range(len(pos)):
                    rj = pos[j]
                    gncompl = np.conj(vectn[i])
                    gn = vectnprime[j]
                    rhonnprime = 2*gn*gncompl*tauj*tauj
                    rhomatrix[i, j] = rhomatrix[i, j] + rhonnprime 
    return rhomatrix
