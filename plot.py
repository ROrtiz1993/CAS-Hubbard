#This file contains functions for plotting
import numpy as np
import matplotlib.pyplot as plt

def plot_lattice_2D(pos, bondpos):
    """This function plots a planar lattice
    Inputs:
    pos is a list of arrays with the position vectors
    bondpos is a list with the position of the bonds
    """
    X,Y = [],[]
    for i in range(len(pos)):
        r = pos[i]    #These are the positions
        X.append(r[0])
        Y.append(r[1])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for bond in bondpos: #Plots the bonds
        ax1.plot(bond[0:2], bond[-2:], 'k-')

    for i in range(len(pos)): # Plots the lattice with numbers, comment this line and the next one to remove the numbers
        plt.text(X[i],Y[i],i, ha="center", va="center", weight='bold')

    ax1.scatter(X,Y,c='w',s=50.0,edgecolors='none') # Plots the lattice. c is for the colours of the atoms, s for the size
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def bondpos_graph(pos, a):
    """This function gives the position of the bonds in a system.
    Inputs:
    pos is a list of arrays with the position vectors
    a is the bondlength
    returns a list with the positions of the bonds
    """
    bondpos = []
    for i in range(len(pos)):
        ri = pos[i]
        for j in range(len(pos)):
            rj = pos[j]
            dist = np.linalg.norm(rj - ri)
            if (abs(dist - a) < 0.1):
                bondpos.append([ri[0], rj[0], ri[1], rj[1]])
    return bondpos

def plot_espectrum(t, eigvals):
    """This function plots the eigenvalues of a hamiltonian.
    Inputs:
    t is the first-neighbours hopping element
    eigvals is the list of eigenvalues to be plotted
    """
    eigenvalue_numberlist = [] #This gives a list with a number labeling each eigenvalue
    for i in range(len(eigvals)):
        eigenvalue_numberlist.append(i)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if t == 1 or t == -1: #This is for the units of the y axis. If t=1, then it is in units of t; otherwise it is in eV units.
        ax.set_ylabel("E(t)")
    if t != 1 and t != -1:
        ax.set_ylabel("E(eV)")
    ax.set_xlabel("State number")
    ax.scatter(eigenvalue_numberlist, eigvals, c = 'b',s=50., edgecolors='none')

    plt.show()

def plot_wavefunction(pos, bondpos, eigvects, n):
    """This function plots the lattice of a planar system and the wavefunction (|phi_n|^2) of a selected eigenstate
    Inputs:
    pos is a list of arrays with the position vectors
    bondpos is a list with the position of the bonds
    eigvects is a matrix with the eigenvectors
    n labels the eigenvector
    """
    X, Y, S = [], [], []
    for i in range(len(pos)):
        r = pos[i]     #These are the positions
        v = np.array(eigvects[n])[0][i]    # eigvector component corresponding to site i
        X.append(r[0])
        Y.append(r[1])
        Prob = (abs(v*np.conj(v))) #|phi_n|^2
        S.append(Prob*7000) #This is to scale it up

    fig = plt.figure(figsize = (20,10))
    ax = fig.add_subplot(111)
    font = {'size'   : 20}
    plt.rc('font', **font)

    for bond in bondpos:    #This plots the bonds
        ax.plot(bond[0:2], bond[-2:], 'k-',linewidth=1)

    ax.scatter(X,Y,c='r',s=S,edgecolors='none', alpha = 1)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()
