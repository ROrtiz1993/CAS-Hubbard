#This file has the functions for obtaining lists of arrays that are the position vectors for different systems.
import numpy as np

def vects_rectangular_graph0D(width, long, a):
    """This function defines the vectors of a 0D graphene ribbon with defined armchair and zigzag edges.
    Inputs:
    width is the number of hexagonal units in the armchair direction
    long is the number of hexagonal units in the zigzag direction
    returns a list with the position vectors
    """

    vI = np.array([0.0, 0.0, 0.0]) #These are the positions for four initial sites, this will serve as unit cell
    vII = np.array([a/2, a*np.sqrt(3)/2, 0.0])
    vIII = np.array([3*a/2, a*np.sqrt(3)/2, 0.0])
    vIV = np.array([2*a, 0.0, 0.0])
    cellpos = [vI,vII,vIII,vIV] #These are the positions of the unit cell
    pos = [vI,vII,vIII,vIV] #This is the position list, it starts with the unit cell
    xcord = [vI[0],vII[0],vIII[0],vIV[0]] #This keeps the x position of every atom of the first line, it starts with x coordinate of the unit cell
    ycord = [vI[1],vII[1],vIII[1],vIV[1]] #This keeps the y position of every atom of the first column, it starts with y coordinate of the unit cell

    countloopi = 1 #These are counters to handle the while loop, and tell it when to stop
    countx = 1
    while True: #This while loop creates the vectors of the cells of the first line,
        for i in cellpos: #at the end the number of cells created + the unit cell will be equal to the width and the loop will be broken
            if countx == width:
                break
            vnx = np.array([i[0] + countx * 3*a, i[1], i[2]])
            pos.append(vnx)
            xcord.append(vnx[0])
            if countloopi % 4 == 0:
                countx += 1
            countloopi += 1
        if countx == width:
            break

    countloopj = 1 #These are counters to handle the while loop, and tell it when to stop
    county = 1
    countwidth = 1
    while True: #This while loop creates the vectors of the cells of the first column (except those of the unit cell)
        for j in cellpos: #and also for every cell created fills that line with a number of cells equal to the width
            if county == long:
                break
            if county < long:	#This creates the first column
                vny = np.array([j[0], j[1] - county * a*np.sqrt(3), j[2]])
                pos.append(vny)
                ycord.append(vny[1])
            if width > 1 and county < long: #This fills the lines
                for widthpos in range(1, width):
                    vnxy = np.array([xcord[4 * widthpos + countwidth - 1], j[1] - county * a*np.sqrt(3), j[2]])
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
        vnclose1 = np.array([xcord[1 + 4 * edgexpos], ycord[1 + 4 * longminusone] - a*np.sqrt(3), 0.0])
        vnclose2 = np.array([xcord[2 + 4 * edgexpos], ycord[2 + 4 * longminusone] - a*np.sqrt(3), 0.0])
        pos.append(vnclose2)
        pos.append(vnclose1)
    return pos
