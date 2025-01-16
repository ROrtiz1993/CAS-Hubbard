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

def vects_zigzag1D(width, a):
    """This function gives the vectors of the unit cell for a 1D zigzag graphene ribbon of known width
    Inputs:
    width is the number of hexagonal units
    a is the bond length
    returns a list with the position vectors
    """
    vI = np.array([0.0, 0.0, 0.0])
    vII = np.array([a*np.sqrt(3)/2, -a*0.5, 0.0])
    vIII = np.array([a*np.sqrt(3)/2, -a*1.5, 0.0])
    vIV = np.array([0.0, -a*2, 0.0])
    cellpos =[vI,vII,vIII,vIV] #These are the positions of the unit cell
    pos = [vI,vII,vIII,vIV] #This is the position list, it starts with the unit cell

    countloopi = 1 #These are counters to handle the while loop, and tell it when to stop
    countx = 1
    while True: #This while loop creates the vectors of the whole unit cell, it loops until we have a number of hexagon units equal to the width
        for i in cellpos:
            if countx == width:
                break
            newvector = np.array([i[0], i[1] - countx*a*3, i[2]])
            pos.append(newvector)
            if countloopi % 4 == 0: #for every four atoms add 1 to countx
                countx += 1
            countloopi += 1
        if countx == width: #if the number of hexagon units is equal to the width then break the loop
            break
    return pos

def vects_armchair1D(width, a):
    """This function gives the vectors of the unit cell for a 1D armchair graphene ribbon of known width
    Inputs:
    width is the number of hexagonal units
    a is the bond length
    returns a list with the position vectors
    """
    vI = np.array([0.0, 0.0, 0.0])
    vII = np.array([a, 0.0, 0.0])
    vIII = np.array([a*1.5, -a*np.sqrt(3)/2, 0.0])
    vIV = np.array([a*2.5, -a*np.sqrt(3)/2, 0.0])
    vV = np.array([0.0, -a*np.sqrt(3), 0.0])
    vVI = np.array([a, -a*np.sqrt(3), 0.0])
    vVII = np.array([a*1.5, -3*a*np.sqrt(3)/2, 0.0])
    vVIII = np.array([a*2.5, -3*a*np.sqrt(3)/2, 0.0])
    cellpos =[vV,vVI,vVII,vVIII] #These are the positions of the unit cell
    pos = [vI,vII,vIII,vIV,vV,vVI,vVII,vVIII] #This is the position list, it starts with 8 atoms

    countloopi = 1 #These are counters to handle the while loop, and tell it when to stop
    countx = 1
    while True: #This while loop creates the vectors of the whole unit cell, it loops until we have a number of hexagon units equal to the width
        for i in cellpos:
            if countx == width:
                break
            newvector = np.array([i[0], i[1] - countx*a*np.sqrt(3), i[2]])
            pos.append(newvector)
            if countloopi % 4 == 0: #for every four atoms add 1 to countx
                countx += 1
            countloopi += 1
        if countx == width: #if the number of hexagon units is equal to the width then break the loop
            break
    return pos

def vects_triangullenes(n,a):
    """This function gives the vectors of the atoms for triangular zigzag graphene flakes known as [n]triangulenes
    Inputs:
    n is an integer. It is the number of atoms of one zigzag edge. n must be bigger than 1.
    a is the bondlength
    returns a list with the position vectors
    """
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
    cellpos =[vII,vIII,vIV,vV,vXI,vXII] #These are the positions of the unit cell
    pos = [vI,vII,vIII,vIV,vV,vVI,vVII,vVIII,vIX,vX,vXI,vXII,vXIII] #This is the position list, it starts with the 13 atoms of phenalenyl

    countloopi = 1 #These are counters to handle the while loop, and tell it when to stop
    countx = 1
    while True: #This while loop creates the vectors of the whole unit cell, it loops until we have a number of zigzag atoms in one edge is equal to n
        if n == 2: #This limits the smallest molecule (n=2) to be the phenalenyl
            break
        for i in cellpos: #it loops over all the atoms in the cellpos
            vn = np.array([i[0] + (countx * np.sqrt(3)*a), i[1], i[2]]) #it creates a new atom displaced in x from the unit cell
            pos.append(vn)
            if countloopi % (6+2*(countx-1)) == 0: #this creates the missing atoms to complete the triangular flake
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
        if countx == n-1: #break the loop if n is equal to the number of zigzag atoms in one edge
            break
    return pos
