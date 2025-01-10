#These functions are for saving lists, matrixes, xyz, xsf, etc. and reading them

def write_xsf_current(filename, pos, current):
    """ This function creates an xsf format file with the pos vects and the current vectors. It considers all the sites to be C atoms
    Inputs:
    filename is a string with the name of the file
    pos is a list with the position vectors of the sites
    current is a dictionary that stores the current for pairs of sites with str(a,b) as keys
    """
    filename2=filename+'.xsf' #it adds .xsf to the name of the file
    natom = len(pos) #number of sites (atoms)
    st='\n'+'ATOMS'+'\n' #it creates a line with "ATOMS"
    f=open(filename2,'w') #write mode
    f.write(st) #writes "ATOMS"
    for i in list(range(len(pos))): #for each site
        st= 'C  '+str(np.real(pos[i][0]))+' '+str(np.real(pos[i][1]))+' '+ str(np.real(pos[i][2])) +'\n' #st is the line with the atom and position in xyz format
        f.write(st) #writes st
    #This part reads the a and b indexes from the dictionary key
    for x in current:
        alist = []
        blist = []
        counter = 0
        for l in x:
            if l == ',':
                counter += 1
            if l != ',' and counter == 0:
                alist.append(float(l))
            if l != ',' and counter > 0:
                blist.append(float(l))
        alistagain = []
        blistagain = []
        for w in range(len(alist)):
            alistagain.append(alist[w]*(10**(len(alist)-w-1)))
        for w in range(len(blist)):
            blistagain.append(blist[w]*(10**(len(secondlist)-w-1)))
        a = sum(alistagain)
        b = sum(blistagain)
        #This part writes the Iab vector in xsf format, considering whether it is positive or negative
        Iab = np.real(current[x])
        if Iab > 0.0:
            posa = pos[int(a)]
            posb = pos[int(b)]
            posI = posb - posa
            st= 'X  '+str(np.real(posa[0]))+' '+str(np.real(posa[1]))+' '+str(np.real(posa[2]))+ '     '+ str(np.real(Iab*posI[0]))+' '+ str(np.real(Iab*posI[1]))+' ' + '0.'+'\n'
            f.write(st)
        if Iab < 0.0:
            Iab = abs(Iab)
            posa = pos[int(a)]
            posb = pos[int(b)]
            posI = posa - posb
            st= 'X  '+str(np.real(posb[0]))+' '+str(np.real(posb[1]))+' '+str(np.real(posb[2]))+ '     '+ str(np.real(Iab*posI[0]))+' '+ str(np.real(Iab*posI[1]))+' ' + '0.'+'\n'
            f.write(st)


def read_xyz(filename):
    """This function reads an xyz file and returns a list with position vectors as np.arrays, it doesnt distinguish if there are different atomic elements
    Inputs:
    filename is a string with the name of the file to be created
    """
    pos = [] #a list with that will store the vectors
    file = open(filename+".xyz", 'r') #it opens the xyz file in read mode
    lines = file.readlines() #it takes all the lines of the file, each one of the after the second one is an atom
    counter = 0 #a counter that monitors in which line we are in the next for loop
    for line in lines: #for each line in lines
        countx = 0 #these counters
        county = 0
        countz = 0
        x = '' #creates three empty objects x y z, that will be the coordinates
        y = ''
        z = ''
        if counter > 1: #this ignores the first and second lines
            for i in range(len(line)): #it runs through the line
                if i > 0: #it ignores the atom symbol, this should be 1 if the symbol has more than one letter
                    #the next lines just store the coordinates in x y z and create the pos vector
                    if countx == 0: #countx, county and countz are to differentiate the three coordinates in the line
                        if line[i] != ' ':
                            countx += 1
                    if countx == 1 and line[i] == ' ':
                        countx += 1
                    if countx == 2 and line[i] != ' ':
                        countx += 1
                        county += 1
                    if county == 1 and line[i] == ' ':
                        county += 1
                    if county == 2 and line[i] != ' ':
                        county += 1
                        countz += 1
                    if countz == 1 and line[i] == ' ':
                        countz += 1

                    if countx == 1:
                        x += line[i]
                    if county == 1:
                        y += line[i]
                    if countz == 1:
                        z += line[i]
            pos.append(np.array([float(x), float(y), float(z)]))
        counter += 1 #it goes to the next line
    return pos

def read_list():
    """This function reads a .dat document with one value per line and creates a list object with them
    """
    list1 = []
    file = open('list1.dat', 'r')
    lines = file.readlines()
    for line in lines:
        x = ''
        for i in range(len(line)):
            x += line[i]
            if line[i] == ' ':
                break
        list1.append(float(x))
    return list1

def save_representation(leftlist, rightlist, name):
    """This function takes two lists and writes them in a .dat document in two columns, so they can be read to draw a chart
    Inputs:
    leftlist and rightlist are lists for the left and right columns
    name is a string with the name for the document
    """
    fil= name + '.dat'
    for i in list(range(len(y))):
        leftlisti = leftlist[i]
        rightlisti = rightlist[i]
        with open(fil, 'a') as f:
            f.write(str(np.real(leftlisti)) + ' ' + str(rightlisti) + '\n')
