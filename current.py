#This functions are to do calculations of the orbital current

def build_currentop_nospinSP(a,b,pos, t):
    """This function builds the single-particle current operator
    """
    n = len(pos)
    Iop = np.matrix(np.zeros((n,n),dtype=np.complex_)) #makes empty zero matrix with dimensions equal to the number of atoms
    for i in [a]:
        ri = pos[i]     #generates a quantity of variables equal to len of pos and gives them the value of each vector in the list
        for j in [b]:
            rj = pos[j]      #generates variables equal to len of pos and gives them the value of each vector in the list
            dist = np.linalg.norm(rj - ri)
            if abs(dist - 2.424871131) < 0.1 :   #if two atoms are bond, place t in the matrix's entrance ([i,j] and [j,i])
            #if abs(dist - 1.212435565*2) < 0.1 :   #if two atoms are bond, place t in the matrix's entrance ([i,j] and [j,i])
                Iop[i, j] = (-0.00000000000000000016021766208*1j/(0.000000000000000658211962468))*t
                Iop[j, i] = (-0.00000000000000000016021766208*1j/(0.000000000000000658211962468))*(-t)
    return Iop

def current_sp(pos, t, Ndist, psi):
    """This function creates a dictionary with the Iab current for a single-particle spinless eigenvector.
    Inputs:
    pos is a list with the position vectors
    t is the hopping parameter
    Ndist is the distance between a and b
    psi is the single-particle eigenvector to calculate the current
    returns a dictionary with the current for each a,b pair
    """
    currentdict = {} #This dictionary stores the Iab current between a and b sites
    for a in range(len(pos)): #loops over sites
        ri = pos[a] #ri is a position vector
        for b in range(a,len(pos)): #loops over sites again
            rj = pos[b] #rj is another position vector
            dist = np.linalg.norm(rj - ri) #distance between ri and rj
            if abs(dist - Ndist) < 0.1: #if dist is around Ndist
                Iop = build_currentop_nospinSP(a, b, pos, t) #It calls for a function that creates the current operator for a spinless TB hamil
                Iab = psi*Iop*np.conj(psi.transpose()) #it calculates the current
                currentdict[str(a)+','+str(b)] = np.array(Iab[0])[0][0] #it stores Iab in the dictionary
    return currentdict

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
