#Here there are some miscelaneous functions

def rotational_matrix3DZaxisX(vects, degree):
    """This function gets some vects and it returns them rotated by a given degree in the Z axis
    Inputs:
    vects are the vectors to be rotated
    degree is the rotation angle in radians
    """
    Rmatrix = np.matrix(np.zeros(((3,3)),dtype=np.complex_))
    Rmatrix[1,1] = np.cos(degree)
    Rmatrix[1,2] = -np.sin(degree)
    Rmatrix[2,1] = np.sin(degree)
    Rmatrix[2,2] = np.cos(degree)
    Rmatrix[0,0] = 1
    rotvects = []
    for atom in vects:
        rotvect =(np.dot(Rmatrix,np.array([atom[0], atom[1], atom[2]]).transpose()))
        rotvect = np.array(rotvect)[0]
        rotvects.append(np.array([rotvect[0], rotvect[1], rotvect[2]]))
    return rotvects
