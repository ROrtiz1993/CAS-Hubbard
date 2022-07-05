import numpy as np
from numpy import linalg as LA
from numpy import matrix
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import scipy.sparse.linalg as lg
import pylab
import codeCAS_github as hub
from joblib import Parallel, delayed
import multiprocessing

hub.save_representation(['firstcheck'], [''], 'checks')

edgelength = 3
t = -2.7
U = abs(1.5*t)
No = 8
Ne = 8
Nbot = 4
Ntop = 4
zeromodes = 0
addelectron = 0


#pos = hub.vects_triangullenes(edgelength,1.42)
#pos = hub.vects_Mullen0D(6, 3)
pos = hub.transpentachain_FM(8)
bondpos = hub.bondpos_graph(pos,1.42)

#eigvects = hub.read_and_write_eigvalues()
#print(len(pos))
#print(len(eigvects))
#hub.plot_wavefunction_sign(pos, eigvects,eigvects,  bondpos)

hub.plot_lattice(pos, bondpos)
h = hub.build_hamtnospin(pos, t,1.42)
eig_valsSP, eig_vectsSP = LA.linalg.eigh(h)
eig_vectsSP = eig_vectsSP.transpose()
print(eig_valsSP)
hub.plot_espectrum(t, eig_valsSP)

Enumber = len(pos)+addelectron


configurationslist0 = hub.CAS_better(No,Ne)
print(len(configurationslist0))

print(configurationslist0[1])
print(configurationslist0[len(configurationslist0)-1])





hub.save_representation(['checkbeforeCAS'], [''], 'checks')



eig_vals0listordered, eig_vects0matordered, Hamtdic, HamUdic, configurationslistnew = hub.solve_Ham_bySz(configurationslist0, eig_valsSP, eig_vectsSP, pos, zeromodes, Nbot, Ntop,U,'yes','Yes')

hub.save_representation(['doneCAS'], [''], 'checks')

for Unew in range(0,51):
    Unew = 2.7*Unew/10
    eig_vals0listordered, eig_vects0listordered = hub.loopUHam(configurationslist0, Hamtdic, HamUdic, Unew)

    print (eig_vals0listordered-eig_vals0listordered[0],Unew)

    #Hamil0 = tHamil0 + UHamil0*Unew
    #eig_vals0listordered, eig_vects0matordered = LA.linalg.eigh(Hamil0)
    #eig_vects0matordered = eig_vects0matordered.transpose()
    for i in range(100):
        hub.save_representation([eig_vals0listordered[i]-eig_vals0listordered[0]], [Unew], 'isomer2_N8_CAS(8,8)'+str(i))
    #print(t,'t')
    #for i in range(20):
    #save_representation([eig_vals0listordered[i]-eig_vals0listordered[0]], [tthird], i

    #print(eig_vects0[6])
st
for i in range(30):
    hub.save_representation([str(eig_vals0listordered[i])], [''], 'vals')
    print (eig_vals0listordered[i])
print(eig_vects0matordered[0])
st
print((configurationslist0))
print(eig_vects0matordered[0])

num_cores = 4
#num_cores = int(os.getenv('OMP_NUM_THREADS'))
#save_representation([num_cores+'num_cores'], [''], 'checks')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 6, zeromodes, Nbot, Ntop,13,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 13, zeromodes, Nbot, Ntop,20,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 20, zeromodes, Nbot, Ntop,27,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 27, zeromodes, Nbot, Ntop,34,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 34, zeromodes, Nbot, Ntop,41,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 41, zeromodes, Nbot, Ntop,48,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = Parallel(n_jobs=num_cores)(delayed(hub.correlator_map_improved_paral)(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 48, zeromodes, Nbot, Ntop,55,jconf) for jconf in range(len(configurationslistnew)))
hub.save_representation([sum(maplist)], [''], 'corrmap')
st
"""
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 6, zeromodes, Nbot, Ntop,13)
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 13, zeromodes, Nbot, Ntop,20)
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 20, zeromodes, Nbot, Ntop,27)
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 27, zeromodes, Nbot, Ntop,34)
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 34, zeromodes, Nbot, Ntop,41)
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 41, zeromodes, Nbot, Ntop,48)
hub.save_representation([sum(maplist)], [''], 'corrmap')
maplist = hub.correlator_map_improved(pos, eig_vectsSP, eig_vects0matordered[0], configurationslistnew, 48, zeromodes, Nbot, Ntop,55)
hub.save_representation([sum(maplist)], [''], 'corrmap')
"""
#print(eig_vects0matordered[0])
#print(eig_vects0matordered[3])
