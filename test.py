import hamiltonians as hamiltonians
import vects as vects
import plot as plot
import numpy as np
a=1.4
t=-2.7
No,Ne = 4,4
Extra_e = 0
pos = vects.vects_rectangular_graph0D(6, 3, a)
#pos = [np.array([0.0,0.0,0.0]),np.array([1.4,0.0,0.0]),np.array([0.7,1.4*np.sqrt(3)/2,0.0])]
print(pos)
bondpos = plot.bondpos_graph(pos, a)
plot.plot_lattice_2D(pos, bondpos)
hamt = hamiltonians.buildham_TB_SP_nospin(pos, t, a)
eig_valsSP, eig_vectsSP = hamiltonians.get_eigvals_eigvects_Herm(hamt)
print(eig_valsSP)
plot.plot_espectrum(t, eig_valsSP)
plot.plot_wavefunction(pos, bondpos, eig_vectsSP, int(len(pos)/2)-1)
configurationslist0 = hamiltonians.CAS_better(No,Ne)
eig_vals0listordered, eig_vects0matordered, Hamtdic, HamUdic, configurationslistnew = hamiltonians.solve_Ham_bySz(configurationslist0, eig_valsSP, eig_vectsSP, pos,2.7,'yes','no', No, Ne, Extra_e)
print(eig_vals0listordered-eig_vals0listordered[0])
print(configurationslistnew)
print(eig_vects0matordered[0])
print(eig_vects0matordered[1])
print (hamiltonians.manybody_densmatrix(np.array(eig_vects0matordered[0])[0], eig_vectsSP, pos, configurationslistnew, Ne, Extra_e))
