"""
kaldo
Anharmonic Lattice Dynamics
"""

from hiphive import ForceConstants 
import numpy as np

def import_second_from_hiphive(folder, n_replicas, n_atoms):
    second_hiphive_file = str(folder) + '/model2.fcs'
    fcs2 = ForceConstants.read(second_hiphive_file)
    second_order = fcs2.get_fc_array(2).transpose(0, 2, 1, 3)
    second_order = second_order.reshape((n_replicas, n_atoms, 3,
                                 n_replicas, n_atoms, 3))
    second_order = second_order[0, np.newaxis]
    return second_order

def import_third_from_hiphive(atoms, supercell, folder):
    third_hiphive_file  = str(folder) + '/model3.fcs'
    supercell = np.array(supercell)

    # Derive constants used for third-order reshape
    n_prim = atoms.positions.shape[0]
    n_sc = np.prod(supercell)
    dim = len(supercell[supercell > 1])
    fcs3 = ForceConstants.read(third_hiphive_file)
    third_order = fcs3.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5).reshape(n_sc, n_prim, dim,
            n_sc, n_prim, dim, n_sc, n_prim,dim)
    return third_order