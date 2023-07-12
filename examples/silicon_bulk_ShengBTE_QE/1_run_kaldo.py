# Runs kALDo on crystal silicon using force constants calculated on a variable
# number of k-points in Quantum Espresso (aka how many images).
#
# Usage: python 1_run_kaldo.py <replicas_per_axis> <u/n> <disp> <overwrite>
# u/n controls if kALDo unfolds the force constants
# disp will exit the calculations after making a dispersion
# overwrite allows the script to replace data written in previous runs
#
# WARNING: Please note, that crystal silicon is sometimes represented with
# atoms at negative coordinates ((0,0,0)+(-1/4, 1/4, 1/4)) but shengbte forces
# it to be represented as ((0,0,0)+(1/4, 1/4, 1/4)). Similar differences in
# representation across interfaces will result in unphysical output from kALDo
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']=""
import numpy as np
from ase.io import read
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.controllers.plotter import plot_dispersion

# Number of replicas controlled by command line argument
nrep = int(sys.argv[1][0])
supercell, folder = np.array([nrep, nrep, nrep]), '{}x{}x{}'.format(nrep, nrep, nrep)
third_supercell = np.array([3,3,3])

# Parameter controlling k-pt grid
k = 9
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k,k,k)

# Detect whether you want to unfold
unfold_bool = False
unfold = 'n'
if 'u' in sys.argv[1]:
    unfold_bool = True
    unfold = 'u'

# Dispersion information
npoints = 150
pathstring = 'GXULG' # edit this to change dispersion path
atoms = read('3x3x3/POSCAR', format='vasp')
cell = atoms.cell
lat = cell.get_bravais_lattice()
path = cell.bandpath(pathstring, npoints=npoints)
print('Unit cell detected: {}'.format(atoms))
print('Special points on cell:')
print(lat.get_special_points())
print('Path: {}'.format(path))
if 'disp' in sys.argv:
    only_dispersion = True

# Control data IO + overwriting controls
overwrite = False; prefix='data'
outfolder = prefix+'/{}{}'.format(nrep, unfold)
print('\n\n\t\tPlotting for supercell {}x{}x{} -- '.format(nrep, nrep, nrep))
print('\t\t Unfolding (u/n): {}'.format(unfold))
print('\t\t In folder:       {}'.format(folder))
print('\t\t Out folder:      {}'.format(outfolder))
if 'overwrite' in sys.argv:
    overwrite = True
if os.path.isdir(prefix):
    print('!! - '+prefix+' directory already exists')
    if os.path.isdir(outfolder):
        print('!! - '+outfolder+' directory already exists')
        print('!! - continuing may overwrite, or load previous data')
        if not overwrite:
            exit()
else:
    os.mkdir(prefix)

# You shouldn't need to edit below this line
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

forceconstant = ForceConstants.from_folder(
                       folder=folder,
                       supercell=supercell,
                       third_supercell=third_supercell,
                       is_acoustic_sum=True,
                       format='shengbte-qe')
#if unfold: # unfold the third order force constants too
#    unfolded_third = forceconstant.unfold_third_order(distance_threshold=4)
#    forceconstant.third.value = unfolded_third
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder=outfolder,
              is_unfolding=unfold,
              storage='numpy')

# Dispersion - plotted by an imported function from our `controllers/plotter.py`
plot_dispersion(phonons, is_showing=False,
            manually_defined_path=path, folder=outfolder+'/dispersion')

# Conductivity - different methods of calculating the conductivity can be compared
# but full inversion of the three-phonon scattering matrix typically agrees with
# experiment more than say the relaxation time approximation.
# Calculating this quantity will produce as a byproduct things like the phonon
# bandwidths, phase space etc.
cond = Conductivity(phonons=phonons, method='inverse', storage='numpy')
inv_cond_matrix = cond.conductivity.sum(axis=0)
diag = np.diag(inv_cond_matrix)
offdiag = np.abs(inv_cond_matrix).sum() - np.abs(diag).sum()
print('Conductivity from full inversion (W/m-K):\n%.3f' % (np.mean(diag)))
print('Sum of off-diagonal terms: %.3f' % offdiag)
print('Full matrix:')
print(inv_cond_matrix)