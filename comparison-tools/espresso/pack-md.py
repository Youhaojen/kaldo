from ase.units import Rydberg, Bohr
from ase.io import read
import subprocess as sp
import numpy as np
import pandas as pd
import os
np.set_printoptions(linewidth=300, suppress=True)

rtoll = 5e-3
atoll = 1e-6 #for forces
format = 'espresso'
matdynoutfile = 'md.out.txt'
output_text_file = 'md.tmp.txt'
repacked_fn = 'md.out'
atoms = read('forces/POSCAR', format='vasp')
cell = atoms.cell.array
cellt = cell.T / (cell.max() * 2)
n_unit = len(atoms)
ninteractions = n_unit**2

# Conversions
RyBr_to_eVA = Rydberg/(Bohr**2)
force_prefactor = RyBr_to_eVA if format=='espresso' else 1


print('Processing {}'.format(matdynoutfile))
print('Creating/Loading intermediate text files ..')
# DATA OUTPUT BY MATDYN.F90
# 0-2  3-5  6  7  8     9     10 11 12        13        14          15
# q    n    na nb alpha beta  w  qr eiqr-real eiqr-imag totfrc-real totfrc-imag
if not os.path.exists(output_text_file):
    sp.run('grep "FORCES  " {} > {}'.format(matdynoutfile, output_text_file), shell=True)
raw = pd.read_csv(output_text_file, header=None, usecols=np.arange(16)+1,
    delim_whitespace=True).to_numpy()

__, qind= np.unique(raw[:, :3], axis=0, return_index=True)
q_unique = raw[np.sort(qind), :3]
sc_unique = np.unique(raw[:, 3:6], axis=0)

# Setup new data
# qvector, supercell, q dot r, eiqr, weight, forces
ii = 0
warningstring = ''
repack_shape = q_unique.shape[0]*sc_unique.shape[0]
repack_raw_type = [("qvec", float, (3)),
                  ("sc", int, (3)),
                  ("qr", float),
                  ("eiqr", complex),
                  ("weights", float, (n_unit, n_unit)),
                  ("forces", complex, (n_unit, 3, n_unit, 3))]
repack = np.zeros(repack_shape, dtype=repack_raw_type)
for nq,q in enumerate(q_unique): # loop over q
    qp = q @ cellt
    q_args = np.prod(np.isclose(raw[:, :3], q), axis=1)
    print('Processing {}-th q: {} (md equiv: {})'.format(nq, qp, q))
    for sc in sc_unique: # loop over supercells
        sc_args = np.prod((raw[:,3:6]==sc), axis=1)
        sum_args = (q_args*sc_args).astype(bool)
        total_matches = raw[sum_args]

        if total_matches.shape[0]==0: # missing SC on a q-point
            print('\n\n\t\tSC missing on q-point')
            print('\tii {} | q {} | sc {} | Supercell absent in dataset'.format(ii, sc, matrix[:, 10]))
            exit(1) # fatal

        #print('\tSupercell: {}'.format(sc))

        # Check if phase-arg is the same for every entry
        unique_phase_arg, index_upa = np.unique(total_matches[:, 11], axis=0, return_index=True)
        if unique_phase_arg.size != 1:
            print('\n\n\tNon-constant phase-arg detected at single q-point')
            print('ii {} | q {} | sc {} - Phase warning: {}\n'.format(ii, qk, sc, unique_phase_arg))
            exit(1) # fatal
        qr = unique_phase_arg.flatten()

        # Check if phase is the same for every entry
        unique_phases, index_uph = np.unique(total_matches[:, 12:14], axis=0, return_index=True)
        if unique_phases.size != 2:
            print('\n\n\tNon-constant phase detected at single q-point')
            print('ii {} | q {} | sc {} - Phase warning: {}\n'.format(ii, qk, sc, unique_phases))
            exit(1) # fatal
        eiqr = unique_phases[0, 0] + 1j*unique_phases[0, 1]

        # Loop over interactions
        weights = np.zeros((n_unit, n_unit))
        forces  = np.zeros((n_unit, 3, n_unit, 3), dtype=complex)
        interactions, index_uint = np.unique(total_matches[:, 6:8], axis=0, return_index=True)

        #print('\tTotal Interactions Found: {}'.format(index_uint.size))

        for nanb in interactions:
            na, nb = int(nanb[0]-1), int(nanb[1]-1) # removes fortran offset
            matrix = total_matches[np.prod((total_matches[:, 6:8]==nanb), axis=1).astype(bool)]

            # Are all weights the same for this 3x3 matrix (and any copies)?
            if (matrix[:, 10].sum() == 0.): # Interaction carries no weight
                continue
            elif not np.isclose(matrix[0, 10], np.mean(matrix[:, 10]), rtol=rtoll):
                print('\n\n\t\tMultiple weights exists for one interaction')
                print('\tii {} | q {} | sc {} | Weight mismatch - Weights: {}'.format(ii, qk, sc, matrix[:, 10]))
                exit(1)  # Fatal error
            else:
                weights[na, nb] = matrix[0, 10]

            # Fill (2,3,2,3) force matrix
            if (matrix.shape[0] % 9) != 0.:
                # We have unfilled 3x3 matrices -- Weirdest case. Defective matdyn.f90
                print('\t\tNA-NB: {} {}-{}'.format(nanb, na, nb))
                print('\t\tN force entries: {}'.format(matrix.shape[0]))
                print('\t\tN matrices expected: {}'.format(matrix.shape[0]%9))
                print('\t\tn-unique shape: {}'.format(np.unique(matrix[:, 6:8], axis=0).shape))
                print('\t\talpha-betas\n', matrix[:, 8:10])
                print('ii {} | q {} | sc {} - Force N-component warning\n'.format(ii, qk, sc))
                exit(1) # fatal
            elif matrix.shape[0]==9: # The whole 3x3 matrix is unique
                for row in matrix:   # Best case scenario
                    alpha, beta, = int(row[8]-1), int(row[9]-1)
                    nanbab = (na, alpha, nb, beta)
                    forces[na, alpha, nb, beta] = force_prefactor * (row[14] + 1j * row[15])
            else: # Implying there are multiple 3x3 matrices.
                for row in matrix:  # Probably copies from a duplicated q-point (e.g. gamma)
                    alpha, beta, = int(row[8] - 1), int(row[9] - 1)
                    nanbab = (na, alpha, nb, beta)
                    oforce = forces[nanbab]
                    nforce = (row[14] + 1j * row[15])
                    is_zero = (oforce.real==0 and oforce.imag==0)
                    if (not is_zero) and (not np.isclose(np.abs(nforce), np.abs(oforce), 1e-6, atol=atoll)):
                        print('\t\tWarning, overwriting on {}'.format(nanbab))
                        print(matrix)
                        warningstring += 'ii {} | q {} | sc {} | Force mismatch - Difference: {}\n'.format(ii, qk, sc,
                                                                                                           absdiff)
                        print(warningstring)
                        exit()
                    forces[nanbab] = (row[14] + 1j * row[15])
        # Looped over force components for q-vec + supercell
        # Push filled temp arrays to output array
        repack[ii] = qp, sc, qr, eiqr, weights, forces
        ii += 1

repack['forces'] *= force_prefactor
np.save(repacked_fn, repack)

# Compile and print warnings
print('''
##################################################################################
##################################################################################
WARNINGS DETECTED DURING PROCESSING  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
{}
##################################################################################
##################################################################################
'''.format(warningstring))
print('completed repackage succesful!')
# Detect if we missed a supercell at any q-point
if ii!=repack.shape[0]:
    print('zero rows exist')
    print(ii-repack.shape[0])
exit(0)
