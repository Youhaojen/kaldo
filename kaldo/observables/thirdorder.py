from kaldo.observables.forceconstant import ForceConstant
from ase import Atoms
import os
import ase.io
import numpy as np
from scipy.sparse import load_npz, save_npz
from sparse import COO
from kaldo.interfaces.eskm_io import import_from_files
import kaldo.interfaces.shengbte_io as shengbte_io
import ase.units as units
from kaldo.controllers.displacement import calculate_third
from kaldo.helpers.logger import get_logger
from ase.build import make_supercell

logging = get_logger()


REPLICATED_ATOMS_THIRD_FILE = 'replicated_atoms_third.xyz'
REPLICATED_ATOMS_FILE = 'replicated_atoms.xyz'
THIRD_ORDER_PROGRESS = 'third_order_displacements'
THIRD_ORDER_FILE_SPARSE = 'third.npz'
THIRD_ORDER_FILE = 'third.npy'

class ThirdOrder(ForceConstant):

    @classmethod
    def load(cls, folder, supercell=(1, 1, 1), format='sparse', third_energy_threshold=0., distance_threshold=None, third_order_delta=None):
        """
        Create a finite difference object from a folder
        :param folder:
        :param supercell:
        :param format:
        :param third_energy_threshold:
        :param is_acoustic_sum:
        :return:
        """

        if format == 'sparse':

            if folder[-1] != '/':
                folder = folder + '/'
            try:
                config_file = folder + REPLICATED_ATOMS_THIRD_FILE
                replicated_atoms = ase.io.read(config_file, format='extxyz')
            except FileNotFoundError:
                config_file = folder + REPLICATED_ATOMS_FILE
                replicated_atoms = ase.io.read(config_file, format='extxyz')

            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])

            _third_order = COO.from_scipy_sparse(load_npz(folder + THIRD_ORDER_FILE_SPARSE)) \
                .reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            third_order = ThirdOrder(atoms=atoms,
                                     replicated_positions=replicated_atoms.positions,
                                     supercell=supercell,
                                     value=_third_order,
                                     folder=folder)

        elif format == 'eskm' or format == 'lammps':
            if format == 'eskm':
                config_file = str(folder) + "/CONFIG"
                replicated_atoms = ase.io.read(config_file, format='dlp4')
            elif format == 'lammps':
                config_file = str(folder) + "/replicated_atoms.xyz"
                replicated_atoms = ase.io.read(config_file, format='extxyz')

            third_file = str(folder) + "/THIRD"
            n_replicas = np.prod(supercell)
            n_total_atoms = replicated_atoms.positions.shape[0]
            n_unit_atoms = int(n_total_atoms / n_replicas)
            unit_symbols = []
            unit_positions = []
            for i in range(n_unit_atoms):
                unit_symbols.append(replicated_atoms.get_chemical_symbols()[i])
                unit_positions.append(replicated_atoms.positions[i])
            unit_cell = replicated_atoms.cell / supercell

            atoms = Atoms(unit_symbols,
                          positions=unit_positions,
                          cell=unit_cell,
                          pbc=[1, 1, 1])


            out = import_from_files(replicated_atoms=replicated_atoms,
                                                third_file=third_file,
                                                supercell=supercell,
                                                third_energy_threshold=third_energy_threshold)
            third_order = ThirdOrder(atoms=atoms,
                                     replicated_positions=replicated_atoms.positions,
                                     supercell=supercell,
                                     value=out[1],
                                     folder=folder)


        elif format == 'shengbte' or format == 'shengbte-qe':
            grid_type='F'
            config_file = folder + '/' + 'CONTROL'
            try:
                atoms, supercell = shengbte_io.import_control_file(config_file)
            except FileNotFoundError as err:
                config_file = folder + '/' + 'POSCAR'
                logging.info('\nTrying to open POSCAR')
                atoms = ase.io.read(config_file)

            third_file = folder + '/' + 'FORCE_CONSTANTS_3RD'

            third_order = shengbte_io.read_third_order_matrix(third_file, atoms, supercell, order='C')
            third_order = ThirdOrder.from_supercell(atoms=atoms,
                                                    grid_type=grid_type,
                                                    supercell=supercell,
                                                    value=third_order,
                                                    folder=folder)

        elif format == 'hiphive':
            filename = 'atom_prim.xyz'
            # TODO: add replicated filename in example
            replicated_filename = 'replicated_atoms.xyz'
            try:
                import kaldo.interfaces.hiphive_io as hiphive_io
            except ImportError:
                logging.error('In order to use hiphive along with kaldo, hiphive is required. \
                      Please consider installing hihphive. More info can be found at: \
                      https://hiphive.materialsmodeling.org/')

            atom_prime_file = str(folder) + '/' + filename
            replicated_atom_prime_file = str(folder) + '/' + replicated_filename
            # TODO: Make this independent of replicated file
            atoms = ase.io.read(atom_prime_file)
            replicated_atoms = ase.io.read(replicated_atom_prime_file)

            if 'model3.fcs' in os.listdir(str(folder)):
                # Derive constants used for third-order reshape
                supercell = np.array(supercell)
                n_prim = atoms.copy().get_masses().shape[0]
                n_sc = np.prod(supercell)
                dim = len(supercell[supercell > 1])
                _third_order = hiphive_io.import_third_from_hiphive(atoms, supercell, folder)
                _third_order = _third_order[0].reshape(n_prim * dim, n_sc * n_prim * dim,
                                                                       n_sc * n_prim * dim)
                third_order = cls(atoms=atoms,
                                  replicated_positions=replicated_atoms.positions,
                                  supercell=supercell,
                                  value=_third_order,
                                  folder=folder)

        elif format == 'trajectory':
            logging.info('calculating third order from trajectory')
            # TODO: make these defaults changeable -> kwargs
            filename = folder+'third_order_displacements/forces.lmp'
            format = 'lammps-dump-text'
            atoms = ase.io.read(folder+'atoms.xyz', format='xyz')
            replicated_atoms = ase.io.read(folder+'replicated_atoms.xyz', format='xyz')

            n_atoms = len(atoms)
            n_replicas = np.prod(supercell)
            n_replicated_atoms = len(replicated_atoms)
            i_at_sparse = []
            i_coord_sparse = []
            j_at_sparse = []
            j_coord_sparse = []
            k_sparse = []
            _third_order = []
            n_forces_to_calculate = n_replicas * (n_atoms * 3) ** 2
            n_forces_done = 0
            n_forces_skipped = 0

            # ASE reads all frames into memory at once, even when specifying a single index, so best just call this once
            frames = ase.io.read(filename, format=format, index=':')
            num_frames = len(frames)
            frame_index = 0
            if ((num_frames * 4) != n_forces_to_calculate) and distance_threshold is None:
                logging.info('number of frames loaded incorrect.')
                exit()
            logging.info('found %i frames' % num_frames)

            for iat in range(n_atoms):
                for jat in range(n_replicas * n_atoms):
                    is_computing = True
                    m, j_small = np.unravel_index(jat, (n_replicas, n_atoms))
                    if (distance_threshold is not None):
                        dxij = atoms.positions[iat] - replicated_atoms.positions[jat]
                        if (np.linalg.norm(dxij) > distance_threshold):
                            is_computing = False
                            n_forces_skipped += 9
                    if is_computing:
                        for icoord in range(3):
                            for jcoord in range(3):
                                _partial_third = np.zeros((n_replicated_atoms * 3))
                                for isign in (1, -1):
                                    for jsign in (1, -1):
                                        single_frame = frames[frame_index]
                                        forces = single_frame.get_forces().flatten()
                                        dpotential = forces / (isign * third_order_delta)
                                        dpotential = dpotential / (jsign * third_order_delta)
                                        _partial_third[:] += -1 * dpotential / 4
                                        frame_index += 1

                                for id in range(_partial_third.shape[0]):
                                    i_at_sparse.append(iat)
                                    i_coord_sparse.append(icoord)
                                    j_at_sparse.append(jat)
                                    j_coord_sparse.append(jcoord)
                                    k_sparse.append(id)
                                    _third_order.append(_partial_third[id])
                        n_forces_done += 9
                    if (n_forces_done + n_forces_skipped % 300) == 0:
                        logging.info('loading third derivatives ' + str
                        (int((n_forces_done + n_forces_skipped) / n_forces_to_calculate * 100)) + '%')
            coords = np.array([i_at_sparse, i_coord_sparse, j_at_sparse, j_coord_sparse, k_sparse])
            shape = (n_atoms, 3, n_replicas * n_atoms, 3, n_replicas * n_atoms * 3)
            _third_order = COO(coords, np.array(_third_order), shape)
            _third_order = _third_order.reshape((n_atoms * 3, n_replicas * n_atoms * 3, n_replicas * n_atoms * 3))
            third_order = cls(atoms=atoms,
                              replicated_positions=replicated_atoms.positions,
                              supercell=supercell,
                              value=_third_order,
                              folder=folder)
        else:
            logging.error('Third order format not recognized: ' + str(format))
            raise ValueError
        return third_order


    def save(self, filename='THIRD', format='sparse', min_force=1e-6):
        folder = self.folder
        filename = folder + '/' + filename
        n_atoms = self.atoms.positions.shape[0]
        if format == 'eskm':
            logging.info('Exporting third in eskm format')
            n_replicas = self.n_replicas
            n_replicated_atoms = n_atoms * n_replicas
            tenjovermoltoev = 10 * units.J / units.mol
            third = self.value.reshape((n_atoms, 3, n_replicated_atoms, 3, n_replicated_atoms, 3)) / tenjovermoltoev
            with open(filename, 'w') as out_file:
                for i in range(n_atoms):
                    for alpha in range(3):
                        for j in range(n_replicated_atoms):
                            for beta in range(3):
                                value = third[i, alpha, j, beta].todense()
                                mask = np.argwhere(np.linalg.norm(value, axis=1) > min_force)
                                if mask.any():
                                    for k in mask:
                                        k = k[0]
                                        out_file.write("{:5d} ".format(i + 1))
                                        out_file.write("{:5d} ".format(alpha + 1))
                                        out_file.write("{:5d} ".format(j + 1))
                                        out_file.write("{:5d} ".format(beta + 1))
                                        out_file.write("{:5d} ".format(k + 1))
                                        for gamma in range(3):
                                            out_file.write(' {:16.6f}'.format(third[i, alpha, j, beta, k, gamma]))
                                        out_file.write('\n')
            logging.info('Done exporting third.')
        elif format=='sparse':
            config_file = folder + '/' + REPLICATED_ATOMS_THIRD_FILE
            ase.io.write(config_file, self.replicated_atoms, format='extxyz')

            save_npz(folder + '/' + THIRD_ORDER_FILE_SPARSE, self.value.reshape((n_atoms * 3 * self.n_replicas *
                                                                           n_atoms * 3, self.n_replicas *
                                                                           n_atoms * 3)).to_scipy_sparse())
        else:
            super(ThirdOrder, self).save(filename, format)



    def calculate(self, calculator, delta_shift, distance_threshold=None, trajectory=False, is_storing=True, is_verbose=False):
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        atoms.set_calculator(calculator)
        replicated_atoms.set_calculator(calculator)
        if is_storing:
            try:
                self.value = ThirdOrder.load(folder=self.folder, supercell=self.supercell).value

            except FileNotFoundError:
                logging.info('Third order not found. Calculating.')
                self.value = calculate_third(atoms,
                                             replicated_atoms,
                                             delta_shift,
                                             distance_threshold=distance_threshold,
                                             is_verbose=is_verbose)
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')
            else:
                logging.info('Reading stored third')
        else:
            self.value = calculate_third(atoms,
                                         replicated_atoms,
                                         delta_shift,
                                         distance_threshold=distance_threshold,
                                         is_verbose=is_verbose)
            if is_storing:
                self.save('third')
                ase.io.write(self.folder + '/' + REPLICATED_ATOMS_THIRD_FILE, self.replicated_atoms, 'extxyz')


    def store_displacements(self, delta_shift, distance_threshold=None, format='extxyz', trajectory=True, dir=THIRD_ORDER_PROGRESS):
        '''
        Code that will store configuration files in a folder for use in parallel computing
        of force constants. Defaults to 'second_order_displacements'

        Parameters
        ----------
        delta_shift: float
            How far to move the atoms. This is a sensitive parameter
            for force constant calculations.

        distance_threshold: float
            How far apart two atoms can be to have their third order deriv considered.

        trajectory: bool
            whether or now to format as an extxyz trajectory.

        dir: string
            Where to place the xyz files. Directory created if
            not existant.
            Defaults to 'second_order_xyz'
        '''
        if not os.path.isdir(dir):
            os.mkdir(dir)
        extensions = {
            'extxyz' : '.xyz',
            'xyz':'.xyz',
            'lammps-dump-text':'.lmp',
            'lammps-dump-binary':'.lmp'
        }
        atoms = self.atoms
        replicated_atoms = self.replicated_atoms
        positions = replicated_atoms.positions

        copied_atoms = replicated_atoms.copy()
        n_unit_cell_atoms = len(atoms.numbers)
        n_replicated_atoms = len(replicated_atoms.numbers)
        n_replicas = n_replicated_atoms // n_unit_cell_atoms
        box = np.max(replicated_atoms.positions)

        # TODO: implement multi-element support here
        copied_atoms.set_chemical_symbols([1] * n_replicated_atoms)
        mag = len(str(n_unit_cell_atoms))
        n_displacements = n_unit_cell_atoms * 3 * 2 * n_replicated_atoms * 3 * 2
        logging.info('%i, %i, %i' % (n_unit_cell_atoms, n_replicated_atoms, n_displacements))
        alpha = ['x', 'y', 'z']
        move_names = [None, '+', '-']
        logging.info('Storing '+str(n_displacements)+' frames with '+str(delta_shift)+'A shift')
        logging.info('Writing to '+str(dir))
        n_frames_skipped = 0; n_frames_written = 0
        for iat in range(n_unit_cell_atoms):
            for jat in range(n_replicated_atoms):
                is_computing = True
                m, j_small = np.unravel_index(jat, (n_replicas, n_replicated_atoms))
                if (distance_threshold is not None):
                    dxij = atoms.positions[iat] - replicated_atoms.positions[jat]
                    if (np.linalg.norm(dxij) > distance_threshold):
                            is_computing = False
                            n_frames_skipped += 9
                if is_computing:
                    for icoord in range(3):
                        for jcoord in range(3):
                                    for isign in (1, -1):
                                        for jsign in (1, -1):
                                            n_frames_written +=1
                                            shift = np.zeros((n_replicated_atoms, 3))
                                            shift[iat, icoord] += isign * delta_shift
                                            shift[jat, jcoord] += jsign * delta_shift
                                            coords = positions + shift
                                            if np.max(coords) > box:
                                                index = np.argmax(coords)
                                                coords[np.unravel_index(index, coords.shape)] -= box
                                            if np.min(coords) < box:
                                                index = np.argmin(coords)
                                                coords[np.unravel_index(index, coords.shape)] += box
                                            copied_atoms.positions = coords
                                            if trajectory:
                                                ase.io.write(dir+'/'+'trajectory'+extensions[format],
                                                             images = copied_atoms,
                                                             format = format,
                                                             columns = ['numbers', 'positions'],
                                                             append = True)
                                            else:
                                                filestring = str(iat).zfill(mag) + alpha[icoord] + move_names[isign]
                                                filestring += '_' + str(jat).zfill(mag) + alpha[jcoord] + move_names[
                                                    jsign]
                                                ase.io.write(dir+'/'+filestring+extensions[format],
                                                             images = copied_atoms,
                                                             format = format,
                                                             columns = ['numbers', 'positions'])
        logging.info('Third order displacements stored, %i skipped, %i written ' % (n_frames_skipped, n_frames_written))


    def __str__(self):
        return 'third'
