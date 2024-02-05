"""
kaldo
Anharmonic Lattice Dynamics
"""
import numpy as np
from sparse import COO

from kaldo.grid import wrap_coordinates
from kaldo.helpers.logger import get_logger
from kaldo.observables.secondorder import SecondOrder
from kaldo.observables.thirdorder import ThirdOrder
from ase.io import read, write

logging = get_logger()

MAIN_FOLDER = 'displacement'


class ForceConstants:
    """
    Class for constructing the finite difference object to calculate
    the second/third order force constant matrices after providing the
    unit cell geometry and calculator information.

    Parameters
    ----------

    atoms: Tabulated xyz files or ASE Atoms object
        The atoms to work on.
    supercell: (3) tuple, optional
        Size of supercell given by the number of repetitions (l, m, n) of
        the small unit cell in each direction.
        Defaults to (1, 1, 1)
    third_supercell: tuple, optional
        Same as supercell, but for the third order force constant matrix.
        If not provided, it's copied from supercell.
        Defaults to `self.supercell`
    folder: str, optional
        Name to be used for the displacement information folder.
        Defaults to 'displacement'
    distance_threshold: float, optional
        If the distance between two atoms exceeds threshold, the interatomic
        force is ignored.
        Defaults to `None`
    """

    def __init__(self,
                 atoms,
                 supercell=(1, 1, 1),
                 third_supercell=None,
                 folder=MAIN_FOLDER,
                 distance_threshold=None):

        # Store the user defined information to the object
        self.atoms = atoms
        self.supercell = supercell
        self.n_atoms = atoms.positions.shape[0]
        self.n_modes = self.n_atoms * 3
        self.n_replicas = np.prod(supercell)
        self.n_replicated_atoms = self.n_replicas * self.n_atoms
        self.cell_inv = np.linalg.inv(atoms.cell)
        self.folder = folder
        self.distance_threshold = distance_threshold
        self._list_of_replicas = None

        # TODO: we should probably remove the following initialization
        self.second = SecondOrder.from_supercell(atoms,
                                                 supercell=self.supercell,
                                                 grid_type='C',
                                                 is_acoustic_sum=False,
                                                 folder=folder)
        if third_supercell is None:
            third_supercell = supercell
        self.third = ThirdOrder.from_supercell(atoms,
                                               supercell=third_supercell,
                                               grid_type='C',
                                               folder=folder)

        if distance_threshold is not None:
            logging.info('Using folded IFC matrices.')

    @classmethod
    def from_folder(cls, folder, supercell=(1, 1, 1), format='numpy', third_energy_threshold=0., third_supercell=None,
                    is_acoustic_sum=False, only_second=False, distance_threshold=None):
        """
        Create a finite difference object from a folder

        Parameters
        ----------
        folder : str
            Chosen folder to load in system information.
        supercell : (int, int, int), optional
            Number of unit cells in each cartesian direction replicated to form the input structure.
            Default is (1, 1, 1)
        format : 'numpy', 'eskm', 'lammps', 'shengbte', 'shengbte-qe', 'hiphive'
            Format of force constant information being loaded into ForceConstants object.
            Default is 'numpy'
        third_energy_threshold : float, optional
            When importing sparse third order force constant matrices, energies below
            the threshold value in magnitude are ignored. Units: ev/A^3
                Default is `None`
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than the
            distance threshold will be ignored.
        third_supercell : (int, int, int), optional
            Takes in the unit cell for the third order force constant matrix.
            Default is self.supercell
        is_acoustic_sum : Bool, optional
            If true, the accoustic sum rule is applied to the dynamical matrix.
            Default is False

        Inputs
        ------
        numpy: replicated_atoms.xyz, second.npy, third.npz
        eskm: CONFIG, replicated_atoms.xyz, Dyn.form, THIRD
        lammps: replicated_atoms.xyz, Dyn.form, THIRD
        shengbte: CONTROL, POSCAR, FORCE_CONSTANTS_2ND/FORCE_CONSTANTS, FORCE_CONSTANTS_3RD
        shengbte-qe: CONTROL, POSCAR, espresso.ifc2, FORCE_CONSTANTS_3RD
        hiphive: atom_prim.xyz, replicated_atoms.xyz, model2.fcs, model3.fcs


        Returns
        -------
        ForceConstants object
        """
        second_order = SecondOrder.load(folder=folder, supercell=supercell, format=format,
                                        is_acoustic_sum=is_acoustic_sum)
        atoms = second_order.atoms
        # Create a finite difference object
        forceconstants = {'atoms': atoms,
                          'supercell': supercell,
                          'folder': folder}
        forceconstants = cls(**forceconstants)
        forceconstants.second = second_order
        if not only_second:
            if format == 'numpy':
                third_format = 'sparse'
            else:
                third_format = format
            if third_supercell is None:
                third_supercell = supercell
            third_order = ThirdOrder.load(folder=folder, supercell=third_supercell, format=third_format,
                                          third_energy_threshold=third_energy_threshold)

            forceconstants.third = third_order
        forceconstants.distance_threshold = distance_threshold
        return forceconstants

    def unfold_third_order(self, reduced_third=None, distance_threshold=None):
        """
        This method extrapolates a third order force constant matrix from a unit
        cell into a matrix for a larger supercell.

        Parameters
        ----------

        reduced_third : array, optional
            The third order force constant matrix.
            Default is `self.third`
        distance_threshold : float, optional
            When calculating force constants, contributions from atoms further than
            the distance threshold will be ignored.
            Default is self.distance_threshold
        """
        logging.info('Unfolding third order matrix')
        if distance_threshold is None:
            if self.distance_threshold is not None:
                distance_threshold = self.distance_threshold
            else:
                raise ValueError('Please specify a distance threshold in Angstrom')

        logging.info('Distance threshold: ' + str(distance_threshold) + ' A')
        if (self.atoms.cell[0, 0] / 2 < distance_threshold) | \
                (self.atoms.cell[1, 1] / 2 < distance_threshold) | \
                (self.atoms.cell[2, 2] / 2 < distance_threshold):
            logging.warning('The cell size should be at least twice the distance threshold')
        if reduced_third is None:
            reduced_third = self.third.value
        n_unit_atoms = self.n_atoms
        atoms = self.atoms
        n_replicas = self.n_replicas
        replicated_cell_inv = np.linalg.inv(self.third.replicated_atoms.cell)

        reduced_third = reduced_third.reshape(
            (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3))
        replicated_positions = self.third.replicated_atoms.positions.reshape((n_replicas, n_unit_atoms, 3))
        dxij_reduced = wrap_coordinates(atoms.positions[:, np.newaxis, np.newaxis, :]
                                        - replicated_positions[np.newaxis, :, :, :], self.third.replicated_atoms.cell,
                                        replicated_cell_inv)
        indices = np.argwhere(np.linalg.norm(dxij_reduced, axis=-1) < distance_threshold)

        coords = []
        values = []
        for index in indices:
            for l in range(n_replicas):
                for j in range(n_unit_atoms):
                    dx2 = dxij_reduced[index[0], l, j]

                    is_storing = (np.linalg.norm(dx2) < distance_threshold)
                    if is_storing:
                        for alpha in range(3):
                            for beta in range(3):
                                for gamma in range(3):
                                    coords.append([index[0], alpha, index[1], index[2], beta, l, j, gamma])
                                    values.append(reduced_third[index[0], alpha, 0, index[2], beta, 0, j, gamma])

        logging.info('Created unfolded third order')

        shape = (n_unit_atoms, 3, n_replicas, n_unit_atoms, 3, n_replicas, n_unit_atoms, 3)
        expanded_third = COO(np.array(coords).T, np.array(values), shape)
        expanded_third = expanded_third.reshape(
            (n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
        return expanded_third

    def df(self, calculator, replicated_cell, delta_shift=1e-3, with_sigma2=True):
        atoms = self.atoms
        replicated_atoms = replicated_cell
        n_unit_atoms = atoms.positions.shape[0]
        n_replicas = np.prod(self.supercell)
        atoms.set_calculator(calculator)
        replicated_atoms.set_calculator(calculator)
        new_super = replicated_atoms.copy()
        new_atoms = atoms.copy()
        R0 = replicated_atoms.get_positions()
        sup_d = delta_shift*np.ones(new_super.get_positions().shape)
        prim_d = delta_shift*np.ones(new_atoms.get_positions().shape)
        new_atoms.set_calculator(calculator)
        new_super.set_calculator(calculator)
        forces = []
        two_forces = []
        three_forces = []
        s2 = []
        s3 = []
        for s in [-2, -1, 1, 2]:
            new_super.positions = replicated_atoms.get_positions() + s*sup_d
            new_atoms.positions = atoms.get_positions() + s*prim_d
            delta = np.array(new_super.get_positions() - R0).flatten()
            forces.append(new_atoms.get_forces().flatten())
            phi_2 = self.second.value.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            two_forces.append(np.dot(phi_2, delta))
            phi_3 = self.third.value.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            three_forces.append(np.dot(np.dot(phi_3, delta), delta))
            forces = np.array(forces)
            two_forces = np.array(two_forces)
            three_forces = np.array(three_forces)
            s2.append(np.sqrt(np.mean((forces - two_forces)**2))/forces.std())
            s3.append(np.sqrt(np.mean((forces - two_forces - three_forces)**2))/forces.std())
            atoms.positions = new_atoms.positions
            replicated_atoms = new_super.positions
        sigma2 = np.mean(s2)
        sigma3 = np.mean(s3)
        if with_sigma2:
            return sigma2, sigma3
        else:
            return sigma3

    def df_trajectory(self, replica_atoms, calculator, traj_file='dump.xyz', with_sigma2=True):
        traj_atoms = read(traj_file, format='extxyz')
        forces = []
        two_forces = []
        three_forces = []
        R0 = replica_atoms.get_positions()
        for a in range(len(traj_atoms)):
            sup_atoms = traj_atoms[a]
            sup_atoms.set_calculator(calculator)
            delta = np.array(sup_atoms.get_positions() - R0).flatten()
            forces.append(sup_atoms.get_forces().flatten())
            phi_2 = self.second.value.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            two_forces.append(np.dot(phi_2, delta).flatten())
            phi_3 = self.third.value.reshape((n_unit_atoms * 3, n_replicas * n_unit_atoms * 3, n_replicas * n_unit_atoms * 3))
            three_forces.append(np.dot(np.dot(phi_3, delta), delta).flatten())
            #forces = np.array(forces)
            #two_forces = np.array(two_forces)
            #three_forces = np.array(three_forces)
            #s2.append(np.sqrt(np.mean((forces - two_forces)**2))/forces.std())
            #s3.append(np.sqrt(np.mean((forces - two_forces - three_forces)**2))/forces.std())
        forces = np.array(forces)
        two_forces = np.array(two_forces)
        three_forces = np.array(three_forces)
        #sigma2 = np.mean(s2)
        #sigma3 = np.mean(s3)
        sigma2 = np.sqrt(np.mean((forces - two_forces)**2)/forces.std())
        sigma3 = np.sqrt(np.mean((forces - two_forces - three_forces)**2)/forces.std())
        if with_sigma2:
            return sigma2, sigma3
        else:
            return sigma3

