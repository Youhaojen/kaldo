from opt_einsum import contract
from ballistico.helpers.tools import apply_boundary_with_cell
from scipy.linalg.lapack import dsyev
import numpy as np
import ase.units as units

KELVINTOTHZ = units.kB / units.J / (2 * np.pi * units._hbar) * 1e-12
KELVINTOJOULE = units.kB / units.J
THZTOMEV = units.J * units._hbar * 2 * np.pi * 1e15
EVTOTENJOVERMOL = units.mol / (10 * units.J)

DELTA_DOS = 1
NUM_DOS = 100
FOLDER_NAME = 'ald-output'


def calculate_occupations(phonons):
    frequencies = phonons.frequency
    temp = phonons.temperature * KELVINTOTHZ
    density = np.zeros_like(frequencies)
    physical_modes = phonons.physical_modes.reshape((phonons.n_k_points, phonons.n_modes))
    if phonons.is_classic is False:
        density[physical_modes] = 1. / (np.exp(frequencies[physical_modes] / temp) - 1.)
    else:
        density[physical_modes] = temp / frequencies[physical_modes]
    return density


def calculate_heat_capacity(phonons):
    frequencies = phonons.frequency
    c_v = np.zeros_like(frequencies)
    physical_modes = phonons.physical_modes.reshape((phonons.n_k_points, phonons.n_modes))
    temperature = phonons.temperature * KELVINTOTHZ
    if (phonons.is_classic):
        c_v[physical_modes] = KELVINTOJOULE
    else:
        f_be = phonons.population
        c_v[physical_modes] = KELVINTOJOULE * f_be[physical_modes] * (f_be[physical_modes] + 1) * phonons.frequency[
            physical_modes] ** 2 / \
                              (temperature ** 2)
    return c_v


def calculate_frequencies(phonons, q_points=None):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = apply_boundary_with_cell(q_points)
    eigenvals = calculate_eigensystem(phonons, q_points, only_eigenvals=True)
    frequencies = np.abs(eigenvals) ** .5 * np.sign(eigenvals) / (np.pi * 2.)
    return frequencies.real


def calculate_dynmat_derivatives(phonons, q_points=None):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = apply_boundary_with_cell(q_points)
    atoms = phonons.atoms
    list_of_replicas = phonons.finite_difference.list_of_replicas
    replicated_cell = phonons.finite_difference.replicated_atoms.cell
    replicated_cell_inv = phonons.finite_difference.replicated_cell_inv
    dynmat = phonons.finite_difference.dynmat
    positions = phonons.finite_difference.atoms.positions

    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]
    ddyn = np.zeros((n_k_points, n_unit_cell * 3, n_unit_cell * 3, 3)).astype(np.complex)
    for index_k in range(n_k_points):
        qvec = q_points[index_k]
        if phonons._is_amorphous:
            dxij = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dxij = apply_boundary_with_cell(dxij, replicated_cell, replicated_cell_inv)
            dynmat_derivatives = contract('ija,ibjc->ibjca', dxij, dynmat[:, :, 0, :, :])
        else:
            list_of_replicas = list_of_replicas
            dxij = positions[:, np.newaxis, np.newaxis, :] - (
                    positions[np.newaxis, np.newaxis, :, :] + list_of_replicas[np.newaxis, :, np.newaxis, :])
            dynmat_derivatives = contract('ilja,ibljc,l->ibjca', dxij, dynmat, phonons.chi(qvec))
        ddyn[index_k] = dynmat_derivatives.reshape((phonons.n_modes, phonons.n_modes, 3), order='C')
    return ddyn


def calculate_sij(phonons, q_points=None, is_antisymmetrizing=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = apply_boundary_with_cell(q_points)
    if is_main_mesh:
        dynmat_derivatives = phonons._dynmat_derivatives
        eigenvects = phonons._eigensystem[:, 1:, :]
    else:
        dynmat_derivatives = calculate_dynmat_derivatives(phonons, q_points)
        eigenvects = calculate_eigensystem(phonons, q_points)[:, 1:, :]

    if is_antisymmetrizing:
        error = np.linalg.norm(dynmat_derivatives + dynmat_derivatives.swapaxes(0, 1)) / 2
        dynmat_derivatives = (dynmat_derivatives - dynmat_derivatives.swapaxes(0, 1)) / 2
        print('Symmetrization error: ' + str(error))
    if phonons._is_amorphous:
        sij = contract('kim,kija,kjn->kmna', eigenvects, dynmat_derivatives, eigenvects)
    else:
        sij = contract('kim,kija,kjn->kmna', eigenvects.conj(), dynmat_derivatives, eigenvects)
    return sij


def calculate_velocities_af(phonons, q_points=None, is_antisymmetrizing=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = apply_boundary_with_cell(q_points)
    if is_main_mesh:
        sij = phonons._sij
        frequencies = phonons.frequency
    else:
        sij = calculate_sij(phonons, q_points, is_antisymmetrizing)
        frequencies = calculate_frequencies(phonons, q_points)
    velocities_AF = contract('kmna,kmn->kmna', sij,
                             1 / (2 * np.pi * np.sqrt(frequencies[:, :, np.newaxis]) * np.sqrt(
                                 frequencies[:, np.newaxis, :]))) / 2
    return velocities_AF


def calculate_velocities(phonons, q_points=None, is_antisymmetrizing=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = apply_boundary_with_cell(q_points)
    if is_main_mesh:
        velocities_AF = phonons._velocities_af
    else:
        velocities_AF = calculate_velocities_af(phonons, q_points, is_antisymmetrizing=is_antisymmetrizing)
    velocities = 1j * contract('kmma->kma', velocities_AF)
    return velocities.real


def calculate_eigensystem(phonons, q_points=None, only_eigenvals=False):
    is_main_mesh = True if q_points is None else False
    if not is_main_mesh:
        if q_points.shape == phonons._main_q_mesh.shape:
            if (q_points == phonons._main_q_mesh).all():
                is_main_mesh = True
    if is_main_mesh:
        q_points = phonons._main_q_mesh
    else:
        q_points = apply_boundary_with_cell(q_points)
    atoms = phonons.atoms
    n_unit_cell = atoms.positions.shape[0]
    n_k_points = q_points.shape[0]
    # Here we store the eigenvalues in the last column
    if phonons._is_amorphous:
        dtype = np.float
    else:
        dtype = np.complex
    if only_eigenvals:
        esystem = np.zeros((n_k_points, n_unit_cell * 3), dtype=dtype)
    else:
        esystem = np.zeros((n_k_points, n_unit_cell * 3 + 1, n_unit_cell * 3), dtype=dtype)
    for index_k in range(n_k_points):
        qvec = q_points[index_k]
        is_at_gamma = (qvec == (0, 0, 0)).all()
        dynmat = phonons.finite_difference.dynmat
        if is_at_gamma:
            dyn_s = contract('ialjb->iajb', dynmat)
        else:
            # TODO: the following espression could be done on the whole main_q_mesh
            dyn_s = contract('ialjb,l->iajb', dynmat, phonons.chi(qvec))
        dyn_s = dyn_s.reshape((phonons.n_modes, phonons.n_modes), order='C')
        if only_eigenvals:
            evals = np.linalg.eigvalsh(dyn_s)
            esystem[index_k] = evals
        else:
            if is_at_gamma:
                evals, evects = dsyev(dyn_s)[:2]
            else:
                evals, evects = np.linalg.eigh(dyn_s)
            esystem[index_k] = np.vstack((evals, evects))
    return esystem


def calculate_physical_modes(phonons):
    physical_modes = np.ones_like(phonons.frequency.reshape(phonons.n_phonons), dtype=bool)
    if phonons.min_frequency is not None:
        physical_modes = physical_modes & (phonons.frequency.reshape(phonons.n_phonons) > phonons.min_frequency)
    if phonons.max_frequency is not None:
        physical_modes = physical_modes & (phonons.frequency.reshape(phonons.n_phonons) < phonons.max_frequency)
    if phonons.is_nw:
        physical_modes[:4] = False
    else:
        physical_modes[:3] = False
    return physical_modes