from ase import Atoms
from ase.constraints import FixAtoms
import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import VoronoiNN

def constrain_slab(atoms, z_cutoff=3.):
    '''
    This function fixes sub-surface atoms of a slab. Also works on systems that
    have slabs + adsorbate(s), as long as the slab atoms are tagged with `0`
    and the adsorbate atoms are tagged with positive integers.
    Inputs:
        atoms       ASE-atoms class of the slab system. The tags of these atoms
                    must be set such that any slab atom is tagged with `0`, and
                    any adsorbate atom is tagged with a positive integer.
        z_cutoff    The threshold to see if slab atoms are in the same plane as
                    the highest atom in the slab
    Returns:
        atoms   A deep copy of the `atoms` argument, but where the appropriate
                atoms are constrained
    '''
    # Work on a copy so that we don't modify the original
    atoms = atoms.copy()

    # We'll be making a `mask` list to feed to the `FixAtoms` class. This list
    # should contain a `True` if we want an atom to be constrained, and `False`
    # otherwise
    mask = []

    # If we assume that the third component of the unit cell lattice is
    # orthogonal to the slab surface, then atoms with higher values in the
    # third coordinate of their scaled positions are higher in the slab. We make
    # this assumption here, which means that we will be working with scaled
    # positions instead of Cartesian ones.
    scaled_positions = atoms.get_scaled_positions()
    unit_cell_height = np.linalg.norm(atoms.cell[2])

    # If the slab is pointing upwards, then fix atoms that are below the
    # threshold
    if atoms.cell[2, 2] > 0:
        max_height = max(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = max_height - z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] < threshold:
                mask.append(True)
            else:
                mask.append(False)

    # If the slab is pointing downwards, then fix atoms that are above the
    # threshold
    elif atoms.cell[2, 2] < 0:
        min_height = min(position[2] for position, atom in zip(scaled_positions, atoms)
                         if atom.tag == 0)
        threshold = min_height + z_cutoff / unit_cell_height
        for position, atom in zip(scaled_positions, atoms):
            if atom.tag == 0 and position[2] > threshold:
                mask.append(True)
            else:
                mask.append(False)

    else:
        raise RuntimeError('Tried to constrain a slab that points in neither '
                           'the positive nor negative z directions, so we do '
                           'not know which side to fix')

    atoms.constraints += [FixAtoms(mask=mask)]
    return atoms

def remove_adsorbate(adslab):
    '''
    This function removes adsorbates from an adslab and gives you the locations
    of the binding atoms. Note that we assume that the first atom in each adsorbate
    is the binding atom.
    Arg:
        adslab  The `ase.Atoms` object of the adslab. The adsorbate atom(s) must
                be tagged with non-zero integers, while the slab atoms must be
                tagged with zeroes. We assume that for each adsorbate, the first
                atom (i.e., the atom with the lowest index) is the binding atom.
    Returns:
        slab                The `ase.Atoms` object of the bare slab.
        binding_positions   A dictionary whose keys are the tags of the
                            adsorbates and whose values are the cartesian
                            coordinates of the binding site.
    '''
    # Operate on a local copy so we don't propagate changes to the original
    slab = adslab.copy()

    # Remove all the constraints and then re-constrain the slab. We do this
    # because ase does not like it when we delete atoms with constraints.
    slab.set_constraint()
    slab = constrain_slab(slab)

    # Delete atoms in reverse order to preserve correct indexing
    binding_positions = []
    checksum = 0
    for i, atom in reversed(list(enumerate(slab))):
        if atom.tag != 0:
            checksum = 1
            binding_positions.append(atom.position)
            del slab[i]
    
    assert checksum == 1, 'There is no tagged adsorbates'
        
    return slab, binding_positions
