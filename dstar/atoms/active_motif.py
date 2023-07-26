import sys
sys.path.append("../.")

import pickle
import os
from ase.io import read,write
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import pymatgen as pmg
from pymatgen.analysis.local_env import VoronoiNN

from dstar.atoms.from_gaspy import constrain_slab, remove_adsorbate

def get_position(index_a,index_b,structures, z_cutoff = None):
    """
    Calculate z-axis distance between atom a and b to determine 
    relative position of atom b to a.
    If atom b is located below the atomic radius of atom a, 
    positon of atom b will be 'sub', otherwise 'same'. 
    
    Args:
        index_a (int) : Criteria atom index of structure
        index_b (int) : Relative atom index of structure
        structure (pymatgen.core.structure.Structure) : Corresponding Structure
        z_cutoff (int) : The threshold to determine position 'sub'
        
    Return:
        str: 'same' or 'sub'
    """
    
    # Distance is calculated as fractional coordinate to consider 
    # when z-axis of unit cell is not orthogonal.
    fcoord = structures.frac_coords
    
    a_fcoord = fcoord[index_a]
    b_fcoord = fcoord[index_b]
    
    # Z-cutoff can be determined arbitraly. If cutoff is not given, 
    # cutoff is automatically set to atomic raidus of atom A 
    if z_cutoff is None:
        z_cutoff = structures[index_a].specie.atomic_radius
    
    # Convert cutoff to fractional
    fdist = structures.lattice.get_fractional_coords([0,0,z_cutoff])[2]
    
    # There is case of negative lattice parameter c.
    # In this case, z distance will be reversly calculated.
    if fdist < 0:
        z_distance = b_fcoord[2]-a_fcoord[2]
    else:
        z_distance = a_fcoord[2]-b_fcoord[2]
    
    # If Atom B is over the Atom A, position will be 'same'
    if z_distance < 0:
        z_distance = 0
    
    if (z_distance <= abs(fdist)) : 
        return 'same'
    else:
        return 'sub'

def get_active_site(atom):
    """
    Change adsorbates to Uranium and find First Nearest Neighbor (FNN)
    atoms of Uranium (adsorbate) as binding site.
    Adsorbates must be tagged with non-zero positive integer 
    and slab atoms must be tagged with 0.
    This U substitution method is taken from GASpy but modifed to use 
    multi-adosrbates system. 
    
    Args:
        atom (ase.atoms.Atoms) : surface with adsorbates tagged with positive integer
    Return:
        atoms (ase.atoms.Atoms) : surface with uranium as adsorbate
    """
    
    # Remove Adosrbates tagged with positiv integer and save positions of adsorbates
    atoms,binding_positions = remove_adsorbate(atom)
    adsorb_count = len(binding_positions)/9
    # Substitue all adosrbates to Uranium
    
    for i,pos in enumerate(binding_positions):
        if (i) == 4*adsorb_count:
            atoms += Atoms('U', positions=[pos])
    u_idx = [i for i,v in enumerate(atoms.get_chemical_symbols()) if v == 'U']
    
    
    structures = AseAtomsAdaptor.get_structure(atoms)
    # This VoronoiNN parameter is used to find FNN with tight condition.
    vnn = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10)
    # In our study, we repeat the slab to 3 X 3 X 1 to avoid index
    # repetition when find FNN in narrow unit surface.
    # 4th Uranium is positioned at center of slab 
    nn = vnn.get_nn_info(structures,u_idx[0])
    
    # Get atom index of 4th Uranium
    active_idx = [i['site_index'] for i in nn]
    
    return active_idx, atoms

def divide_site(structures, index):
    """
    Get FNN of atom of input index and determine the position of 
    each FNN atom relative to input index atom.
    
    Args:
        structure (pymatgen.core.structure.Structure) : corresponding structuer
        index (int) : Atom index of structure
    Return:
        dictionary : Index of FNN atom in 'Same' position or 'Sub' position
    """
    
    # This VoronoiNN parameter is used to find FNN with loose condition.
    vnn = VoronoiNN(allow_pathological=False, tol=0.2, cutoff=13)
    nn = vnn.get_nn_info(structures,index)
    
    # Get index of FNN atoms with weight over 0.1
    nn_idx = [i['site_index'] for i in nn if i['weight'] > 0.1]
    
    same_idx, sub_idx = [], []
     
    for idx in nn_idx:
        if get_position(index, idx, structures) == 'same':
            same_idx.append(idx)
        elif get_position(index, idx, structures) == 'sub':
            sub_idx.append(idx)
        else:
            raise RuntimeError(f'Position of {idx} is not defined.')
            
    return {'same' : same_idx, 'sub' : sub_idx}

def get_active_motif(atom):
    """
    Find active motif (FNN & SNN of adsorbates) from surface structure with adsorbates.
    Adsorbates must be tagged with positive integer and slab atoms must be tagged with '0'.
    Active motif will be provided as array with dictionary form of the number of elements
    in each site, FNN, Same SNN and Sub SNN, ex) {Cu : 1},{Cu : 1, Al : 2},{Cu : 3, Al : 3}
    and array of atom index of each site.  
    
    Args: 
        atom (ase.atoms.Atoms) : Surface with adsorbates tagged with postive integer
    Return:
        array 1 : composition of each active motif site
        array 2 : atom index of each acitve motif site
    """
   
   # Input surface is repeated to 3 X 3 X 1 to avoid index repetition 
   # when unit surface narrow enough to capture same index of atoms as FNN or SNN.
    atom = atom.repeat((3,3,1))
    
    fnn_idx, atoms = get_active_site(atom)
    
    structures = AseAtomsAdaptor.get_structure(atoms)
    
    # u_idx is 4th Uranium index in 3 X 3 X 1 cell.
    u_idx = [i for i,v in enumerate(atoms.get_chemical_symbols()) if v == 'U']
    
    fnn_dict = {}
    same_dict = {}
    sub_dict = {}
    
    same_idx = []
    sub_idx = []
     
    # Count the number of elements in FNN
    fnn_elements = [str(structures.species[i]) for i in fnn_idx]
    for elem in [i for i in set(fnn_elements)]:
        fnn_dict[elem] = fnn_elements.count(elem)
    
    # Count the number of elements in Same/Sub Elements
    for idx in fnn_idx:
        layer_dict = divide_site(structures,idx)
        for i in layer_dict['same']:
            if i not in (fnn_idx+same_idx+u_idx):
                same_idx.append(i)
        
        for i in layer_dict['sub']:
            if i not in (fnn_idx+same_idx+sub_idx+u_idx):
                sub_idx.append(i)
       
    same_elements = [str(structures.species[i]) for i in same_idx
                    if str(structures.species[i]) != 'U']
    for elem in [i for i in set(same_elements)]:
        same_dict[elem] = same_elements.count(elem)
     
    sub_elements = [str(structures.species[i]) for i in sub_idx
                   if str(structures.species[i]) != 'U']
    for elem in [i for i in set(sub_elements)]:
        sub_dict[elem] = sub_elements.count(elem)
    
    # If no atoms in certain site, 'Empty' is set to key
    # of dictionary instead of elements
    for dict_ in [fnn_dict,same_dict,sub_dict]:
        if len(dict_) == 0:
            dict_['Empty'] = 1
    
    return [fnn_dict, same_dict, sub_dict], [fnn_idx, same_idx, sub_idx]
                
        
