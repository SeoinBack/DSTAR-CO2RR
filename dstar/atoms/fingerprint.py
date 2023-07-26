import sys
sys.path.append("../.")

import os
import tqdm
import numpy as np
import pandas as pd
import pymatgen as mg


from ase import Atoms
from ase.io import read,write
from ast import literal_eval

from dstar.atoms.active_motif import get_active_motif


def block_to_num(block):
    """
    Convert blokc to number
    
    Args:
        block (str) : 's', 'p', 'd' or 'f'
        
    Return:
        int : 1, 2, 3, 4
    """

    if block == 's':
        return 1
    elif block == 'p':
        return 2
    elif block == 'd':
        return 3
    elif block == 'f':
        return 4

    
def motif_to_prop(el_dict):
    """
    Convert acitve motif composition dictionary obtained from get_active_motif function
    to average atomic properties and the number of atoms of each site. 
    
    Args:
        el_dict (dictionary) : dictionary component in output of get_acitve_motif function
    Return:
        array : average value of eleven atomic properties and the number of atom
    """
    N = sum(el_dict.values())
    atomic_number_list = []
    average_ionic_radius_list=[]
    common_oxidation_states_list=[]
    Pauling_electronegativity_list = []
    group_list = []
    row_list = []
    thermal_conductivity_list = []
    melting_point_list = []
    boiling_point_list = []
    block_list = []
    IE_list = []
    
    # Bring atomic properties of element from Pymatgen and append to the list
    for el in el_dict:
        if el != 'Empty':
        
            # Convert element as string to Pymatgen.core.Element type to get property 
            e = mg.core.Element(el)
            atomic_number = e.Z
            average_ionic_radius = e.average_ionic_radius.real
            
            # Lowest oxidiation state of the element is used as common oxidation state
            common_oxidation_states = e.common_oxidation_states[0]
            Pauling_electronegativity = e.X
            row = e.row
            group = e.group
            thermal_conductivity = e.thermal_conductivity.real
            boiling_point = e.boiling_point.real
            melting_point = e.melting_point.real
            block = block_to_num(e.block)
            IE = e.ionization_energy
        
        # All properties is 0 when site is empty
        elif el == 'Empty':
            natom = 0
            ele = 0
            atomic_number = 0
            average_ionic_radius = 0
            common_oxidation_states = 0
            Pauling_electronegativity = 0
            row = 0
            group = 0
            thermal_conductivity = 0
            boiling_point = 0
            melting_point = 0
            block = 0
            IE = 0
        
        atomic_number_list += [atomic_number]*el_dict[el]
        average_ionic_radius_list += [average_ionic_radius]*el_dict[el]
        common_oxidation_states_list += [common_oxidation_states]*el_dict[el]
        Pauling_electronegativity_list += [Pauling_electronegativity]*el_dict[el]
        row_list += [row]*el_dict[el]
        group_list += [group]*el_dict[el]
        thermal_conductivity_list += [thermal_conductivity]*el_dict[el]
        boiling_point_list += [boiling_point]*el_dict[el]
        melting_point_list += [melting_point]*el_dict[el]
        block_list += [block]*el_dict[el]
        IE_list += [IE]*el_dict[el]
    
    # Average list by the number of atom in site
    atomic_number_mean = np.sum(atomic_number_list)/N
    average_ionic_radius_mean = np.sum(average_ionic_radius_list)/N
    common_oxidation_states_mean = np.sum(common_oxidation_states_list)/N
    Pauling_electronegativity_mean = np.sum(Pauling_electronegativity_list)/N
    row_mean = np.sum(row_list)/N
    group_mean = np.sum(group_list)/N
    thermal_conductivity_mean = np.sum(thermal_conductivity_list)/N
    boiling_point_mean = np.sum(boiling_point_list)/N
    melting_point_mean = np.sum(melting_point_list)/N
    block_mean = np.sum(block_list)/N
    IE_mean = np.sum(IE_list)/N 
    
    return [atomic_number_mean, average_ionic_radius_mean, common_oxidation_states_mean, Pauling_electronegativity_mean, row_mean, group_mean, thermal_conductivity_mean, boiling_point_mean, melting_point_mean, block_mean, IE_mean, N]

def surf_to_fp(atoms):
    """
    Convert surface with tagged adsorbates to input for DSTAR
    
    Args:
        atoms (ase.atom.Atoms) : Surface with tagged adsorbates 
    Return:
        fp (array) : 36 properties components (12 per site)
        active_motif (array) : output of get_acitive_motif
    """
    
    active_motif, _ = get_active_motif(atoms)
    
    if list(active_motif[0].keys())[0] == 'Empty':
        raise RuntimeError('Active site is not captured. Check the surface')
    
    fp = []
    
    for dict_ in active_motif:
        fp += motif_to_prop(dict_)
    
    return fp, active_motif

def motifs_to_df(motif_df, skip=False):
    """
    Convert dataframe consisted with active motif information generated from surfs_to_df function
    to properties dataframe
    Args:
        motif_df (pd.DataFrame) : name and element dictionary of each site
        skip (bool) : skip printing...
    Return:
        prop_df (pd.DataFrame) : dataframe with properteis of each site
    """
    
    prop_df= pd.DataFrame(columns = ['name'] + [i+str(j) for i in ['FNN','Same','Sub'] for j in range(12)])
    if not skip:
     # print('Initiate Conversion Motifs to Fingerprint...')
      for i,v in motif_df.iterrows():
          fp = [] 
          name = str(v['name'])
          
          # Read dictionary in string type using literal_eval function 
          motif = [literal_eval(str(v['FNN'])), literal_eval(str(v['Same'])), literal_eval(str(v['Sub']))]
          
          for dict_ in motif:
              fp += motif_to_prop(dict_)
                
          prop_df.loc[len(prop_df)] = [name] + fp
      #print('Successfully Generate Fingerprint!')
      #print('')
    
    else:
      for i,v in motif_df.iterrows():
          fp = [] 
          name = str(v['name'])
          motif = [literal_eval(str(v['FNN'])), literal_eval(str(v['Same'])), literal_eval(str(v['Sub']))]
          
          for dict_ in motif:
              fp += motif_to_prop(dict_)
                
          prop_df.loc[len(prop_df)] = [name] + fp
    return prop_df

def surfs_to_df(path):
    """
    Convert all surface files in path to dataframe with properties and active motif informaton.
    Args:
        path (str) : path including surfaces
    Return:
        prop_df (pd.DataFrame) : dataframe with properties
        motif_df (pd.DataFrame) : dataframe with active motifs
    """
    atoms_lst = os.listdir(path)
    
    prop_df = pd.DataFrame(columns = ['name']+[i+str(j) for i in ['FNN','Same','Sub'] for j in range(12)])
    motif_df = pd.DataFrame(columns = ['name']+['FNN','Same','Sub'])
    
    #print('Initiate Conversion Atoms to Fingerprint...')
    for file in atoms_lst:
        if not file.endswith('.csv'):
        
            # Read atom and convert to properties and active motif
            name = file.split('.')[0]
            atoms = read(path+'/'+file)
            
            fp, motif = surf_to_fp(atoms)
            
            motif_df.loc[len(prop_df)] = [name] + motif
            prop_df.loc[len(prop_df)] = [name] + fp
    #print('Successfully Generate Fingerprint!')    
    #print('')
    return prop_df, motif_df
