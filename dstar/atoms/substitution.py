import pandas as pd
from ast import literal_eval
import numpy as np
from itertools import combinations, product

def el_from_motif(motif_dict):
    """
    Get elements from dictionary type of active motif
    Args:
        motif_dict (dictionary)
    Return:
        elements (list)
    """
    elements = [i for i in motif_dict.keys() if i != 'Empty']
    return elements

def get_all_element(motif_df):
    """
    Get all unique elements in dataset
    Args:
        motif_df (pd.DataFrame) : output of surf_to_df function
    Retrun:
        elements (list)
    """
    el = []
    for i,v in motif_df.iterrows():
        for motif in ['FNN','Same','Sub']:
            el += el_from_motif(literal_eval(v[motif]))
    return [i for i in set(el)]        
    
def get_binary(motif_df):
    """
    Remain unary and binary materials and drop the others.
    This function only consider elements in  active motif so cannot distinguish 
    over ternary material but with 2 or 1 elements in active motif.
    
    Args:
        motif_df (pd.DataFrame) : output of surf_to_df function
    Return:
        motif_df (pd.DataFrame) : motif_df only with unary and binary materials
    """
    nelem = []
    
    # Count the number of unique elements in active motif
    for i,v in motif_df.iterrows():
        el = []
        for motif in ['FNN','Same','Sub']:
            el += el_from_motif(literal_eval(v[motif]))
        nel = len(set(el))
        nelem.append(nel)
    
    motif_df['nelem'] = nelem
    
    # Remain material with element count <= 2
    motif_df = motif_df[motif_df['nelem'] <= 2].copy()
    motif_df.drop(['nelem'],axis=1,inplace=True)
    
    return motif_df

def generalizer(motif_df):
    """  
    Change elements in acitve motif dictionary to 'A' or 'B' for substitution.
    Rich elements will be 'A' and other will be 'B'
    ex) {Cu : 1}, {Cu : 2, Al :2}, {Cu : 1, Al :1} -> {A : 1}, {A : 2, B : 2}, {A : 1, B : 1}
    
    Args:
        motif_df (pd.DataFrame) : output of surf_to_df function
    Return:
        general_df (pd.DataFrame) : Generalzied dataframe with 'A' and 'B'
    """
  
    motif_df = get_binary(motif_df)
    general_df = pd.DataFrame(columns = ['name','FNN','Same','Sub','target'])
    
    names, fnns, sames, subs = [], [], [], []
    for i,v in motif_df.iterrows():
        names.append(v['name'])
        
        ## Collect element from motif
        el = []
        for motif in ['FNN','Same','Sub']:
            el += el_from_motif(literal_eval(v[motif]))
        #el = [i for i in set(el)]
        
        ## Rich element is set to 'B'
        b = max(set(el),key=el.count)
        try:
          a = [j for j in el if j != b][0]
        except IndexError:
          a = b
        fnn = literal_eval(v['FNN'])
        same = literal_eval(v['Same'])
        sub = literal_eval(v['Sub'])
        
        ## Change element to 'A' or 'B'
        for dict_ in [fnn,same,sub]:
            for alph, e in zip(['A','B'],[a,b]):
                try:
                    dict_[alph] = dict_.pop(e)
                except KeyError:
                    pass
        fnns.append(fnn)
        sames.append(same)
        subs.append(sub)
    
    general_df['name'] = names
    general_df['FNN'] = fnns
    general_df['Same'] = sames
    general_df['Sub'] = subs
    general_df['target'] = np.zeros(len(general_df))
    
    return general_df

def binary_subs(atom_set,motif_dict):
    copy_A = motif_dict.copy()
    copy_B = motif_dict.copy()
    A = atom_set[0]
    B = atom_set[1]
    
    for el,alph in zip([A,B],['A','B']):
        try:
            copy_A[el] = copy_A.pop(alph)
        except:
            pass
    
    for el,alph in zip([B,A],['A','B']):
        try:
            copy_B[el] = copy_B.pop(alph)
        except:
            pass
        
    return copy_A,copy_B
    

def substitution(general_df, atom_set):
    """  
    Substitute generalized dataframe from generalizer function with given elements set
    Consider two possibilities, A = atom_set[0] B = atom_set[1] and A = atom_set[1] B = atom_set[2]
    Thus length of output dataframe will be doubled. 
    Args:
        general_df (pd.DataFrame) : output of generalizer function
        atom_set (array) : array containing two element
    Return:
        subs_df (pd.DataFrame) : substituted dataframe with given element set
    """
    subs_df = pd.DataFrame(columns = ['name','FNN','Same','Sub','target'])
    names, fnns, sames, subs, targets = [], [], [], [], []
    
    for i,v in general_df.iterrows():
        for j in range(2):
            names.append(v['name'])
            targets.append(v['target'])
        for col, lst in zip(['FNN','Same','Sub'], [fnns,sames,subs]):
            subs_A, subs_B = binary_subs(atom_set,v[col])
            lst.append(subs_A)
            lst.append(subs_B)
    
    subs_df['name'] = names
    subs_df['FNN'] = fnns
    subs_df['Same'] = sames
    subs_df['Sub'] = subs
    subs_df['target'] = targets

    return subs_df
        
        
