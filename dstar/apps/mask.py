import pandas as pd
import pickle
import numpy as np

from ast import literal_eval

def masking_dataframe(df, fnn = None, composition = None, host = None, comp_operator = '>', cn = None, cn_operator = '>'):
    if fnn != None:
        fnn_masking = [True if fnn == list(literal_eval(i).keys())[0] else False for i in df['FNN'].to_list()]
        df = df[fnn_masking]
    
    if composition != None:
        assert host != None, "Host must be defined"
        compos_lst = get_composition(df,host)
        if comp_operator == '>':
            compos_masking = [True if i > composition else False for i in compos_lst]
        elif comp_operator == '==':
            compos_masking = [True if i == composition else False for i in compos_lst]
        elif comp_operator == '>=':
            compos_masking = [True if i >= composition else False for i in compos_lst]
        elif comp_operator == '<':
            compos_masking = [True if i < composition else False for i in compos_lst]
        elif comp_operator == '<=':
            compos_masking = [True if i <= composition else False for i in compos_lst]
        df = df[compos_masking]
    
    if  cn != None:
        cn_dict = load_cn()
        cn_lst = [cn_dict[id_] for id_ in df['name'].to_list()]
        
        if cn_operator == '>':
            cn_masking = [True if i > cn else False for i in cn_lst]
        elif cn_operator == '>=':
            cn_masking = [True if i >= cn else False for i in cn_lst]
        elif cn_operator == '==':
            cn_masking = [True if i == cn else False for i in cn_lst]
        elif cn_operator == '<=':
            cn_masking = [True if i <= cn else False for i in cn_lst]
        elif cn_operator == '<':
            cn_masking = [True if i < cn else False for i in cn_lst]
        df = df[cn_masking]
    return df
        
def get_composition(df, host):
    compos_lst = []
    
    key_lst =[list(literal_eval(i).keys()) for i in df['FNN'].to_list()]
    key_lst = list(set([i for sublst in key_lst for i in sublst]))
    key_lst.remove(host)
    guest = key_lst[0]
    
    
    for i, v in df.iterrows():
        fnn = literal_eval(v['FNN'])
        same = literal_eval(v['Same'])
        sub = literal_eval(v['Sub'])
    
        a = 0 
        b = 0
    
        for site in [fnn,same,sub]:
            if host in site.keys():
                a += site[host]
                
            if guest in site.keys():
                b += site[guest]
                    
        comp = a/(a+b)
        compos_lst.append(round(comp*20,0)/2)
    return compos_lst
        
def load_cn(path = '../../data/rebuttal/scaler_min/cn.pkl'):
    with open(path,'rb') as fr:
        cn_dict = pickle.load(fr)
    return cn_dict