# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:59:52 2017

@author: jules
"""

import pandas as pd

def createMeta(set):
    data = []
    for f in set.columns:
        # Defining the role
        if f == 'target':
            role = 'target'
        elif f == 'id':
            role = 'id'
        else:
            role = 'input'
             
        # Defining the level

        if 'bin' in f or f == 'target':
            level = 'binary'
        elif 'cat' in f or f == 'id':
            level = 'nominal'
        elif set[f].dtype == 'float64':
            level = 'interval'
        elif set[f].dtype == 'int64':
            level = 'ordinal'
            
        # Initialize keep to True for all variables except for id
        keep = True
        if f == 'id':
            keep = False
        
        # Defining the data type 
        dtype = set[f].dtype
        
        # Creating a Dict that contains all the metadata for the variable
        f_dict = {
            'varname': f,
            'role': role,
            'level': level,
            'keep': keep,
            'dtype': dtype
        }
        data.append(f_dict)
        
    meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
    meta.set_index('varname', inplace=True)
    
    return meta