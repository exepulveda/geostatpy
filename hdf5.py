import pandas as pd
import h5py
import numpy as np

import gaussian


def default_na(database):
    return database.attrs.get("na",-999)
    
def open_database(filename,mode='a'):
    h5 = h5py.File(filename,mode)
    return h5

def get_variable(database,variable):
    return database["composites/" + variable][:]

def set_variable(database,variable,data,prefix="composites"):
    grp = database.require_group(prefix)
    if variable not in grp:
        dset = grp.create_dataset(variable,data.shape,dtype=data.dtype)
    else:
        dset = grp[variable]
        
    dset[:] = data

def set_attribute(database,variable,attributes,prefix="composites"):
    grp = database.require_group(prefix)
    ds = grp[variable]
    for k,v in attributes.items():
        ds.attrs[str(k)] = v        
    
def compute_normal_scores(database,variable,variable_weights=None):
    data = database["composites/" + variable][:]
    if variable_weights is not None:
        weights = database["composites/" + variable_weights][:]
    else:
        weights = None
    
    #indices = np.where(data != na)[0]
    indices = np.where(np.isfinite(data))[0]
    
    transformed_data, table = gaussian.normal_scores(data,weights,indices=indices,na_value=np.nan)
    
    grp = database.require_group("nscores")
    grp = grp.require_group(variable)

    #nscores
    if "nscores" not in grp:
        dset = grp.create_dataset("nscores", data.shape,dtype="f64")
    else:
        dset = grp["nscores"]
        
    dset[:] = transformed_data
    
    #table
    if "table" not in grp:
        dset = grp.create_dataset("table", table.shape,dtype="f64")
    else:
        dset = grp["table"]
        
    dset[:,:] = table
    
    

def load_csv(database,csv_filename,na_values=[-999]):
    data = pd.read_csv(csv_filename,na_values=na_values)

    if "composites" not in h5: 
        grp = h5.create_group("composites")
    else:
        grp = h5["composites"]

    n = len(data)
    m = len(data.columns)

    for cn in data.columns:
        column = data[cn]
        print cn,column.name,column.dtype, column.dtype.name
        if cn not in grp:
            if column.dtype.name == "object":
                dt = "S" + str(max([len(x) for x in column.values]) + 2)
            else:
                dt = column.dtype
            dset = grp.create_dataset(cn, (n,),dtype=dt)
            
        else:
            dset = grp[cn]

        if column.dtype.name == "object":
            for i,v in enumerate(data[cn].values):
                #print i,v,type(v)
                dset[i] = v
        else:
            dset[:] = data[cn].values
    
    return data
