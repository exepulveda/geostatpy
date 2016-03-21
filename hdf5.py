import pandas as pd
import h5py
import numpy as np
import operator

import gaussian

def export_to_gslib(database,filename,variables,indices=None):
    fd = open(filename,"w")
    
    x = get_variable(database,"composites/midx")
    y = get_variable(database,"composites/midy")
    z = get_variable(database,"composites/midz")

    data = {}
    for variable,path in variables:
        data[variable] = get_variable(database,path)
    
    if indices is None:
        n = len(x)
        indices = xrange(n)
    
    
    fd.write("DATABASE OF {0}\n".format(variable))
    fd.write("{0}\n".format(3+len(variables)))
    fd.write("x\n")
    fd.write("y\n")
    fd.write("z\n")
    for variable,path in variables:
        fd.write("{0}\n".format(variable))
    
    for i in indices:
        fd.write("{0} {1} {2} ".format(x[i],y[i],z[i]))
        for variable,path in variables:
            value = data[variable][i]
            fd.write("{0} ".format(value if np.isfinite(value) else -999))
        fd.write("\n")

def export_to_geostatwin(database,filename,variables,assay_length,indices=None):
    fd = open(filename,"w")
    
    dh = get_variable(database,"composites/dhid")
    x = get_variable(database,"composites/midx")
    y = get_variable(database,"composites/midy")
    z = get_variable(database,"composites/midz")

    data = {}
    for variable,path in variables:
        data[variable] = get_variable(database,path)

    
    if indices is None:
        n = len(x)
        indices = xrange(n)    
    
    fd.write("DATABASE OF {0}    {1} {2} ".format(variable,assay_length,len(variables)))
    for variable,path in variables:
        fd.write("{0}".format(variable[:2]))
    fd.write("\n")
    
    for i in indices:
        fd.write("{0} {1} {2} {3} ".format(dh[i],x[i],y[i],z[i]))
        for variable,path in variables:
            value = data[variable][i]
            fd.write("{0} ".format(value if np.isfinite(value) else -999))
        fd.write("\n")


def export_to_csv(database,filename,variables,indices=None):
    fd = open(filename,"w")
    
    dh = get_variable(database,"composites/dhid")
    x = get_variable(database,"composites/midx")
    y = get_variable(database,"composites/midy")
    z = get_variable(database,"composites/midz")
    
    data = {}
    for variable,path in variables:
        data[variable] = get_variable(database,path)

    if indices is None:
        n = len(x)
        indices = xrange(n)    

    variables_names = [v[0] for v in variables]
    fd.write(",".join(["dhid","x","y","z"]))
    fd.write(",")
    fd.write(",".join(variables_names))
    fd.write("\n")
    
    for i in indices:
        fd.write("{0},{1},{2},{3},".format(dh[i],x[i],y[i],z[i]))

        values = [str(data[v][i]) for v in variables_names]

        fd.write(",".join(values))
        fd.write("\n")

def default_na(database):
    return database.attrs.get("na",-999)
    
def open_database(filename,mode='a'):
    h5 = h5py.File(filename,mode)
    return h5

def get_variable(database,variable):
    return database[variable][:]

def delete_variable(database,variable):
    if variable in database:
        del database[variable]

def set_variable(database,variable,data):
    if variable not in database:
        dset = database.create_dataset(variable,data.shape,dtype=data.dtype)
    else:
        dset = database[variable]
        
    dset[:] = data

def set_attribute(database,variable,attributes):
    ds = database[variable]
    for k,v in attributes.items():
        ds.attrs[str(k)] = v        

def coding_string_variable(database,variable):
    data = database[variable][:]
    
    unique_codes = list(set(data))
    unique_codes.sort()
    
    if "" not in unique_codes:
        unique_codes += [""]
        
    #build dict
    code_dict = {}
    
    for i,code in enumerate(unique_codes):
        code_dict[code] = i
        
    print code_dict
    new_variable_name = variable + "_encoded"
    
    delete_variable(database,new_variable_name)

    
    new_variable = np.empty(len(data),dtype=np.int32)
    
    for i in xrange(len(data)):
        new_variable[i] = code_dict[data[i]]
    
    set_variable(database,new_variable_name,new_variable)
    
    ivd = {v: k for k, v in code_dict.items()}
    
    set_attribute(database,new_variable_name,ivd)
            
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

'''
def add_variable_composites(db,variable_name,data,variable_type,default_value = np.nan,selector=lambda x: x.data,reductor = compositing.main_category, isCategorical = False):
    import compositing
    #load drillholes from 
    composite_holeId = get_variable(db,"composites/dhid")
    composite_from = get_variable(db,"composites/from")
    composite_to = get_variable(db,"composites/to")

    n = len(composite_holeId)

    dh_composites = {}
    for i in xrange(n):
        if composite_holeId[i] not in dh_composites:
            dh_composites[composite_holeId[i]] = []
        
        segment = compositing.FromTo_Record(composite_from[i],composite_to[i],None)
        dh_composites[composite_holeId[i]] += [(segment,i)]

    #adding a feature to databse based on drill hole information

    delete_variable(db,variable_name)

    npdata = np.empty(n,dtype=variable_type)

    for holeId,value in dh_composites.items():
        if holeId in data:
            segments =  data[holeId]

            #value.sort(key=operator.attrgetter('depth_from'))
            composites, indices = zip(*value)
            
            new_feature = compositing.append_feature(composites,segments,selector,reductor,holeId == "CE046")
            #print holeId,new_feature
            #add new variable
            k = np.where(new_feature is None)[0]
            if len(k) > 0:
                new_feature[k] = default_value
                
            #print holeId,len(new_feature),len(indices), [x for x in indices if x >= n]
            
            npdata[list(indices)] = new_feature

        else:
            print "dh not found",holeId


    #save data
    set_variable(db,variable_name,npdata)
    if isCategorical:
        coding_string_variable(db,variable_name)
    
'''
