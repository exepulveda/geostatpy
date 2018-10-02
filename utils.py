import numpy as np
import csv

def load_gslib_data(filename):
    reader = csv.reader(open(filename),delimiter=' ',skipinitialspace=True)

    header = next(reader)
    row = next(reader)
    n_cols = int(row[0])
    var_names = []
    for i in range(n_cols):
        row = next(reader)
        name = ' '.join(row)
        var_names += [name.strip()]
    #read values
    data = []
    for row in reader:
        data += [[float(x) for x in row[:n_cols]]]

    return var_names,np.array(data)

def save_gslib_data(filename,data,title=None,colnames=None):
    writer = csv.writer(open(filename,"w"),delimiter=' ')

    if len(data.shape)>1:
        n,m = data.shape
    else:
        n = len(data)
        m = 1

    if title is None:
        title = "Data"
    if colnames is None:
        colnames = ["column_%d"%(i+1) for i in range(m)]

    writer.writerow([title])
    writer.writerow([m])
    for i in range(m):
        writer.writerow([colnames[i]])
    #write values
    for row in range(n):
        if m > 1:
            writer.writerow(list(data[row,:]))
        else:
            writer.writerow([data[row]])
