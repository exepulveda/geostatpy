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

# Print iterations progress
# From https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
