# ID: 2018115809
# NAME: Dohun Kim
# File name: read_iris_v2.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): sys numpy=1.19.2

##############################################################
# NOTE: import sys and numpy only. 
# No other packages are allowed to be imported
##############################################################
import sys
import numpy as np

if len(sys.argv) < 2:
    print('usage: ' + sys.argv[0] + ' text_file_name')
else:
    ##############################################################
    # WRITE YOUR OWN CODE LINES
    # - open the input file, without pandas or csv packages
    # - read header line
    # - read data and class labels
    # - compute mean and standard deviation
    # - disply them 
    ##############################################################
    
    # define delimeter according to the file format
    if sys.argv[1][-3:].lower() == 'csv': delimeter = ','
    else: delimeter = '\t'
    
    # create empty list (to make ndarray)
    data_list = []

    # open and read file
    f = open(sys.argv[1], 'r')
    while True:
        # read one line
        line = f.readline()
        if not line: break
        
        # split one line by delimeter and drop last element
        splitted = line.split(delimeter)
        splitted.pop()

        # add splitted line to list
        data_list.append(splitted)
    f.close()

    # read header line
    col_names = data_list.pop(0)

    # get ndarray from list
    data = np.array(data_list, dtype=float)

    # get mean and stddev values from ndarray
    data_means = data.mean(axis=0)
    data_stds  = data.std(axis=0)

    # print mean and stddev values following the given display format
    print('-----------------------------------------------------------------')

    print('     ', end='')
    for col_name in col_names:
        print(f'{col_name:>15s}', end='')
    print()
    
    print('mean ', end='')
    for data_mean in data_means:
        print(f'{data_mean:>15.2f}', end='')
    print()

    print('std  ', end='')
    for data_std in data_stds:
        print(f'{data_std:>15.2f}', end='')
    print()

    print('-----------------------------------------------------------------')
