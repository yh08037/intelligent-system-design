# ID: 2018115809
# NAME: Dohun Kim
# File name: read_iris_v1.py
# Platform: Python 3.7.4 on Ubuntu Linux 18.04
# Required Package(s): sys numpy=1.19.2 pandas=1.2.3

import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 2:
    print('usage: ' + sys.argv[0] + ' text_file_name')
else:
    # determine delimieter based on file extension - may be used by pandas
    # this is just to show how to use command line arguments. 
    # any modification is accepted depending on your implementation.
    if sys.argv[1][-3:].lower() == 'csv': delimeter = ','
    else: delimeter = '[ \t\n\r]'  # default is all white spaces 

    # read CSV/Text file with pandas
    df = pd.read_csv(sys.argv[1],sep=delimeter,engine='python')

    ##############################################################
    # WRITE YOUR OWN CODE LINES
    # - read header line
    # - read data and class labels
    # - compute mean and standard deviation
    # - disply them 
    ##############################################################

    # read header line
    col_names = df.columns.values.tolist()
    
    # get name of label column (CLASS or species)
    label_name = col_names.pop()

    '''
    # Simple solution without numpy
    # get mean, std values directly from pandas dataframe
    data_means = df.mean().values
    data_stds  = df.std().values
    '''

    # get ndarray from dataframe without label column
    data = df.drop(label_name, axis=1).values

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
