
import array
import random
import json

import numpy

from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
###
from sympy.combinatorics.graycode import GrayCode
from sympy.combinatorics.graycode import gray_to_bin
from sympy.combinatorics.graycode import bin_to_gray

##
import random
import pandas as pd

# # Split 10:20:
# x1 = [0]*10
# x2 = [0]*10
# x3 = [0]*10



population = 5 # should be 25
bit_length = 10
# Decision variables are a list in gray code
list = [1,0,1,1,0,1,1,0]
numOfBits = 10 # Number of bits in the chromosomes
maxnum = 2**numOfBits # absolute max size of number coded by binary list 1,0,0,1,1,....


def f1(x1, x2, x3):
    return ((x1/2.0)**2 + (x2/4.0)**2 + (x3)**2) / 3.0


def f2(x1, x2, x3):
    return ((x1/2.0-1.0)**2 + (x2/4.0-1.0)**2 + (x3-1.0)**2) / 3.0

def chrom2real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange = -4+8*numasint/maxnum # CHANGED TO OUR VALUES
    # print(numinrange)
    return numinrange

def generateDataFrame():
    df = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f1', 'f2'])
    for i in range(population):
        x1 = ""
        x2 = ""
        x3 = ""

        for x in range(bit_length):
            x1 += str(random.randint(0, 1))
            x2 += str(random.randint(0, 1))
            x3 += str(random.randint(0, 1))



        _f1 = f1(chrom2real(x1), chrom2real(x2), chrom2real(x3))
        _f2 = f2(chrom2real(x1), chrom2real(x2), chrom2real(x3))

        df.loc[i] = [x1, x2, x3, _f1, _f2]

    return df


def ENDS(df):
    df = df.sort_values(by="f1")
    df.reset_index(inplace=True, drop=True)

    fronts = [[df.loc[0]]]
    num_fronts = 1


    print(df)
    for col, row in df.loc[1:].iterrows():
        #print(row)
        for i in range(len(fronts)):
            for value in fronts[i]:
                print(f'FRONT {value} \n\n')

                print(f'ROW {col,row} \n\n')

                # Modify this, does it need f1?
                if row['f1'] > value['f1'] and row['f2'] > value['f2']:
                    print('value dominates row')
                    # Check with next front. If doesn't exist create new front

                    if i + 1 == num_fronts:
                        num_fronts += 1
                        fronts.append([row])
                        print('Adding to new front...')
                        break
                    # Make it so it checks other values as well before creating new front
                else:
                    print(f'value doesn\'t dominate row')
                    fronts[i].append(row)
                    break

    # Print fronts
    for i in range(len(fronts)):
        print(f'\nFront: {i + 1}')
        for val in fronts[i]:
            print(f'{val.name} f1: {val["f1"]} f2: {val["f2"]}')

    return df

def main():
    df = generateDataFrame()
    # print(ENDS(df))
    ENDS(df)




if __name__ == '__main__':
    main()

'''
for i in range(16):
    bin='{0:b}'.format(i)
    gray=bin_to_gray(bin)
    print(i, "in binary: ", bin, "  in Gray: ", gray)
'''
# Returned number is not correct

# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: real value
def chrom2real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange=-5+10*numasint/maxnum
    return numinrange

#print(chrom2real([1,0,1,1,0,1,1,1,0,1]))
###


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

