
import array
import random
import json

import numpy as np

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



population = 25 # should be 25
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
        #df.to_pickle('testing.pickle')
        #df = pd.read_pickle('testing.pickle')
    return df


def ENDS(df):
    df = df.sort_values(by="f1")
    df.reset_index(inplace=True, drop=True)
    print("Q1.1", df)

    fronts = [[df.loc[0]]]
    df.at[0, 'front number'] = 1

    num_fronts = 1
    added = False

    for col, row in df.loc[1:].iterrows():
        added = False

        #print(row)
        for i in range(len(fronts)):
            for value in fronts[i]:
                # print(f'FRONT {value.name} f1: {value["f1"]} f2: {value["f2"]}')
                # print(f'ROW {row.name} f1: {row["f1"]} f2: {row["f2"]}')

                # Modify this, does it need f1?
                if row['f1'] > value['f1'] and row['f2'] > value['f2']:
                    # print('value dominates row')
                    # Check with next front. If doesn't exist create new front

                    if i + 1 == num_fronts:
                        num_fronts += 1
                        fronts.append([row])
                        df.at[row.name, 'front number'] = num_fronts
                        # print('Adding to new front...\n')
                        break
                    # Make it so it checks other values as well before creating new front
                elif added == False:
                    # print(f'value doesn\'t dominate row\n')
                    fronts[i].append(row)
                    df.at[row.name, 'front number'] = i+1

                    added = True
                    break

    # Print fronts
    # for i in range(len(fronts)):
    #     # print(f'\nFront: {i + 1}')
    #     for val in fronts[i]:
    #         print(f'{val.name} f1: {val["f1"]} f2: {val["f2"]}')
    #         # df.at[val.name, 'front number'] = i+1
    #         # print(df.at[val.name, 'front'])
    #         #df[df.index == val.name]
    #         pass

    df = df.sort_values(by="front number")
    #df.reset_index(inplace=True, drop=True)
    print("\nQ1.2\n", df[['f1', 'f2', 'front number']])
    print(f"\nworst f1: {max(df['f1'])}\nworst f2: {max(df['f2'])}")
    return df


def crowding_distance(df):
    # for every front
    crowding_distances = []
    for i in range(1, int(max(df['front number'])) + 1):
        # front = df.sort_values(by="f1")
        front = df.loc[df['front number'] == float(i)]
        print("\n\n\n", front)

        # print(front)
        crowding_distances.append([])
        crowding_distances[-1].append(np.inf) # first distance is always infinite
        for index, element in front[1:len(front)-1].iterrows():
            # You only need to work out for f2 if you sort by f1
            closest = np.inf

            for i, element2 in front.iterrows():
                if abs(element2['f2'] - element['f2']) + abs(element2['f1'] - element['f1']) < closest and abs(element2['f2'] - element['f2']) + abs(element2['f1'] - element['f1']) != 0:
                    closest = abs(element2['f2'] - element['f2']) + abs(element2['f1'] - element['f1'])
                    print("New Value:", closest)



                    #closest = max(-((element2['f2'] - element['f2'])) ,element2['f2'] - element['f2']) + max(-((element2['f1'] - element['f1'])), (element2['f1'] - element['f1']))

                #print('VALUE', abs(float(element2['f2'] - element['f2'])) )
            print('things')
            print(index)
            print("Crowding Distance:", closest)
            df.at[index, 'crowding distance'] = closest

        # First and last inf crowding distances

        # Surely this is affected by scale?

        # crowding_distance[-1].append(np.inf) # last distance is always infinite



    # find adjacent elements in the front (two closest to it)

    #
    df.fillna(value=np.inf, inplace=True)

    print(df)

    return df


def main():
    df = generateDataFrame()
    df = ENDS(df)
    df = crowding_distance(df)





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

