
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

'''
for i in range(16):
    bin='{0:b}'.format(i)
    gray=bin_to_gray(bin)
    print(i, "in binary: ", bin, "  in Gray: ", gray)
'''
# Returned number is not correct

# Decision variables are a list in gray code
list = [1,0,1,1,0,1,1,0]
numOfBits = 10 # Number of bits in the chromosomes
maxnum = 2**numOfBits # absolute max size of number coded by binary list 1,0,0,1,1,....

# x1 = [0,0,0,0,0,0,0,0,0,0] # x1 is any 10bit number

# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: real value
def chrom2real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange=-5+10*numasint/maxnum
    return numinrange

print(chrom2real([1,0,1,1,0,1,1,1,0,1]))
###


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

