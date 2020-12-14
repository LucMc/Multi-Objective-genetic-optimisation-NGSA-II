
import array
import random
import json

import numpy as np
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
import pandas as pd
##
import array

'''

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def calcFitness(individual):
    # x1 in [1, 2,x, 16]; x2 in [1, 2,x, 8]; x3 in [1, 2, 3, 4]
    #        4 bits              3 bits             2 bits
    x1 = individual[0:4]
    x2 = individual[4:7]
    x3 = individual[7:9]
    x1 = int("".join(str(i) for i in x1), 2)
    x2 = int("".join(str(i) for i in x2), 2)
    x3 = int("".join(str(i) for i in x3), 2)
    f1 = ((x1 / 4.0) ** 2 + (x2 / 2.0) ** 2 + x3 ** 2.0) / 3.0
    f2 = ((x1 / 4.0 - 2.0) ** 2 + (x2 / 2.0 - 2.0) ** 2 + (x3 - 2.0) ** 2.0) / 3.0
    return f1, f2

toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, 9)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", calcFitness)
toolbox.register("mate", tools.cxTwoPoint)
flipProb = 1.0 / 9
toolbox.register("mutate", tools.mutFlipBit, indpb=flipProb)
toolbox.register("select", tools.selNSGA2)

def main(seed=None):
    random.seed(seed)

    NGEN = 250
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        # selTournamentDCD means Tournament selection based on dominance (D)
        # followed by crowding distance (CD). This selection requires the
        # individuals to have a crowding_dist attribute
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # make pairs of all (even,odd) in offspring
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

    return pop, logbook

if __name__ == "__main__":
    pop, stats = main()
    pop.sort(key=lambda x: x.fitness.values)

    front = numpy.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.show()

    # print some individuals
    for n in range(10):
        i = pop[random.choice(range(0, len(pop)))]
        x1 = i[0:4]
        x2 = i[4:7]
        x3 = i[7:9]
        x1 = int("".join(str(i) for i in x1), 2)
        x2 = int("".join(str(i) for i in x2), 2)
        x3 = int("".join(str(i) for i in x3), 2)
        print(x1, x2, x3)


'''

# # Split 10:20:
# x1 = [0]*10
# x2 = [0]*10
# x3 = [0]*10



population = 28 # changed from 25
bit_length = 10
# Decision variables are a list in gray code
list = [1,0,1,1,0,1,1,0]
numOfBits = 10 # Number of bits in the chromosomes
maxnum = 2**numOfBits # absolute max size of number coded by binary list 1,0,0,1,1,....
flip_prob = 0.9

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
        # Change this so its one long variable that is split up


        _f1 = f1(chrom2real(x1), chrom2real(x2), chrom2real(x3))
        _f2 = f2(chrom2real(x1), chrom2real(x2), chrom2real(x3))

        df.loc[i] = [x1, x2, x3, _f1, _f2]
        #df.to_pickle('testing.pickle')
        #df = pd.read_pickle('testing.pickle')
    return df


def ENDS(df):
    df = df.sort_values(by="f1")
    df.reset_index(inplace=True, drop=True)
    print("Q1.1\n", df)

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
    # find adjacent elements in the front (two closest to it)

    df.fillna(value=np.inf, inplace=True)

    # sort df
    df.sort_values(['front number', 'crowding distance'], ascending=[True, False], inplace=True)
    print("\nQ1.3\n", df[['f1', 'f2', 'front number', "crowding distance"]])

    return df



def tournament_selection(df):
    # Chose two random rows
    def sample_pair(df):
        # Tournament Selection
        selection = df.sample(n=2)

        # Compare front number
        print('\n\n\n\n')
        print(selection)

        if len(selection[selection['front number'] == min(selection['front number'])]) == 2:
            selection = selection[selection['crowding distance'] == min(selection['crowding distance'])]
        else:
            selection = selection[selection['front number'] == min(selection['front number'])]

        return selection

    def uniform(parent1, parent2):
        # Crossover
        child = ''

        parent1_chromosome = "" + parent1['x1'].item() + parent1['x2'].item() + parent1['x3'].item()
        parent2_chromosome = "" + parent2['x1'].item() + parent2['x2'].item() + parent2['x3'].item()
        print('chromosomes')
        # print(parent1['x1'].item())

        print(parent1_chromosome)
        print(parent2_chromosome)

        for i in range(numOfBits*3):
            bit = random.choice([parent1_chromosome, parent2_chromosome])[i]
            # mutation
            if random.uniform(0,1) > flip_prob:
                print("BIT FLIP")
                bit = int(bit) ^ 1

            child += str(bit)
            # print(child)
            # print(str(bit))

        print("RESULT:", child)
        return child

    # Select two parents through tournament
    parent1 = sample_pair(df)
    parent2 = sample_pair(df)

    child = uniform(parent1, parent2)

    # Convert to pandas row
    # Check the above process
    # child = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f1', 'f2'])
    # df.loc[i] = [child[:bit_length], child[bit_length:], , _f1, _f2]

    # child = pd.DataFrame({'x1': child[:bit_length], 'x2': child[bit_length:]})
    return child



def main():
    df = generateDataFrame()
    df = ENDS(df)
    df = crowding_distance(df)

    # Fix table generation since x3 and f values are messed up
    tournament_selection(df)


if __name__ == '__main__':
    main()

'''
# for i in range(16):
#     bin='{0:b}'.format(i)
#     gray=bin_to_gray(bin)
#     print(i, "in binary: ", bin, "  in Gray: ", gray)
'''
# Returned number is not correct

# Convert chromosome to real number
# input: list binary 1,0 of length numOfBits representing number using gray coding
# output: real value

#print(chrom2real([1,0,1,1,0,1,1,1,0,1]))
###

#
