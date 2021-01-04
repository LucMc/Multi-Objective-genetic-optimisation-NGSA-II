'''
random was used for a randomised dataset
numpy was used for mathematical operations and arrays
matplotlib was used for visualisation
sympy was used for graycode to binary conversion
pandas was used to store variables in a nicely formatted dataframe
pygmo was used to calculate the hypervolume
seaborn was used to visualise the hypervolume
'''

import random
import numpy as np
import matplotlib.pyplot as plt
from sympy.combinatorics.graycode import gray_to_bin
import pandas as pd
import pygmo as pg                   # multi-objective optimisation framework
import seaborn as sns


# display entire dataframe when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# random seed for reproducable results when testing
np.random.seed(1)
random.seed(1)
'''
numOfBits = 10*3 # Number of bits in the chromosomes
maxnum = 2**bit_length # absolute max size of number coded by grey list 1,0,0,1,1,....
flip_prob = 1/numOfBits # probability of mutation
'''


population = 28 # changed from 25
bit_length = 10
numOfBits = 10*3 # Number of bits in the chromosomes
maxnum = 2**bit_length # absolute max size of number coded by grey list 1,0,0,1,1,....
flip_prob = 1/numOfBits # probability of mutation
crossover_prob = 0.9 # probability of crossover
NGEN = 30 # number of generations

# target functions
def f1(x1, x2, x3):
    return ((x1/2.0)**2 + (x2/4.0)**2 + (x3)**2) / 3.0

def f2(x1, x2, x3):
    return ((x1/2.0-1.0)**2 + (x2/4.0-1.0)**2 + (x3-1.0)**2) / 3.0

# convert chromosome (grey list) to real
def chrom_to_real(c):
    indasstring=''.join(map(str, c))
    degray=gray_to_bin(indasstring)
    numasint=int(degray, 2) # convert to int from base 2 list
    numinrange = -4 + 8*numasint/maxnum # CHANGED TO OUR VALUES
    # print(numinrange)
    return numinrange

# create decision variables and store in dataframe
def generateDataFrame():
    df = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f1', 'f2'])

    for i in range(population):
        # random list of 0's or 1's to form gray coded chromosome
        individual = [random.randint(0, 1) for _ in range(numOfBits)]
        individual = [str(x) for x in individual]

        # split the chromosome into decision variables for the dataframe
        # making them easier to individually access
        x1 = "".join(individual[:bit_length])
        x2 = "".join(individual[bit_length:bit_length*2])
        x3 = "".join(individual[bit_length*2:])

        # find function values (to be minimised)
        _f1 = f1(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))
        _f2 = f2(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))

        # assign values to dataframe
        df.loc[i] = [x1, x2, x3, _f1, _f2]
    return df

# Efficient Non-Dominated Sorting
def ENDS(df):
    # Sort by one function
    df = df.sort_values(by="f1")
    df.reset_index(inplace=True, drop=True)
    # Assign first value to first front
    fronts = [[df.loc[0]]]
    df.at[0, 'front number'] = 1
    num_fronts = 1
    added = False
    # For every row in dataframe
    for col, row in df.loc[1:].iterrows():
        added = False
        # For every front
        for i in range(len(fronts)):
            # For every value in front (starting from back)
            for value in reversed(fronts[i]):
                # print(f'FRONT({i}) {value.name} f1: {value["f1"]} f2: {value["f2"]}')
                # print(f'ROW {row.name} f1: {row["f1"]} f2: {row["f2"]}')

                # If front value dominates row
                if row['f1'] > value['f1'] and row['f2'] > value['f2']:
                    # print('value dominates row')
                    # Check with next front. If doesn't exist create new front

                    # final front, add to new front and stop
                    if i + 1 == num_fronts:
                        num_fronts += 1
                        fronts.append([row])
                        df.at[row.name, 'front number'] = num_fronts
                        break
                # If it isn't dominated then append it to the same front and stop
                elif added == False:
                    fronts[i].append(row)
                    df.at[row.name, 'front number'] = i+1
                    added = True
                    break

    # Finally sort by the front number and return DataFrame
    df = df.sort_values(by="front number")
    return df


# Assign crowding distances
def crowding_distance(df):
    crowding_distances = []

    # For every front
    for i in range(1, int(max(df['front number'])) + 1):
        front = df.loc[df['front number'] == float(i)]
        # print("\n\n\n", front)

        # Create a new crouding_distances list for new front
        crowding_distances.append([])

        # Append infinite for first value in front
        crowding_distances[-1].append(np.inf)

        # for every other element in front (inbetween first and least elements)
        for index, element in front[1:len(front)-1].iterrows():
            # assign infinite for comparison so it will be certainly overwritten
            closest = np.inf

            # for each row in front
            for i, element2 in front.iterrows():
                # If crwoding distance is less than current closest
                if abs(element2['f2'] - element['f2']) + abs(element2['f1'] - element['f1']) < closest and\
                        abs(element2['f2'] - element['f2']) + abs(element2['f1'] - element['f1']) != 0:
                    # Assign new crowding distance
                    closest = abs(element2['f2'] - element['f2']) + abs(element2['f1'] - element['f1'])

            # Update crowding distance in DataFrame
            df.at[index, 'crowding distance'] = closest

    # Assign infinite values to last values in DataFrame
    df.fillna(value=np.inf, inplace=True)

    # Sort DataFrame by front number and crowding distance
    df.sort_values(['front number', 'crowding distance'], ascending=[True, False], inplace=True)
    return df

# Tournament selection and crossover
def tournament_selection(df):

    # Tournament Selection
    def sample_pair(df):
        # Choose two random rows in DataFrame for comparison
        selection = df.sample(n=2)

        # Compare front number, if they have the same front number use crowding distances
        if len(selection[selection['front number'] == min(selection['front number'])]) == 2:
            # Incase they are both the same crowding distance as well, choose a random one
            selection = selection[selection['crowding distance'] == min(selection['crowding distance'])].sample(n=1)
        else:
            selection = selection[selection['front number'] == min(selection['front number'])]
        return selection

    # Uniform Crossover
    def uniform(indv1, indv2):
        # Individuals chromosomes
        individual1_chromosome = "" + indv1['x1'].item() + indv1['x2'].item() + indv1['x3'].item()
        individual2_chromosome = "" + indv2['x1'].item() + indv2['x2'].item() + indv2['x3'].item()

        # Crossover probability
        if random.random() < crossover_prob: # crossover probability
            for i in range(len(individual1_chromosome)):
                if random.random() < 0.5: # Chance of picking either chromosome

                    individual1_chromosome = individual1_chromosome[:i] + individual2_chromosome[i] + individual1_chromosome[i+1:]
                    individual2_chromosome = individual2_chromosome[:i] + individual1_chromosome[i] + individual2_chromosome[i+1:]

                    # parent1_chromosome[i], parent2_chromosome[i] = parent2_chromosome[i], parent1_chromosome[i]
                if random.random() < flip_prob:
                    individual1_chromosome = individual1_chromosome[:i] + str(int(individual1_chromosome[i]) ^ 1) + individual1_chromosome[i+1:]
                if random.random() < flip_prob:
                    individual2_chromosome = individual2_chromosome[:i] + str(int(individual2_chromosome[i]) ^ 1) + individual2_chromosome[i+1:]

        return individual1_chromosome, individual2_chromosome


    # Select two parents through tournament

    parent1 = sample_pair(df)
    parent2 = sample_pair(df)
    child1, child2 = uniform(parent1, parent2)

    return child1, child2


def next_generation(df):
    next_gen_df = pd.DataFrame(columns=['x1', 'x2', 'x3', 'f1', 'f2'])
    index = 0
    # for gen in range(20):
    for i in range(int(population/2)):
        children = tournament_selection(df)

        for child in children:

            x1 = child[:bit_length]
            x2 = child[bit_length:bit_length * 2]
            x3 = child[bit_length*2:]

            _f1 = f1(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))
            _f2 = f2(chrom_to_real(x1), chrom_to_real(x2), chrom_to_real(x3))

            next_gen_df.loc[index] = [x1, x2, x3, _f1, _f2]
            index += 1

    return next_gen_df


def plot(df, initial_df):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(initial_df['f1'], initial_df['f2'], s=20, c='b', marker="o", label='initial generation')
    ax1.scatter(df['f1'], df['f2'], s=20, c='r', marker="o", label='next generation')
    plt.legend(loc='upper left')
    plt.show()


def main():
    # Q1.1
    df = generateDataFrame()
    print("Q1.1\n", df)

    # Q1.2
    df = ENDS(df)
    print("\nQ1.2\n", df[['f1', 'f2', 'front number']])
    worst_f1 = max(df['f1'])
    worst_f2 = max(df['f2'])
    print(f"\nworst f1: {worst_f1}\nworst f2: {worst_f2}")

    # Q1.3
    df = crowding_distance(df)
    print("\nQ1.3\n", df[['f1', 'f2', 'front number', "crowding distance"]])

    # Q1.4
    initial_df = df
    df = next_generation(df)
    plot(df, initial_df)

    # Q1.5 Combined
    df = pd.concat([df, initial_df])
    df = ENDS(df)
    df = crowding_distance(df)

    # Select 25 individuals based on ENDS
    df.sort_values(['front number', 'crowding distance'], ascending=[True, False], inplace=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(df[:25]['f1'], df[:25]['f2'], s=20, c='b', marker="o", label='selected generation')
    ax1.scatter(df[25:]['f1'], df[25:]['f2'], s=20, c='r', marker="o", label='overall generation')
    plt.legend(loc='upper left')
    plt.show()

    # Q1.6 Hypervolume
    hypervolumes = []
    # Calculate for every gen
    for i in range(NGEN):
        # create next generation
        df = next_generation(df)
        df = ENDS(df)
        df = crowding_distance(df)

        # Calculate hypervolume from previously determined worst values in initial gen
        hyp = pg.hypervolume(df[['f1', 'f2']].values)
        hyp = hyp.compute([worst_f1, worst_f2])
        print(f"Hypervolume: {hyp}")
        # Normalise the hypervolume
        hyp = hyp/np.prod([worst_f1, worst_f2])
        print(f'Normalised Hypervolume: {hyp}')
        hypervolumes.append(hyp)

    # plot the hypervolumes against generation number
    sns.regplot([i for i in range(len(hypervolumes))], hypervolumes)
    plt.show()

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