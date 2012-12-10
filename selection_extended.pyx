# cython: profile=True
import numpy as np
import sys
import pylab
cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f

G = [0.018, 0.02]
C = [0.1, 0.2]
cdef float death_rate = 0.1
cdef float size_advantage = 1.25
cdef float mut_rate = 0.1
cdef unsigned int pop_size = 1000
cdef unsigned int max_generations = 100
cdef unsigned int large_group_size = 40
cdef unsigned int small_group_size = 4
cdef unsigned int group_time = 1

# Population pool
dtype = [('group_size', int), ('resource_use', int),('group_time', int)]
population = np.array([], dtype=dtype)


@cython.boundscheck(False)
def reproduce_groups(groups):
    """ Perform reproduction on all the groups provided and
    return the resulting groups.
    """
    cdef unsigned int i
    # Small groups
    for i in range(len(groups)):
        #print "small pre reproduce:", groups[0][i]
        # Resource allocation
        resources = groups[i][0][0][0]
        if resources > 4:
            resources *= size_advantage
        groups[i] = reproduce_group(resources, groups[i])
        #print "small after reproduce:", groups[0][i]
    return groups


@cython.boundscheck(False)
def reproduce_group(float resource, group):
    """ Perform reproduction in a group.
    """
    cdef float r = 0
    cdef unsigned int i
    cdef unsigned int groups_num = len(group)
    new_group = []
    for i in range(groups_num):
        r = resource_share(group[i], group, resource)
        #print "Reproducing group %d of %d with %f resources." % (i + 1, groups_num, r)
        new_pop = reproduce(group[i], r)
        #print "New pop size is %d from %d originals." % (new_pop[1], group[i][1])
        new_group += mutate(new_pop, group[i][1] - death_rate * group[i][1])
    return new_group


@cython.boundscheck(False)
def count_type(group):
    """ Return a list of genotypes and the frequency
    of that genotype in the provided group.
    """
    keys = np.unique(np.array(group))
    bins = keys.searchsorted(group)
    return zip(keys, np.bincount(bins))


@cython.boundscheck(False)
def reproduce(target, float resource):
    """ Perform reproduction of a single group.
    """
    genome, count = target
    new_size = count + resource / C[genome[1]] - death_rate * count
    #print new_size, resource, C[genome[1]], count
    return (genome, new_size)


@cython.boundscheck(False)
def resource_share(target, others, float R):
    """ Calculate the magnitude of a group's resources
    a particular genotype receives.
    """
    #print "Target:",target, "Others:",others
    cdef float group_sum = 0
    cdef float target_sum = target[1] * G[target[0][1]] * C[target[0][1]]
    for genotype, count in others:
        group_sum += count * G[genotype[1]] * C[genotype[1]]

    return R * target_sum / group_sum


@cython.boundscheck(False)
def disperse(groups):
    global population
    pop = []
    for group in groups:
        for genotype, count in group:
            pop += [genotype] * np.floor(count)
    population = np.array(pop, dtype=dtype)


@cython.boundscheck(False)
def subpops():
    """ Return a list of subpopulations in the overall population,
    split by resource use type and group time duration.
    """
    sorted_population = np.sort(population, order=['group_size', 'group_time'])
    subpops = []
    subpop = []
    # Break down to sublists
    for i in range(len(sorted_population)):
        # New subpopulation if different group size or group time
        if i > 0 and (sorted_population[i-1][0] != sorted_population[i][0] or
         sorted_population[i-1][2] != sorted_population[i][2]):
            np.random.shuffle(subpop)
            subpops.append(subpop)
            subpop = []
        subpop.append(sorted_population[i])
    np.random.shuffle(subpop)
    subpops.append(subpop)
    return subpops


@cython.boundscheck(False)
def aggregate():
    """ Assign individuals to groups based on
    their group size preference. Index 0 for small groups
    and index 1 for large.
    """
    groups = []
    small_groups = []
    large_groups = []
    cdef int group_size_pref, group_time_pref

    # Make subpopulations
    subs = subpops()
    
    for sub in subs:
        # Get group size pref
        group_size_pref = sub[0][0]
        # Get group time pref
        group_time_pref = sub[0][2]
        while len(sub) >= group_size_pref:
            group = []
            for i in range(group_size_pref):
                group += [sub.pop()]
            groups += [count_type(group)]

    return groups


@cython.boundscheck(False)
def mutate(individual, num_ineligible):
    """ Mutate an genotype population, preserving some
    number ineligible for mutation.
    """
    genome, pop_size = individual
    pop = [genome] * (pop_size - num_ineligible)
    new_pop = [genome] * num_ineligible
    for (a, b, c) in pop:
        if np.random.rand() < mut_rate:
            #print "Mutating", a, b, c
            step = 1
            if np.random.rand() > 0.5:
                step = -1
            new_pop += [(a, b, step + c)]#min(max(step + c, 1), 10))]
        else:
            new_pop += [(a, b, c)]
    #print np.array(new_pop, dtype=dtype)
    return count_type(np.array(new_pop, dtype=dtype))


@cython.boundscheck(False)
def genotypes():
    """ Return the relative frequency of each genotype.
    """
    genotypes, counts = zip(*count_type(population))
    return [(tuple(x), y) for (x, y) in zip(genotypes, np.asarray(counts) / float(pop_size))]


@cython.boundscheck(False)
def gross_genotypes():
    """ Return the relative frequency of each of the fixed allele
    comibinations.
    """
    gs = genotypes()
    gross = {}
    for (x, y, z), proportion in gs:
        if (x, y) in gross:
            gross[(x, y)] += proportion
        else:
            gross[(x, y)] = proportion
    return [(tuple(x), y) for x, y in gross.items()]


@cython.boundscheck(False)
def timing_means():
    gs = count_type(population)
    means = {}
    for (x, y, z), proportion in gs:
        if (x, y) in means:
            p, q = means[(x, y)]
            means[(x, y)] = (z * proportion + p, proportion + q)
        else:
            means[(x, y)] = (z * proportion, float(proportion))
    return [(tuple(x), a / b) for x, (a, b) in means.items()]


@cython.boundscheck(False)
def timing_mean():
    gs = genotypes()
    mean = 0
    for (x, y, z), proportion in gs:
        mean += z * proportion
    return mean / float(pop_size)


@cython.boundscheck(False)
def rescale():
    """ Rescale the population to pop_size maintaining
    proportions of genotypes.
    """
    global population
    cdef float current_pop_size = len(population)
    #print "pop_size:",current_pop_size
    cdef float scale_factor = pop_size / current_pop_size
    new_pop = []
    for genome, count in count_type(population):
        new_pop += [genome] * int(count * scale_factor) 
    population = np.array(new_pop, dtype=dtype)
    #print "Rescaled to:", np.sum(small) + np.sum(large), "Small:", np.sum(small), "Large:", np.sum(large)


@cython.boundscheck(False)
def init():
    """ Initialise the migrant pool with a population
    of pop_size individuals.
    """
    global population
    cdef int type_num = pop_size / 4
    pop = [(small_group_size, 0, group_time)] * type_num
    pop += [(small_group_size, 1, group_time)] * type_num
    pop += [(large_group_size, 0, group_time)] * type_num
    pop += [(large_group_size, 1, group_time)] * type_num
    population = np.array(pop, dtype=dtype)


@cython.boundscheck(False)
def __main__():
    """ Run an experiment.
    """
    cdef unsigned int i, t
    init()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in gross_genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    timings_means_results = dict([(x, [y]) for (x, y) in timing_means()])
    cdef float resources
    for i in range(max_generations):
        #resource_results[i] = resource_results[i] + resource_use()
        
        for genotype, count in genotype_results.items():
            count += [0]
            #print genotype_results[genotype]

        for genotype, count in gross_genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count
        for genotype, mean in timing_means():
            timings_means_results[genotype] += [mean]

        #group_size_results[i] = group_size_results[i] + group_sizes()
        groups = aggregate()
        new_groups = []
        for group in groups:
            for t in range(group[0][0][2]):
                resources = float(group[0][0][0])
                if resources > 4:
                    resources *= size_advantage
                group = reproduce_group(resources, group)
                #print group
                #print "Reproduce cycle", t
            new_groups += [group]
        disperse(new_groups)
        #print count_type(population)
        rescale()
        print "gen", i, count_type(population)
        #print i, resource_results[i], resource_use()
    #print resource_results
    for key, value in genotype_results.items():
        pylab.plot(range(len(value)), value, label=str(key) + " Pop")
    pylab.legend()
    pylab.show()
    for key, value in timings_means_results.items():
        pylab.plot(range(len(value)), value, label=str(key) + " Group time")
    pylab.legend()
    pylab.show()
