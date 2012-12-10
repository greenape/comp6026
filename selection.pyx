# cython: profile=True
import numpy as np
import sys
import pylab
cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_f

cdef float Gc = 0.018
cdef float Gs = 0.02
cdef float Cc = 0.1
cdef float Cs = 0.2
cdef float death_rate = 0.1
cdef float size_advantage = 1.25
cdef unsigned int pop_size = 1000
cdef unsigned int max_generations = 120
cdef unsigned int large_group_size = 40
cdef unsigned int small_group_size = 4
cdef unsigned int group_time = 4
# Number of each genotype in the pool
# Index 0 is cooperative, index 1 is greedy
large = np.zeros(2, dtype=np.int32)
small = np.zeros(2, dtype=np.int32)


@cython.boundscheck(False)
def reproduce_groups(groups):
    """ Perform reproduction on all the groups provided and
    return the resulting groups.
    """
    cdef unsigned int i
    # Small groups
    for i in range(len(groups[0])):
        #print "small pre reproduce:", groups[0][i]
        groups[0][i] = reproduce_group(small_group_size, groups[0][i])
        #print "small after reproduce:", groups[0][i]
    # Large groups
    for i in range(len(groups[1])):
        groups[1][i] = reproduce_group(large_group_size * size_advantage, groups[1][i])
    return groups


@cython.boundscheck(False)
cdef np.ndarray[DTYPE_f, ndim=2] reproduce_group(float resource, np.ndarray[DTYPE_f, ndim=2] group):
    """ Perform reproduction in a group.
    """
    cdef np.ndarray[DTYPE_f, ndim=2] new_group = np.zeros((len(group), 3))
    cdef float r = 0
    cdef unsigned int groups_num = len(group)
    cdef unsigned int i
    for i in range(groups_num):
        r = resource_share(group[i], group, resource)
        #print "Reproducing group %d of %d with %f resources." % (i + 1, groups_num, r)
        new_group[i] = reproduce(group[i], r)
    return new_group


@cython.boundscheck(False)
cdef np.ndarray[DTYPE_f] reproduce(np.ndarray[DTYPE_f] target, float resource):
    """ Perform reproduction of a single group.
    """
    cdef result = np.array(target)
    result[0] = target[0] + resource / target[1] - death_rate * target[0]
    return result



cdef float resource_share(np.ndarray[DTYPE_f] target, np.ndarray[DTYPE_f, ndim=2] others, float R):
    """ Calculate the magnitude of a group's resources
    a particular genotype receives.
    """
    #print "Target:",target, "Others:",others
    cdef float group_sum = np.sum(np.product(others, axis=1))
    if group_sum > 0:
        return R * (np.product(target) / group_sum)
    else:
        return R


@cython.boundscheck(False)
cdef void disperse(groups):
    global large, small
    cdef unsigned int i
    # Return small groups
    for i in range(len(groups[0])):
        small[0] = small[0] + groups[0][i][0][0]
        small[1] = small[1] + groups[0][i][1][0]
    for i in range(len(groups[1])):
        large[0] = large[0] + groups[1][i][0][0]
        large[1] = large[1] + groups[1][i][1][0]
    np.floor(small)
    np.floor(large)
    #print "large:",large, "small:",small



def aggregate():
    """ Assign individuals to groups based on
    their group size preference. Index 0 for small groups
    and index 1 for large.
    """
    global large, small
    groups = []
    small_groups = []
    large_groups = []
    cdef np.ndarray[DTYPE_f, ndim=2] group
    cdef unsigned int i, group_num
    cdef float random_bias

    # Make small groups
    opt = [0 for x in range(int(small[0]))] + [1 for x in range(int(small[1]))]
    #print opt
    np.random.shuffle(opt)
    while len(opt) >= small_group_size:
        group = np.array([[0, Cc, Gc], [0, Cs, Gs]])
        for i in range(small_group_size):
            group_num = opt.pop()
            group[group_num, 0] = group[group_num, 0] + 1
        #print "Small:", group
        small_groups.append(group)

    opt = [0 for x in range(int(large[0]))] + [1 for x in range(int(large[1]))]
    np.random.shuffle(opt)
    while len(opt) >= large_group_size:
        group = np.array([[0, Cc, Gc], [0, Cs, Gs]])
        for i in range(large_group_size):
            group_num = opt.pop()
            group[group_num, 0] = group[group_num, 0] + 1
        #print "Large:", group
        large_groups.append(group)

    # Zero remainder
    large = np.zeros(2)
    small = np.zeros(2)

    # Add groups
    groups.append(small_groups)
    groups.append(large_groups)

    #print "Made %d small groups, %d large groups." % (len(small_groups), len(large_groups))

    return groups


cdef np.ndarray[DTYPE_f, ndim=2] group_sizes():
    """ Return the relative frequency of each group size
    at this juncture, index 0 is large, 1 is small.
    """
    return np.array([[np.sum(large), np.sum(small)]]) / float(pop_size)


cdef np.ndarray[DTYPE_f, ndim=2] resource_use():
    """ Return the relative frequency of each resource use
    type. Index 0 is cooperative, 1 is selfish.
    """
    return np.array([[large[0] + small[0], large[1] + small[1]]]) / float(pop_size)


cdef np.ndarray[DTYPE_f, ndim=2] genotypes():
    """ Return the relative frequency of each genotype.
    0 = cooperative-large
    1 = selfish-large
    2 = cooperative-small
    3 = selfish-small
    """

    return np.array([[large[0], large[1], small[0], small[1]]]) / float(pop_size)


def rescale():
    """ Rescale the population to pop_size maintaining
    proportions of genotypes.
    """
    global small, large
    cdef float current_pop_size = np.sum(small) + np.sum(large)
    #print "pop_size:",current_pop_size
    cdef float scale_factor = pop_size / current_pop_size
    small = small * scale_factor
    large = large * scale_factor
    #print "Rescaled to:", np.sum(small) + np.sum(large), "Small:", np.sum(small), "Large:", np.sum(large)


def init():
    """ Initialise the migrant pool with a population
    of pop_size random individuals.
    """
    global small, large
    small = np.ones(2) * pop_size / 4
    large = np.ones(2) * pop_size / 4


def run():
    """ Run an experiment.
    """
    cdef unsigned int i, t
    resource_results = np.zeros((max_generations, 2))
    genotype_results = np.zeros((max_generations, 4))
    group_size_results = np.zeros((max_generations, 2))
    init()
    for i in range(max_generations):
        resource_results[i] = resource_results[i] + resource_use()
        genotype_results[i] = genotype_results[i] + genotypes()
        group_size_results[i] = group_size_results[i] + group_sizes()
        groups = aggregate()
        for t in range(group_time):
            groups = reproduce_groups(groups)
            #print "Reproduce cycle", t
        disperse(groups)
        rescale()
        #print i, resource_results[i], resource_use()
    #print resource_results
    pylab.plot(range(max_generations), np.rot90(resource_results)[0], 'b', label="Selfish")
    pylab.plot(range(max_generations), np.rot90(group_size_results)[1], 'g', label="Large")
    pylab.legend()
    pylab.show()

    pylab.plot(range(max_generations), np.rot90(genotype_results)[0], 'b', label="Selfish-small")
    pylab.plot(range(max_generations), np.rot90(genotype_results)[1], 'g', label="cooperative-small")
    pylab.plot(range(max_generations), np.rot90(genotype_results)[2], 'r', label="selfish-large")
    pylab.plot(range(max_generations), np.rot90(genotype_results)[3], 'y', label="cooperative-large")
    pylab.legend()
    pylab.show()
