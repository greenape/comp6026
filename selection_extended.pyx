#cython: profile=True
import numpy as np
import sys
import pylab
cimport cython
cimport numpy as np

ctypedef np.int_t DTYPE_t
ctypedef double DTYPE_f

G = [0.018, 0.02]
C = [0.1, 0.2]
cdef double death_rate = 0.1
cdef double size_advantage = 1.25
cdef double mut_rate = 0.0
cdef unsigned int pop_size = 4000
cdef unsigned int max_generations = 1000
cdef unsigned int large_group_size = 40
cdef int small_group_size = 4
cdef int group_time = 4
genotype_name = {(40, 0):"Large", (40, 1):"Large-selfish", (4, 0):"Small",(4, 1):"Small-selfish"}

# Population pool
dtype = [('group_size', int), ('resource_use', int),('group_time', int)]
population = np.array([], dtype=dtype)


@cython.boundscheck(False)
def reproduce_group(float resource, group):
    """ Perform reproduction in a group.
    """
    cdef double r = 0
    cdef unsigned int i
    cdef unsigned int groups_num = len(group)
    new_group = []
    for i in range(groups_num):
        r = resource_share(group[i], group, resource)
        #print "Reproducing group %d of %d with %f resources." % (i + 1, groups_num, r)
        new_pop = reproduce(group[i], r)
        #print "New pop size is %d from %d originals." % (new_pop[1], group[i][1])
        new_group += [new_pop]
        #mutate(new_pop, group[i][1] - death_rate * group[i][1]) 
    return new_group


@cython.boundscheck(False)
def count_type(group):
    """ Return a list of genotypes and the frequency
    of that genotype in the provided group.
    """
    keys = np.unique(np.array(group))
    bins = keys.searchsorted(group)
    # Weird bit here to avoid the ints being stuck as numpy int64
    # and breaking tuple equalities
    keys = map(lambda x: tuple(map(int, x)), keys)
    return zip(keys, np.bincount(bins))


@cython.boundscheck(False)
def reproduce(target, float resource):
    """ Perform reproduction of a single group.
    """
    genome, count = target
    cdef double new_size = count + resource / C[genome[1]] - death_rate * count
    if new_size < 1:
        new_size = 0
    #print new_size, resource, C[genome[1]], count
    return (genome, new_size)


@cython.boundscheck(False)
def resource_share(target, others, float R):
    """ Calculate the magnitude of a group's resources
    a particular genotype receives.
    """
    #print "Target:",target, "Others:",others
    cdef double group_sum = 0
    cdef double target_sum = target[1] * G[target[0][1]] * C[target[0][1]]
    for genotype, count in others:
        group_sum += count * G[genotype[1]] * C[genotype[1]]
    if group_sum == 0:
        return 0.

    return R * target_sum / group_sum


@cython.boundscheck(False)
def disperse(groups):
    global population
    pop = []
    for group in groups:
        for genotype, count in group:
            pop += [genotype] * int(count)
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
def aggregate_single():
    """ Assign individuals to one big group.
    """
    groups = []
    small_groups = []
    large_groups = []
    cdef int group_size_pref, group_time_pref

    # Make subpopulations
    subs = subpops()
    
    for sub in subs:
        # Get group size pref
        group_size_pref = len(population)
        # Get group time pref
        #group_time_pref = sub[0][2]
        while len(sub) >= group_size_pref:
            group = []
            for i in range(group_size_pref):
                group += [sub.pop()]
            groups += [count_type(group)]

    return groups


@cython.boundscheck(False)
def agglomerate(group1, group2):
    """ Merge two groups into one.
    """
    # Add to the biggest group
    #print "Glomming",group1,group2
    group1_bigger = len(group1) > len(group2)
    group = list(group1) if group1_bigger else list(group2)
    groupb = list(group2) if group1_bigger else list(group1)
    for key, count in groupb:
        found = False
        for i in range(len(group)):
            #print "Checking", key, "vs", group[i][0], key == group[i][0]
            if key == group[i][0]:
                found = True
                break
        if found:
            group[i] = (key, int(group[i][1]) + int(count))
        else:
            group += [(key, int(count))]
    #print "Glommed", group
    return group


@cython.boundscheck(False)
def genotypes():
    """ Return the relative frequency of each genotype.
    """
    genotypes, counts = zip(*count_type(population))
    return [(tuple(x), y) for (x, y) in zip(genotypes, np.asarray(counts) / float(pop_size))]


@cython.boundscheck(False)
def gross_genotypes():
    """ Return the relative frequency of each of the fixed allele
    combinations.
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
def rescale(group):
    """ Rescale the group to pop_size maintaining
    proportions of genotypes.
    """
    cdef float current_size = 0
    cdef float scale_factor
    new_group = []
    for key, count in group:
        current_size += np.floor(count)
    if current_size > pop_size:
        scale_factor = pop_size / current_size
        for key, count in group:
            new_group += [(key, round(count * scale_factor))]
        return new_group
    else:
        return group



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


def init_time():
    """ Initialise the migrant pool with a population
    of pop_size individuals and two time groups with fixed group size.
    """
    global population
    cdef int type_num = pop_size / 4
    pop = [(small_group_size, 0, 0.25*group_time)] * type_num
    pop += [(small_group_size, 1, 0.25*group_time)] * type_num
    pop += [(small_group_size, 0, 0.75*group_time)] * type_num
    pop += [(small_group_size, 1, 0.75*group_time)] * type_num
    population = np.array(pop, dtype=dtype)

def init_time_size():
    """ Initialise the pool with a population of
    mixed group sizes and fixed time preferences.
    """

    global population
    cdef int type_num = pop_size / 4
    pop = [(small_group_size, 0, 0.75*group_time)] * type_num
    pop += [(small_group_size, 1, 0.75*group_time)] * type_num
    pop += [(large_group_size, 0, 0.75*group_time)] * type_num
    pop += [(large_group_size, 1, 0.75*group_time)] * type_num
    population = np.array(pop, dtype=dtype)

def init_short_time_size():
    """ Initialise the pool with a population of
    mixed group sizes and short time preferences.
    """

    global population
    cdef int type_num = pop_size / 4
    pop = [(small_group_size, 0, 0.25*group_time)] * type_num
    pop += [(small_group_size, 1, 0.25*group_time)] * type_num
    pop += [(large_group_size, 0, 0.25*group_time)] * type_num
    pop += [(large_group_size, 1, 0.25*group_time)] * type_num
    population = np.array(pop, dtype=dtype)


def init_mix():
    """ Initialise the pool with a population of
    mixed group sizes and time preferences.
    """

    global population
    cdef int type_num = pop_size / 8
    pop = [(small_group_size, 0, 1*group_time)] * type_num
    pop += [(small_group_size, 1, 1*group_time)] * type_num
    pop += [(small_group_size, 0, 0.75*group_time)] * type_num
    pop += [(small_group_size, 1, 0.75*group_time)] * type_num
    pop += [(large_group_size, 0, 1*group_time)] * type_num
    pop += [(large_group_size, 1, 1*group_time)] * type_num
    pop += [(large_group_size, 0, 0.75*group_time)] * type_num
    pop += [(large_group_size, 1, 0.75*group_time)] * type_num
    population = np.array(pop, dtype=dtype)



@cython.boundscheck(False)
cdef np.ndarray[DTYPE_f, ndim=2] resource_use():
    """ Return the relative frequency of each resource use
    type. Index 0 is cooperative, 1 is selfish.
    """
    gs = genotypes()
    gross = {}
    gross[0] = 0
    gross[1] = 0
    for (x, y, z), proportion in gs:
        if y in gross:
            gross[y] += proportion
        else:
            gross[y] = proportion
    return np.array([[gross[0], gross[1]]])


@cython.boundscheck(False)
def coop_fix():
    """ Test whether cooperative trait has reached fixation
    in the population. """
    print "proportion: %f %f" % (resource_use()[0][0], resource_use()[0][1])
    return resource_use()[0][0] > 0.995


@cython.boundscheck(False)
def selfish_fix():
    """ Test whether selfish trait has reached fixation
    in the population. """
    return resource_use()[0][1] > 0.99 


@cython.boundscheck(False)
def run_fig_1():
    """ 2D parameter exploration.
    """
    global small_group_size, population, group_time
    cdef int type_num = pop_size / 2
    cdef int i, t
    cdef str log
    dump_file = open("dump_fig_1.csv", 'w')
    dump_file.write("Size,Time,Fixated\n")
    dump_file.close()
    # For group sizes 1 - 8
    for small_group_size in range(1, 9):
            # For group times from 0.25, 0.5, 0.75 of total time
            for group_time in range(2, 101):
                # Initialise the population
                pop = [(small_group_size, 0, group_time)] * type_num
                pop += [(small_group_size, 1, group_time)] * type_num
                population = np.array(pop, dtype=dtype)
                for i in range(max_generations):
                    groups = aggregate()
                    population = np.array([], dtype=dtype)
                    new_groups = []
                    working_population = [((small_group_size, 0, group_time), 0.), ((small_group_size, 1, group_time), 0.)]
                    for t in range(group_time + 1):
                        #print "Gen",i,"Timestep", t
                        new_groups = []
                        # Reproduce groups
                        for group in groups:
                            if t < group[0][0][2]:
                                #print "In group"
                                resources = float(group[0][0][0])
                                if resources > small_group_size:
                                    resources *= size_advantage
                                new_groups += [reproduce_group(resources, group)]
                            elif t == group[0][0][2]:
                                #print "Glomming"
                                working_population = agglomerate(working_population, group)
                        groups = new_groups

                        # Reproduce pool
                        #print "WP:",working_population
                        #working_population = reproduce_group(200, working_population)

                    working_population = rescale(working_population)
                    print working_population
                    disperse([working_population])
                    print len(population)
        
                    #rescale()
                    if coop_fix() or selfish_fix():
                        break
                    #print resource_use()

                log =  "%d, %d, %s\n" % (small_group_size, group_time, coop_fix())
                dump_file = open("dump_fig_1.csv", 'a')
                dump_file.write(log)
                dump_file.close()
                print log
    group_time = 4
    small_group_size = 4


@cython.boundscheck(False)
def run_fig_2(dump_file_name="dump_fig_2.csv"):
    """ Basic group competition experiment.
    """
    cdef unsigned int i, t
    global max_generations, genotype_name
    max_generations = 120
    init()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in gross_genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    cdef float resources

    global small_group_size, population, group_time
    cdef str log
    dump_file = open(dump_file_name, 'w')
    dump_file.write("Generation, Genotype, Proportion\n")
    dump_file.close()
    for i in range(max_generations):
        for genotype, count in genotype_results.items():
            count += [0]

        for genotype, count in gross_genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count

        groups = aggregate()
        new_groups = []
        working_population = []
        for t in range(group_time + 1):
                        #print "Gen",i,"Timestep", t
            new_groups = []
                        # Reproduce groups
            for group in groups:
                #print group
                if t < group[0][0][2]:
                                #print "In group"
                    resources = float(group[0][0][0])
                    if resources > small_group_size:
                        resources *= size_advantage
                    new_groups += [reproduce_group(resources, group)]
                elif t == group[0][0][2]:
                                #print "Glomming"
                    working_population = agglomerate(working_population, group)
            groups = new_groups

                        # Reproduce pool
        #print "WP:",working_population
                        #working_population = reproduce_group(200, working_population)

        working_population = rescale(working_population)
        disperse([working_population])
    dump_file = open(dump_file_name, 'a')
    for key, value in genotype_results.items():
        i = 0
        for proportion in value:
            log =  "%d, %s, %f\n" % (i, genotype_name[key], proportion)
            i += 1
            dump_file.write(log)
            print log
    dump_file.close()
    group_time = 4
    small_group_size = 4


@cython.boundscheck(False)
def run_fig_3():
    """ No size advantage. """
    global size_advantage
    size_advantage = 1
    run_fig_2("dump_fig_3.csv")
    size_advantage = 1.25


@cython.boundscheck(False)
def run_fig_4():
    """ As fig 1, but with reproduction in the group pool and
    time in groups as fraction of turn time.
    """
    global small_group_size, population, group_time
    cdef int type_num = pop_size / 2
    cdef int i, t
    cdef str log
    dump_file = open("dump_fig_4.csv", 'w')
    dump_file.write("Size,TotalTime,GTime,Fixated\n")
    dump_file.close()
    # For group sizes 1 - 8
    for small_group_size in range(1, 9):
        for total_time in [x*4 for x in range(1, 26)]:
            # For group times from 0.25, 0.5, 0.75 of total time
            for group_time in [0.25 * total_time, 0.5 * total_time, 0.75 * total_time]:
                # Initialise the population
                pop = [(small_group_size, 0, group_time)] * type_num
                pop += [(small_group_size, 1, group_time)] * type_num
                population = np.array(pop, dtype=dtype)
                for i in range(max_generations):
                    groups = aggregate()
                    population = np.array([], dtype=dtype)
                    new_groups = []
                    working_population = [((small_group_size, 0, group_time), 0.), ((small_group_size, 1, group_time), 0.)]
                    for t in range(total_time):
                        print "Gen",i,"Timestep", t
                        new_groups = []
                        # Reproduce groups
                        for group in groups:
                            if t < group[0][0][2]:
                                #print "In group"
                                resources = float(group[0][0][0])
                                if resources > small_group_size:
                                    resources *= size_advantage
                                new_groups += [reproduce_group(resources, group)]
                            elif t == group[0][0][2]:
                                #print "Glomming"
                                working_population = agglomerate(working_population, group)
                        groups = new_groups

                        # Reproduce pool
                        print "WP:",working_population
                        working_population = reproduce_group(pop_size*1.25, working_population)

                    disperse([rescale(working_population)])
                    print len(population)
        
                    #rescale()
                    if coop_fix() or selfish_fix():
                        break
                    #print resource_use()
                fixation = coop_fix()
                log =  "%d, %d, %f, %s\n" % (small_group_size, total_time, group_time, fixation)
                if fixation:
                    dump_file = open("dump_fig_4.csv", 'a')
                    dump_file.write(log)
                    dump_file.close()
                print log
    group_time = 4
    small_group_size = 4

@cython.boundscheck(False)
def run_fig_5(dump_file_name="dump_fig_5.csv"):
    """ As fig 2, but structured by time instead of group size, with a group size of (4, 4)
    and group time proportion of 0.25 & 0.75. (coop dominates for 0.75, does not for 0.25)
    """
    cdef unsigned int i, t
    global max_generations, genotype_name, small_group_size, group_time
    small_group_size = 4
    group_time = 4
    genotype_name = {(small_group_size, 1, int(0.25*group_time)):"Short-selfish", (small_group_size, 1,int(0.75*group_time)):"Long-selfish",
     (small_group_size, 0, int(0.75*group_time)):"Long",(small_group_size, 0, int(0.25*group_time)):"Short"}
    max_generations = 120
    init_time()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    cdef float resources

    global small_group_size, population, group_time
    cdef str log
    dump_file = open(dump_file_name, 'w')
    dump_file.write("Generation, Genotype, Proportion\n")
    dump_file.close()
    for i in range(max_generations):
        for genotype, count in genotype_results.items():
            count += [0]

        for genotype, count in genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count

        groups = aggregate()
        new_groups = []
        working_population = []
        returned_groups = 0
        for t in range(group_time ):
                        #print "Gen",i,"Timestep", t
            new_groups = []
                        # Reproduce groups
            for group in groups:
                #print group
                if t < group[0][0][2]:
                                #print "In group"
                    resources = float(group[0][0][0])
                    if resources > small_group_size:
                        resources *= size_advantage
                    new_groups += [reproduce_group(resources, group)]
                elif t == group[0][0][2]:
                                #print "Glomming"
                    returned_groups += 1
                    working_population = agglomerate(working_population, group)
            groups = new_groups

                        # Reproduce pool
        #print "WP:",working_population
            working_population = reproduce_group(returned_groups*small_group_size*1.25, working_population)

        working_population = rescale(working_population)
        disperse([working_population])
    dump_file = open(dump_file_name, 'a')
    for key, value in genotype_results.items():
        i = 0
        for proportion in value:
            #genotype_name[key]
            log =  "%d, %s, %f\n" % (i, genotype_name[key], proportion)
            i += 1
            if i == max_generations:
                break
            dump_file.write(log)
            print log
    dump_file.close()
    group_time = 4
    small_group_size = 4


@cython.boundscheck(False)
def run_fig_6(dump_file_name="dump_fig_6.csv"):
    """ As fig 5, but with resource influx to the mixing pool based on
    number of agglomerated groups*group size not population size. So,
    no advantage to early return to pool.
    """
    cdef unsigned int i, t
    global max_generations, genotype_name, small_group_size, group_time
    small_group_size = 4
    group_time = 4
    genotype_name = {(small_group_size, 1, int(0.25*group_time)):"Short-selfish", (small_group_size, 1,int(0.75*group_time)):"Long-selfish",
     (small_group_size, 0, int(0.75*group_time)):"Long",(small_group_size, 0, int(0.25*group_time)):"Short"}
    max_generations = 120
    init_time()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    cdef float resources

    global small_group_size, population, group_time
    cdef str log
    dump_file = open(dump_file_name, 'w')
    dump_file.write("Generation, Genotype, Proportion\n")
    dump_file.close()
    for i in range(max_generations):
        for genotype, count in genotype_results.items():
            count += [0]

        for genotype, count in genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count

        groups = aggregate()
        new_groups = []
        working_population = []
        returned_groups = 0
        for t in range(group_time ):
                        #print "Gen",i,"Timestep", t
            new_groups = []
                        # Reproduce groups
            for group in groups:
                #print group
                if t < group[0][0][2]:
                                #print "In group"
                    resources = float(group[0][0][0])
                    if resources > small_group_size:
                        resources *= size_advantage
                    new_groups += [reproduce_group(resources, group)]
                elif t == group[0][0][2]:
                                #print "Glomming"
                    returned_groups += 1
                    working_population = agglomerate(working_population, group)
            groups = new_groups

                        # Reproduce pool
        #print "WP:",working_population
            working_population = reproduce_group(returned_groups*small_group_size, working_population)

        working_population = rescale(working_population)
        disperse([working_population])
    dump_file = open(dump_file_name, 'a')
    for key, value in genotype_results.items():
        i = 0
        for proportion in value:
            #genotype_name[key]
            log =  "%d, %s, %f\n" % (i, genotype_name[key], proportion)
            i += 1
            if i == max_generations:
                break
            dump_file.write(log)
            print log
    dump_file.close()
    group_time = 4
    small_group_size = 4

@cython.boundscheck(False)
def run_fig_7(dump_file_name="dump_fig_7.csv"):
    """ As fig 2, but with some time spent in the group pool.
    """
    cdef unsigned int i, t
    global max_generations, genotype_name, small_group_size, group_time
    small_group_size = 4
    group_time = 4
    genotype_name = {(small_group_size, 1, int(0.75*group_time)):"Small-selfish", (large_group_size, 1,int(0.75*group_time)):"Large-selfish",
     (small_group_size, 0, int(0.75*group_time)):"Small-cooperative",(large_group_size, 0, int(0.75*group_time)):"Large-cooperative"}
    max_generations = 120
    init_time_size()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    cdef float resources

    global small_group_size, population, group_time
    cdef str log
    dump_file = open(dump_file_name, 'w')
    dump_file.write("Generation, Genotype, Proportion\n")
    dump_file.close()
    for i in range(max_generations):
        for genotype, count in genotype_results.items():
            count += [0]

        for genotype, count in genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count

        groups = aggregate()
        new_groups = []
        working_population = []
        returned_groups = {large_group_size:0,small_group_size:0}
        for t in range(group_time ):
                        #print "Gen",i,"Timestep", t
            new_groups = []
                        # Reproduce groups
            for group in groups:
                #print group
                if t < group[0][0][2]:
                                #print "In group"
                    resources = float(group[0][0][0])
                    if resources > small_group_size:
                        resources *= size_advantage
                    new_groups += [reproduce_group(resources, group)]
                elif t == group[0][0][2]:
                                #print "Glomming"
                    returned_groups[group[0][0][0]] += 1
                    working_population = agglomerate(working_population, group)
            groups = new_groups

                        # Reproduce pool
        #print "WP:",working_population
            resource = returned_groups[small_group_size]*small_group_size*1.25 + returned_groups[large_group_size]*large_group_size*1.25
            working_population = reproduce_group(resource, working_population)

        working_population = rescale(working_population)
        disperse([working_population])
    dump_file = open(dump_file_name, 'a')
    for key, value in genotype_results.items():
        i = 0
        for proportion in value:
            #genotype_name[key]
            log =  "%d, %s, %f\n" % (i, genotype_name[key], proportion)
            i += 1
            if i == max_generations:
                break
            dump_file.write(log)
            print log
    dump_file.close()
    group_time = 4
    small_group_size = 4


@cython.boundscheck(False)
def run_fig_8(dump_file_name="dump_fig_8.csv"):
    """ As fig 7, but with resource influx to the mixing pool fixed
    at population size. Advantage to early return, none to late.
    """
    cdef unsigned int i, t
    global max_generations, genotype_name, small_group_size, group_time
    small_group_size = 4
    group_time = 4
    genotype_name = {(small_group_size, 1, int(0.75*group_time)):"Small-selfish", (large_group_size, 1,int(0.75*group_time)):"Large-selfish",
     (small_group_size, 0, int(0.75*group_time)):"Small-cooperative",(large_group_size, 0, int(0.75*group_time)):"Large-cooperative"}
    max_generations = 120
    init_time_size()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    cdef float resources

    global small_group_size, population, group_time
    cdef str log
    dump_file = open(dump_file_name, 'w')
    dump_file.write("Generation, Genotype, Proportion\n")
    dump_file.close()
    for i in range(max_generations):
        for genotype, count in genotype_results.items():
            count += [0]

        for genotype, count in genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count

        groups = aggregate()
        new_groups = []
        working_population = []
        returned_groups = 0
        for t in range(group_time ):
                        #print "Gen",i,"Timestep", t
            new_groups = []
                        # Reproduce groups
            for group in groups:
                #print group
                if t < group[0][0][2]:
                                #print "In group"
                    resources = float(group[0][0][0])
                    if resources > small_group_size:
                        resources *= size_advantage
                    new_groups += [reproduce_group(resources, group)]
                elif t == group[0][0][2]:
                                #print "Glomming"
                    returned_groups += 1
                    working_population = agglomerate(working_population, group)
            groups = new_groups

                        # Reproduce pool
        #print "WP:",working_population
            working_population = reproduce_group(pop_size, working_population)

        working_population = rescale(working_population)
        disperse([working_population])
    dump_file = open(dump_file_name, 'a')
    for key, value in genotype_results.items():
        i = 0
        for proportion in value:
            #genotype_name[key]
            log =  "%d, %s, %f\n" % (i, genotype_name[key], proportion)
            i += 1
            if i == max_generations:
                break
            dump_file.write(log)
            print log
    dump_file.close()
    group_time = 4
    small_group_size = 4


@cython.boundscheck(False)
def run_fig_10(dump_file_name="dump_fig_10.csv", pool_advantage=1):
    """ 8-way contest, no advantage.
    """
    cdef unsigned int i, t
    global max_generations, genotype_name, small_group_size, group_time, pop_size
    pop_size = 4000
    small_group_size = 4
    group_time = 4
    genotype_name = {(small_group_size, 1, int(1*group_time)):"Small-selfish", (large_group_size, 1,int(1*group_time)):"Large-selfish",
     (small_group_size, 0, int(1*group_time)):"Small-cooperative",(large_group_size, 0, int(1*group_time)):"Large-cooperative",
     (small_group_size, 1, int(0.75*group_time)):"Small-selfish-coalescing", (large_group_size, 1,int(0.75*group_time)):"Large-selfish-coalescing",
     (small_group_size, 0, int(0.75*group_time)):"Small-cooperative-coalescing",(large_group_size, 0, int(0.75*group_time)):"Large-cooperative-coalescing"}
    max_generations = 120
    init_mix()
    resource_results = np.zeros((max_generations, 2))
    genotype_results = dict([(x, [y]) for (x, y) in genotypes()])
    #print genotype_results
    group_size_results = np.zeros((max_generations, 2))
    cdef float resources

    global small_group_size, population, group_time
    cdef str log
    dump_file = open(dump_file_name, 'w')
    dump_file.write("Generation, Genotype, Proportion\n")
    dump_file.close()
    for i in range(max_generations):
        for genotype, count in genotype_results.items():
            count += [0]

        for genotype, count in genotypes():
            if genotype not in genotype_results:
                genotype_results[genotype] = [0] * (i + 1)
            genotype_results[genotype][i] = count

        groups = aggregate()
        new_groups = []
        working_population = []
        returned_groups = {large_group_size:0,small_group_size:0}
        for t in range(group_time ):
                        #print "Gen",i,"Timestep", t
            new_groups = []
                        # Reproduce groups
            for group in groups:
                #print group
                if t < group[0][0][2]:
                                #print "In group"
                    resources = float(group[0][0][0])
                    if resources > small_group_size:
                        resources *= size_advantage
                    new_groups += [reproduce_group(resources, group)]
                elif t == group[0][0][2]:
                                #print "Glomming"
                    returned_groups[group[0][0][0]] += 1
                    working_population = agglomerate(working_population, group)
            groups = new_groups

                        # Reproduce pool
        #print "WP:",working_population
            resource = returned_groups[small_group_size]*small_group_size*pool_advantage + returned_groups[large_group_size]*large_group_size*pool_advantage
            working_population = reproduce_group(resource, working_population)

        for group in groups:
            working_population = agglomerate(working_population, group)

        working_population = rescale(working_population)
        disperse([working_population])
    dump_file = open(dump_file_name, 'a')
    for key, value in genotype_results.items():
        i = 0
        for proportion in value:
            #genotype_name[key]
            log =  "%d, %s, %f\n" % (i, genotype_name[key], proportion)
            i += 1
            if i == max_generations:
                break
            dump_file.write(log)
            print log
    dump_file.close()
    group_time = 4
    small_group_size = 4


@cython.boundscheck(False)
def run_fig_11():
    """ As fig 4, but with reproduction in the group pool and
    time in groups as fraction of turn time at a more granular level
    on proportion for group size 1.
    """
    global small_group_size, population, group_time
    cdef int type_num = pop_size / 2
    cdef int i, t
    cdef str log
    dump_file = open("dump_fig_11.csv", 'w')
    dump_file.write("Size,TotalTime,GTime,Fixated\n")
    dump_file.close()
    # For group sizes 1 - 8
    for small_group_size in range(1,2):
        for total_time in [x*10 for x in range(1, 11)]:
            # For group times from 0.25, 0.5, 0.75 of total time
            for g_time in np.linspace(0.1, 0.99, 10):
                # Initialise the population
                group_time = g_time*total_time
                pop = [(small_group_size, 0, group_time)] * type_num
                pop += [(small_group_size, 1, group_time)] * type_num
                population = np.array(pop, dtype=dtype)
                for i in range(max_generations):
                    groups = aggregate()
                    population = np.array([], dtype=dtype)
                    new_groups = []
                    working_population = [((small_group_size, 0, group_time), 0.), ((small_group_size, 1, group_time), 0.)]
                    for t in range(total_time):
                        print "Gen",i,"Timestep", t
                        new_groups = []
                        # Reproduce groups
                        for group in groups:
                            if t < group[0][0][2]:
                                #print "In group"
                                resources = float(group[0][0][0])
                                if resources > small_group_size:
                                    resources *= size_advantage
                                new_groups += [reproduce_group(resources, group)]
                            elif t == group[0][0][2]:
                                #print "Glomming"
                                working_population = agglomerate(working_population, group)
                        groups = new_groups

                        # Reproduce pool
                        print "WP:",working_population
                        working_population = reproduce_group(pop_size, working_population)

                    disperse([rescale(working_population)])
                    print len(population)
        
                    #rescale()
                    if coop_fix() or selfish_fix():
                        break
                    #print resource_use()
                fixation = coop_fix()
                log =  "%d, %d, %f, %s\n" % (small_group_size, total_time, g_time, fixation)
                if fixation:
                    dump_file = open("dump_fig_11.csv", 'a')
                    dump_file.write(log)
                    dump_file.close()
                print log
    group_time = 4
    small_group_size = 4

def run_fig_12(dump_file_name="dump_fig_12.csv", pool_advantage=1.25):
    """ 1.25x pool advantage. """
    run_fig_10(dump_file_name, pool_advantage)


def all_contest():
    """
    Run the two variations on all-loci contest 50 times each.
    """
    for i in range(50):
        dump_file_name = "dump_fig_10_" + str(i) + ".csv"
        run_fig_10(dump_file_name)
        run_fig_12("dump_fig_12_"+str(i)+".csv")

def __main__():
    run_fig_1()
    run_fig_2()
    run_fig_3()
    run_fig_4()
    run_fig_5()
    run_fig_6()
    run_fig_7()
    run_fig_8()
    run_fig_11()
    all_contest()
