"""
Created on July 17, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.

"""


import sys
import pylab
import random
import math
import numpy
import time
import copy
import pickle
import os
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from veles.config import root
from multiprocessing import Pool


def defaults():
    root.defaults = {"population": {"chromosomes": 10},
                     "selection": {"roulette": {"select_size": 50},
                                   "use": "roulette"},
                     "crossing": {"pointed": {"points": 1,
                                              "probability": 1.0,
                                              "crossings": 1}},
                     "mutation": {"binary_point": {"use": "y",
                                                   "points": 30,
                                                   "probability": 0.5}},
                     "optimization": {"dimensions": 2,
                                      "accuracy": 0.1,
                                      "min_x": 0,
                                      "max_x": 100,
                                      "choice": "betw",
                                      "function": "schwefel",
                                      "extremum": min},
                     "stopping": {"epoch": 25}
                     }


def type_determ(val):
    if val.isdigit() is True or (val[1:].isdigit() is True and val[0] == "-"):
        return int(val)
    elif "." in val:
        return float(val)
    else:
        return val


def set_config(filename):
    defaults()
    infile = open(filename, "r+")
    line = infile.readline()
    deeper_node = False
    _type = ""
    while "parameter" in line:
        param_group = line.split(" ")[0]
        line = infile.readline()
        while ":" in line:
            (param_name, value) = line.split(":")
            param_name = clear(param_name)
            value = clear(value)
            if param_name in ("cross_type", "select_type", "mut_type"):
                deeper_node = True
                _type = value
            if "[" in line:
                value = value[1:len(value) - 1]
                _list = []
                while value != "":
                    if "," in value:
                        ind = value.index(",")
                        _list.append(type_determ(value[0:ind]))
                        value = value[ind + 1:]
                    else:
                        _list.append(type_determ(value))
                        value = ""
                value = _list
            else:
                value = type_determ(value)
            if not deeper_node:
                root.update = {param_group: {param_name: value}}
            elif param_name not in ("cross_type", "select_type", "mut_type"):
                root.update = {param_group: {_type: {param_name: value}}}
            if((param_name in ["probability", "select_size"]) and deeper_node):
                deeper_node = False
            line = infile.readline()
            while "#" in line:
                line = infile.readline()
        line = infile.readline()
    min_x = root.optimization.min_x
    max_x = root.optimization.max_x
    choice = root.optimization.choice
    if (type(min_x) == int or type(min_x) == float):
        root.optimization.min_x = [min_x for _i
                                   in range(root.optimization.dimensions)]
    if (type(max_x) == int or type(max_x) == float):
        root.optimization.max_x = [max_x for _i
                                   in range(root.optimization.dimensions)]
    if (type(choice) == str):
        root.optimization.choice = [choice for _i
                                    in range(root.optimization.dimensions)]
    while len(root.optimization.choice) < root.optimization.dimensions:
        root.optimization.choice.append("betw")


def gray(code_length):
    """Recursive constructing of Gray codes list"""
    if code_length == 2:
        return ["00", "01", "11", "10"]
    else:
        codes = gray(code_length - 1)
        codes_size = len(codes)
        for i in range(codes_size):
            codes.append("1" + codes[codes_size - i - 1])
            codes[codes_size - i - 1] = "0" + codes[codes_size - i - 1]
        return codes


def clear(string):
    clear_str = ""
    for letter in string:
        if letter not in ("\t", "\n", " ", ":"):
            clear_str += letter
    return clear_str


def fitness(param):
        """Schwefel's (Sine Root) Function"""
        dimensions_number = len(param)
        summation = 0
        for i in param:
            summation += (-i * math.sin(math.sqrt(math.fabs(i))))
        return 1 / (418.9829 * dimensions_number + summation)


def bin_to_num(binaries, dl, codes):
    """Convert gray codes of chromosomes to arrays of floats"""
    num = ([], [])
    delimiter1 = 0
    delimiter2 = dl
    chromo_length = len(binaries[0])
    binaries_num = len(binaries)
    while delimiter1 < chromo_length:
        for i in range(binaries_num):
            cut = binaries[i][delimiter1:delimiter2]
            """Gray codes to dec numbers"""
            num[i].append((codes.index(cut[1:])
                           * root.optimization.accuracy)
                          * math.pow(-1, cut[0] == '0'))
        delimiter1 = delimiter2
        delimiter2 += dl
    return num


def num_to_bin(numbers, codes):
    """Convert float numbers to gray codes"""
    binary = ""
    for i in range(len(numbers)):
        if numbers[i] > 0:
            binary += "1"
        else:
            binary += "0"
        binary += codes[int(math.fabs(numbers[i] /
                                      root.optimization.accuracy))]
    return binary


class Chromosome(object):
    def __init__(self, n, min_x, max_x, accuracy, codes,
                 binary=None, numeric=None, *args):
        if n != 0:
            self.size = n
            self.binary = ""
            self.numeric = []
            for j in range(n):
                if root.optimization.choice[j] == "or":
                    rand = numpy.random.choice([min_x[j], max_x[j]])
                    self.numeric.append(rand)
                elif type(min_x[j]) == float or type(max_x[j]) == float:
                    rand = random.randint(int(min_x[j] * accuracy),
                                          int(max_x[j] * accuracy))
                    self.numeric.append(rand / accuracy)
                else:
                    rand = random.randint(min_x[j], max_x[j])
                    self.numeric.append(rand)
                    rand = int(rand * accuracy)
                if root.optimization.code == "gray":
                    if rand > 0:
                        self.binary += ("1" + codes[rand])
                    else:
                        self.binary += ("0" + codes[rand])
        else:
            self.numeric = numeric
            self.numeric_correct()
            self.binary = binary
            self.size = len(numeric)
        self.fitness = self.fit(*args)

    def fit(self, *args):
        """Schwefel's (Sine Root) Function"""
        if len(self.numeric) == 0:
            return 0
        else:
            dimensions_number = len(self.numeric)
            summation = 0
            for i in self.numeric:
                summation += (-i * math.sin(math.sqrt(math.fabs(i))))
            return 1 / (418.9829 * dimensions_number + summation)

    def numeric_correct(self):
        for pos in range(len(self.numeric)):
            max_x = root.optimization.max_x[pos]
            min_x = root.optimization.min_x[pos]
            diff = max_x - min_x
            while not (self.numeric[pos] <= max_x and
                       self.numeric[pos] >= min_x):
                if self.numeric[pos] > max_x:
                    self.numeric[pos] -= diff
                elif self.numeric[pos] < min_x:
                    self.numeric[pos] += diff

    def mutation_binary_point(self, points, probability):
        """changes 0 to 1 and 1 to 0"""
        mutant = ""
        for _i in range(points):
            pos = random.randint(1, len(self.binary) - 2)
            p_m = numpy.random.rand()
            if p_m < probability:
                if self.binary[pos] == "0":
                    mutant = self.binary[:pos] + "1" + self.binary[pos + 1:]
                else:
                    mutant = self.binary[:pos] + "0" + self.binary[pos + 1:]
            else:
                mutant = self.binary
        self.binary = mutant

    def mutation_altering(self, points, probability):
        """changing positions of two floats"""
        if root.optimization.code == "gray":
            mutant = ""
            for _i in range(points):
                pos1 = numpy.random.randint(0, len(self.binary) - 1)
                pos2 = numpy.random.randint(0, len(self.binary) - 1)
                p_m = numpy.random.rand()
                if p_m < probability:
                    if pos1 < pos2:
                        mutant = (self.binary[:pos1] + self.binary[pos1] +
                                  self.binary[pos1 + 1:pos2] +
                                  self.binary[pos2] + self.binary[pos2 + 1:])
                    else:
                        mutant = (self.binary[:pos2] + self.binary[pos2] +
                                  self.binary[pos2 + 1:pos1] +
                                  self.binary[pos1] + self.binary[pos1 + 1:])
                else:
                    mutant = self.binary
            self.binary = mutant
        else:
            for _i in range(points):
                pos1 = numpy.random.randint(0, len(self.numeric) - 1)
                pos2 = numpy.random.randint(0, len(self.numeric) - 1)
                p_m = numpy.random.rand()
                if p_m < probability:
                    temp = self.numeric[pos1]
                    self.numeric[pos1] = self.numeric[pos2]
                    self.numeric[pos2] = temp

    def mutation_gaussian(self, points, probability):
        """adding random gaussian number"""
        min_x = root.optimization.min_x
        max_x = root.optimization.max_x
        mut_pool = [i for i in range(len(self.numeric))]
        for _i in range(points):
            pos = numpy.random.choice(mut_pool)
            if root.optimization.choice == "or":
                self.numeric[pos] = numpy.random.choice[min_x[pos], max_x[pos]]
            else:
                isint = (type(self.numeric[pos]) == int)
                diff = max_x[pos] - min_x[pos]
                max_prob = min_x[pos] + diff / 2
                gauss = random.gauss(max_prob, math.sqrt(diff / 6))
                p_m = numpy.random.rand()
                if p_m < probability:
                    if numpy.random.random() < 0.5:
                        self.numeric[pos] -= gauss
                    else:
                        self.numeric[pos] += gauss
                    # Bringing numeric[pos] to its limits
                    while (self.numeric[pos] < min_x[pos] or
                           self.numeric[pos] > max_x[pos]):
                        if self.numeric[pos] < min_x[pos]:
                            self.numeric[pos] += diff
                        else:
                            self.numeric[pos] -= diff
                    if isint:
                        self.numeric[pos] = int(self.numeric[pos])
                mut_pool.remove(pos)
                if len(mut_pool) == 0:
                    break
        self.fitness = self.fit(500)

    def mutation_uniform(self, points, probability):
        """replacing float number with another random number"""
        min_x = root.optimization.min_x
        max_x = root.optimization.max_x
        mut_pool = [i for i in range(len(self.numeric))]
        for _i in range(points):
            pos = numpy.random.choice(mut_pool)
            if root.optimization.choice == "or":
                self.numeric[pos] = numpy.random.choice[min_x[pos], max_x[pos]]
            else:
                isint = (type(self.numeric[pos]) == int)
                p_m = numpy.random.rand()
                if p_m < probability:
                    rand = numpy.random.uniform(min_x[pos], max_x[pos])
                    if isint:
                        rand = int(rand)
                    self.numeric[pos] = rand
                mut_pool.remove(pos)
                if len(mut_pool) == 0:
                    break
        self.fitness = self.fit(500)


class Genetic(object):
    def __init__(self):
        """Creating the population"""
        # Reading parameters from input file
        set_config(sys.argv[1])
        self.outfile = open("out.txt", "w")
        max_abs_x = 0
        for i in range(root.optimization.dimensions):
            if math.fabs(root.optimization.min_x[i]) > max_abs_x:
                max_abs_x = math.fabs(root.optimization.min_x[i])
            if math.fabs(root.optimization.max_x[i]) > max_abs_x:
                max_abs_x = math.fabs(root.optimization.max_x[i])
        max_coded_int = int(max_abs_x / root.optimization.accuracy)
        # Length of code of one int number
        self.dl = int(math.ceil(math.log2(max_coded_int)))
        self.codes = gray(self.dl)
        # +1 symbol 1/0 for positive/negative
        self.dl += 1
        self.chromosomes = (pickle.load(open("population.p", "rb"))
                            if os.path.exists("population.p") else
                            [])
        self.population_fitness = 0
        for _i in range(root.population.chromosomes - len(self.chromosomes)):
            print("Creating chromo #%d" % _i)
            chromo = self.new_chromo(root.optimization.dimensions,
                                     root.optimization.min_x,
                                     root.optimization.max_x,
                                     1 / root.optimization.accuracy,
                                     self.codes)
            self.add(chromo)
            self.population_fitness += chromo.fitness
        self.cout(0)
#         pickle.dump(self.chromosomes, open("population.p", "wb"))

    def new_chromo(self, n, min_x, max_x, accuracy, codes,
                   binary=None, numeric=None, *args):
        chromo = Chromosome(n, min_x, max_x, accuracy,
                            codes, binary, numeric, *args)
        return chromo

    def add(self, chromo):
        """Adding new chromosome to the population"""
        chromo_num = len(self.chromosomes)
        if chromo_num == 0:
            self.chromosomes.append(chromo)
            return
        elif (chromo_num < root.population.chromosomes or
              chromo.fitness > self.chromosomes[chromo_num - 1].fitness):
            j = 0
            while (j < chromo_num and
                   self.chromosomes[j].fitness > chromo.fitness):
                j += 1
            self.chromosomes.insert(j, chromo)
            self.chromosomes = self.chromosomes[:root.population.chromosomes]

    def cout(self, epoch_number):
        """Printning info about population"""
        self.outfile.write("Epoch #%d:\n" % epoch_number)
        len_num = root.optimization.dimensions
        for i in range(root.population.chromosomes):
            self.outfile.write("\t [")
            for j in range(len_num - 1):
                if math.fabs(self.chromosomes[i].numeric[j]) < 100:
                    self.outfile.write(" ")
                if math.fabs(self.chromosomes[i].numeric[j]) < 10:
                    self.outfile.write(" ")
                if self.chromosomes[i].numeric[j] > 0:
                    self.outfile.write(" ")
                self.outfile.write("%4.3f, " % self.chromosomes[i].numeric[j])
            if math.fabs(self.chromosomes[i].numeric[len_num - 1]) < 100:
                self.outfile.write(" ")
            if math.fabs(self.chromosomes[i].numeric[len_num - 1]) < 10:
                self.outfile.write(" ")
            if self.chromosomes[i].numeric[len_num - 1] > 0:
                self.outfile.write(" ")
            self.outfile.write("%4.3f ]\t  F = %.5f\n" %
                               (self.chromosomes[i].numeric[len_num - 1],
                                self.chromosomes[i].fitness))
        self.outfile.write("\n")

    def sort(self):
        """Sorting chromosomes by fitness function"""
        for i in range(root.population.chromosomes - 2):
            j = i + 1
            while j < root.population.chromosomes:
                if self.chromosomes[i].fitness < self.chromosomes[j].fitness:
                    temp = self.chromosomes[i]
                    self.chromosomes[i] = self.chromosomes[j]
                    self.chromosomes[j] = temp
                j += 1

    def function_plot(self):
        """Drawing the plot of the function"""
        if root.optimization.dimensions == 2:
            x = numpy.r_[-self.max_X:self.max_X:1000j]
            y = numpy.r_[-self.max_X:self.max_X:1000j]
            X, Y = numpy.meshgrid(x, y)
            Z = []
            n = range(len(x))
            for i in n:
                z_part = []
                for j in n:
                    z_part.append(fitness([x[i], y[j]]))
                Z.append(z_part)
            fig = pylab.figure()
            ax = p3.Axes3D(fig)
            ax.plot_surface(X, Y, Z)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('z')
            plt.show()
        elif root.optimization.dimensions == 1:
            x = numpy.r_[-self.max_X:self.max_X:1000j]
            y = []
            for i in x:
                y.append(fitness([i, ], 1))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(x, y)
            plt.show()
        else:
            print("Plot of the function"
                  " is available for number of dimensions < 3")

    def selection_roulette(self):
        """selection of chromosomes for crossing with roulette"""
        bound = []
        sum_v = 0.0
        for i in range(root.population.chromosomes):
            sum_v += 1.0 * self.chromosomes[i].fitness / \
                self.population_fitness
            bound.append(100 * sum_v)
        bound.append(100)
        parents = []
        for i in range(root.selection.roulette.select_size):
            rand = 100 * numpy.random.rand()
            j = 0
            while rand > bound[j]:
                j += 1
            parents.append(self.chromosomes[j])
        return parents

    def selection_random(self):
        """random selection of chromosomes for crossing"""
        parents = []
        for _i in range(root.selection.random.select_size):
            rand = numpy.random.randint(0, root.population.chromosomes - 1)
            parents.append(self.chromosomes[rand])
        return parents

    def selection_tournament(self):
        """tournament selection of chromosomes for crossing"""
        tournament_pool = []
        for _i in range(root.selection.tournament.tournament_size):
            rand = numpy.random.randint(0, root.population.chromosomes - 1)
            j = 0
            while (j != len(tournament_pool) and
                   tournament_pool[j] < self.chromosomes[rand]):
                    j += 1
            tournament_pool.insert(j, self.chromosomes[rand])
        return tournament_pool[:root.selection.tournament.select_size]

    def crossing_pointed(self, parents):
        """Genetic operator"""
        cross_num = 0
        while cross_num < root.crossing.pointed.crossings:
            print("\tcross #%d" % cross_num)
            rand1 = random.randint(0, len(parents) - 1)
            parent1 = parents[rand1].binary
            rand2 = random.randint(0, len(parents) - 1)
            parent2 = parents[rand2].binary
            cross_points = [0, ]
            l = 0
            for _i in range(root.crossing.pointed.points):
                while l in cross_points:
                    l = random.randint(1, len(parent1) - 2)
                j = 0
                while j != len(cross_points) and cross_points[j] < l:
                    j += 1
                cross_points.insert(j, l)
            cross_points.append(len(parent1))
            cross1 = ""
            cross2 = ""
            i = 1
            while i <= root.crossing.pointed.points + 1:
                if i % 2 == 0:
                    cross1 += parent1[cross_points[i - 1]:cross_points[i]]
                    cross2 += parent2[cross_points[i - 1]:cross_points[i]]
                else:
                    cross1 += parent2[cross_points[i - 1]:cross_points[i]]
                    cross2 += parent1[cross_points[i - 1]:cross_points[i]]
                i += 1
            (num1, num2) = bin_to_num([cross1, cross2], self.dl, self.codes)
            chromo_son1 = self.new_chromo(0, root.optimization.min_x,
                                          root.optimization.max_x,
                                          1 / root.optimization.accuracy,
                                          self.codes, cross1, num1)
            chromo_son2 = self.new_chromo(0, root.optimization.min_x,
                                          root.optimization.max_x,
                                          1 / root.optimization.accuracy,
                                          self.codes, cross2, num2)
            chromo_son1.size = len(chromo_son1.numeric)
            chromo_son2.size = len(chromo_son2.numeric)
            self.add(chromo_son1)
            self.add(chromo_son2)
            cross_num += 1
        self.population_fitness = 0

    def crossing_uniform(self, parents):
        cross_num = 0
        while cross_num < root.crossing.uniform.crossings:
            if root.optimization.code == "gray":
                rand1 = random.randint(0, len(parents) - 1)
                parent1 = parents[rand1].binary
                rand2 = random.randint(0, len(parents) - 1)
                parent2 = parents[rand2].binary
                cross = ""
                for i in range(len(parent1)):
                    rand = numpy.random.uniform(0, 2)
                    if rand < 1:
                        cross += parent1[i]
                    else:
                        cross += parent2[i]
                numeric = bin_to_num([cross], self.dl, self.codes)[0]
                chromo_son = self.new_chromo(0, root.optimization.min_x,
                                             root.optimization.max_x,
                                             1 / root.optimization.accuracy,
                                             self.codes, cross, numeric)
            else:
                rand1 = random.randint(0, len(parents) - 1)
                parent1 = parents[rand1].numeric
                rand2 = random.randint(0, len(parents) - 1)
                parent2 = parents[rand2].numeric
                cross = []
                for i in range(len(parent1)):
                    rand = numpy.random.uniform(0, 2)
                    if rand < 1:
                        cross.append(parent1[i])
                    else:
                        cross.append(parent2[i])
                chromo_son = self.new_chromo(0, root.optimization.min_x,
                                             root.optimization.max_x,
                                             1 / root.optimization.accuracy,
                                             self.codes, None, cross)
            print("\tcrossing uniform #%d fitness = %.2f" %
                  (cross_num, chromo_son.fitness))
            self.add(chromo_son)
            cross_num += 1
        self.population_fitness = 0

    def crossing_arithmetic(self, parents):
        """Arithmetical crossingover"""
        cross_num = 0
#         p = Pool(2)
        while cross_num < root.crossing.arithmetic.crossings:
            rand1 = numpy.random.randint(0, len(parents))
            parent1 = parents[rand1].numeric
            rand2 = numpy.random.randint(0, len(parents))
            parent2 = parents[rand2].numeric
            cross1 = []
            cross2 = []
            for i in range(len(parent1)):
                a = numpy.random.random()
                if root.optimization.choice == "or":
                    if a > 0.5:
                        cross1.append(parent1[i])
                        cross2.append(parent2[i])
                    else:
                        cross1.append(parent2[i])
                        cross2.append(parent1[i])
                elif type(parent1[i]) == int:
                    k = int(a * parent1[i] + (1 - a) * parent2[i])
                    cross1.append(k)
                    cross2.append(parent1[i] + parent2[i] - k)
                else:
                    cross1.append(a * parent1[i] + (1 - a) * parent2[i])
                    cross2.append((1 - a) * parent1[i] + a * parent2[i])
            if root.optimization.code == "gray":
                (bin1, bin2) = (num_to_bin(cross1, self.codes),
                                num_to_bin(cross2, self.codes))
            else:
                (bin1, bin2) = ("", "")
            chromo1 = self.new_chromo(
                0, root.optimization.min_x, root.optimization.max_x,
                1 / root.optimization.accuracy, self.codes, bin1, cross1)
            chromo2 = self.new_chromo(
                0, root.optimization.min_x, root.optimization.max_x,
                1 / root.optimization.accuracy, self.codes, bin2, cross2)
            self.add(chromo1)
            self.add(chromo2)
            print("\tcrossing arithmetical #%d fitness = %.2f and %.2f"
                  % (cross_num, chromo1.fitness, chromo2.fitness))
            cross_num += 1

    def crossing_geometric(self, parents):
        """Geometrical crossingover"""
        cross_num = 0
        while cross_num < root.crossing.geometric.crossings:
            cross = []
            rand1 = numpy.random.randint(0, len(parents))
            parent1 = parents[rand1].numeric
            rand2 = numpy.random.randint(0, len(parents))
            parent2 = parents[rand2].numeric
            for i in range(len(parent1)):
                if root.optimization.choice == "or":
                    if numpy.random.random() > 0.5:
                        cross.append(parent1[i])
                    else:
                        cross.append(parent2[i])
                else:
                    # correct1 is used to invert [-x1; -x2] to [x2; x1]
                    correct1 = -1 if root.optimization.max_x[i] < 0 else 1
                    # correct2 is used to alter [-x1; x2] to [0; x2+x1]
                    if root.optimization.min_x[i] > 0 or correct1 == -1:
                        correct2 = 0
                    else:
                        correct2 = -root.optimization.min_x[i]
                    a = numpy.random.rand()
                    gene = (correct1 * (math.pow(
                        correct1 * parent1[i] + correct2, a) * math.pow(
                        correct1 * parent2[i] + correct2, (1 - a)) - correct2))
                    if type(parent1[i]) == int:
                        gene = int(gene)
                    cross.append(gene)
            binary = ""
            if root.optimization.code == "gray":
                binary = num_to_bin(cross, self.codes)
            chromo_son = self.new_chromo(0, root.optimization.min_x,
                                         root.optimization.max_x,
                                         1 / root.optimization.accuracy,
                                         self.codes, binary, cross, 10)
            self.add(chromo_son)
            print("\tcrossing geometric #%d fitness = %.2f"
                  % (cross_num, chromo_son.fitness))
            cross_num += 1

    def stop(self, epoch, med_best_diff, time_diff):
        if(root.stopping.median_best_diff != "no" and
           root.stopping.median_best_diff > med_best_diff):
            return True
        if(root.stopping.epoch == epoch):
            return True
        if(root.stopping.time != "no" and root.stopping.time < time_diff):
            return True
        return False

    def evolution(self):
        n = root.population.chromosomes
        epoch_num = 0
        best_fit = []
        worst_fit = []
        average_fit = []
        median = []
        fig = plt.figure()
        fig.show()
        ax = fig.add_subplot(111)
        fig.canvas.set_window_title("f = fitness(epoch number)")
        plt.xlabel('Epoch number')
        plt.ylabel('Fitness function')
#         ax.set_yscale('log')
        plt.figtext(0.6, 0.8, "n = %d" % root.optimization.dimensions)
        _cross_param = root.__getattribute__("crossing").__dict__
        _mut_param = root.__getattribute__("mutation").__dict__
        _select_type = root.__getattribute__("selection").__dict__["use"]
        while True:
            start_time = time.time()
            print("Epoch #%d" % epoch_num)
            self.cout(epoch_num)
            self.population_fitness = 0
            for i in self.chromosomes:
                self.population_fitness += i.fitness
            average_fit.append(self.population_fitness
                               / root.population.chromosomes)
            best_fit.append(self.chromosomes[0].fitness)
            worst_fit.append(self.chromosomes[n - 1].fitness)
            median.append(self.chromosomes[int(n / 2)].fitness)
            ax.plot(worst_fit, "-b", label="worst")
#             ax.plot(average_fit, "-k", label="average")
#             ax.plot(median, "-g", label="median")
            ax.plot(best_fit, "-r", label="best")
            if epoch_num == 0:
                ax.legend(loc="upper right")
            plt.draw()
            stop = self.stop(epoch_num,
                             best_fit[epoch_num] - median[epoch_num],
                             time.time() - start_time)
            if stop:
                break
            if(median[epoch_num] < median[epoch_num - 1] or
               best_fit[epoch_num] < best_fit[epoch_num - 1]):
                print("epoch %d" % epoch_num)
                print("median - best = %.2f - %.2f = %.2f" %
                      (best_fit[epoch_num], median[epoch_num],
                       best_fit[epoch_num] - median[epoch_num]))
            # Selection witn chosen method
            chromo = getattr(self, "selection_" + _select_type)()
            # Crossing with all chosen methods
            for (k, v) in _cross_param.items():
                if v.__dict__["use"] == "y":
                    getattr(self, "crossing_" + k)(chromo)
            # Mutation with all chosen methods
            for (k, v) in _mut_param.items():
                _dict = v.__dict__
                if _dict["use"] == "y":
                    mut_pool = [i for i in range(n)]
                    mutants = 0
                    while (mutants < _dict["chromosomes"] and
                           len(mut_pool) > 0):
                        rand = numpy.random.choice(mut_pool)
                        mutating = copy.deepcopy(self.chromosomes[rand])
                        getattr(mutating,
                                "mutation_" + k)(_dict["points"],
                                                 _dict["probability"])
                        if root.optimization.code == "gray":
                            mutating.numeric = bin_to_num([mutating.binary],
                                                          self.dl,
                                                          self.codes)[0]
                            mutating.fitness = mutating.fit(mutating.numeric)
                        self.add(mutating)
                        print("\tmutation %s #%d fitness = %.2f" %
                              (k, mutants, mutating.fitness))
#                         print(mutating.numeric)
                        mut_pool.remove(rand)
                        mutants += 1
            print("\ttime for epoch #%d = %.2f" %
                  (epoch_num, time.time() - start_time))
            print("the best chromosome is", end=" ")
            print(self.chromosomes[0].numeric)
            print("its fitness = %.2f" % self.chromosomes[0].fitness)
            pickle.dump(self.chromosomes, open("population500.p", "wb"))
            plt.savefig("plot500.png")
            epoch_num += 1

if __name__ == "__main__":
    sys.exit(Genetic().evolution())
