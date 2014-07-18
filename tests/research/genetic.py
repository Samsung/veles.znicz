"""
Created on July 17, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.

"""


import sys
import pylab
import random
import math
import numpy
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt


def fitness(param, dimensions_number):
    """Schwefel's (Sine Root) Function"""
    summation = 0
    for i in param:
        summation += (-i*math.sin(math.sqrt(math.fabs(i))))
    return 100/(418.9829*dimensions_number+summation)


def gray(code_length):
    """Recursive constructing of Gray codes list"""
    if code_length == 2:
        return ["00", "01", "11", "10"]
    else:
        codes = gray(code_length-1)
        codes_size = len(codes)
        for i in range(codes_size):
            codes.append("1"+codes[codes_size-i-1])
            codes[codes_size-i-1] = "0"+codes[codes_size-i-1]
        return codes


class Chromosome(object):
    def __init__(self, n, mx, accuracy, codes, binary=None, numeric=None):
        if n != 0:
            self.size = n
            self.binary = ""
            self.numeric = []
            for _j in range(n):
                rand = random.randint(-mx*accuracy, mx*accuracy)
                self.numeric.append(rand/accuracy)
                if rand > 0:
                    self.binary += ("1"+codes[rand])
                else:
                    self.binary += ("0"+codes[rand])
        else:
            self.numeric = numeric
            self.binary = binary
            self.size = len(numeric)
        self.fitness = fitness(self.numeric, self.size)

    def mutation(self, probability):
        for i in self.binary:
            p_m = numpy.random.rand()
            if p_m > probability:
                if i == "0":
                    i = "1"
                else:
                    i = "0"


class Genetic(object):
    def __init__(self):
        self.outfile = open("out.txt", "w")
        infile = open(sys.argv[1], "r+")
        line = infile.readline()
        self.dim = int(line[line.find(": ")+2:len(line)-1])
        line = infile.readline()
        self.mx = int(line[line.find(": ")+2:len(line)-1])
        line = infile.readline()
        self.pts = int(line[line.find(": ")+2:len(line)-1])
        line = infile.readline()
        self.acc = int(math.pow(10, line.find('1')-line.find('.')))
        line = infile.readline()
        self.func = line[line.find(": ")+2:len(line)-1]
        self.dl = int(math.ceil(math.log2(self.mx*self.acc)))+1
        self.codes = gray(self.dl-1)
        self.chromosomes = []
        self.population_fitness = 0
        for _i in range(self.pts):
            chromo = Chromosome(self.dim, self.mx, self.acc, self.codes)
            self.chromosomes.append(chromo)
            self.population_fitness += chromo.fitness
        self.sort()
        self.cout(0)

    def cout(self, generation_number):
        self.outfile.write("Generation %d\n" % generation_number)
        for i in range(self.pts):
            self.outfile.write("%d.\t%s = [" % (i, self.chromosomes[i].binary))
            for j in range(self.dim-1):
                self.outfile.write("%.2f," % self.chromosomes[i].numeric[j])
            self.outfile.write("%3.2f]\t  F = %.4f\n" %
                               (self.chromosomes[i].numeric[self.dim-1],
                                self.chromosomes[i].fitness))

    def sort(self):
        """Sorting chromosomes by fitness function"""
        for i in range(self.pts-2):
            j = i+1
            while j < self.pts:
                if self.chromosomes[i].fitness < self.chromosomes[j].fitness:
                    temp = self.chromosomes[i]
                    self.chromosomes[i] = self.chromosomes[j]
                    self.chromosomes[j] = temp
                j += 1

    def plot(self):
        """Drawing the plot"""
        if self.dim == 2:
            x = numpy.r_[-self.mx:self.mx:1000j]
            y = numpy.r_[-self.mx:self.mx:1000j]
            X, Y = numpy.meshgrid(x, y)
            Z = []
            n = range(len(x))
            for i in n:
                z_part = []
                for j in n:
                    z_part.append(fitness([x[i], y[j]], 2))
                Z.append(z_part)
            fig = pylab.figure()
            ax = p3.Axes3D(fig)
            ax.plot_surface(X, Y, Z)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('z')
            plt.show()
        elif self.dim == 1:
            x = numpy.r_[-self.mx:self.mx:1000j]
            y = []
            for i in x:
                y.append(fitness([i, ], 1))
            plt.xlabel('x')
            plt.ylabel('y')
            plt.plot(x, y)
            plt.show()
        else:
            print("Plot is available for number of dimensions < 3")

    def selection(self):
        """Selection of chromosomes for crossing"""
        bound = []
        sum_v = 0.0
        for i in range(self.pts-1):
            sum_v += 1.0*self.chromosomes[i].fitness/self.population_fitness
            bound.append(100*sum_v)
        bound.append(100)
        parents = []
        for i in range(self.pts):
            rand = 100*numpy.random.rand()
            j = 0
            while rand > bound[j]:
                j += 1
            parents.append((j, self.chromosomes[j]))
        return parents

    def crossing(self, parents):
        """Genetic operator"""
        cross_num = 0
        self.outfile.write("\nSelected pairs:\n")
        while cross_num < int(self.pts/2):
            rand1 = random.randint(0, self.pts-1)
            parent1 = parents[rand1][1].binary
            rand2 = random.randint(0, self.pts-1)
            parent2 = parents[rand2][1].binary
            l = random.randint(1, len(parent1)-2)
            cut1 = parent1[:l]
            cut2 = parent2[l:]
            cut3 = parent1[l:]
            cut4 = parent2[:l]
            cross1 = cut1+cut2
            cross2 = cut4+cut3
            num1 = []
            num2 = []
            delimiter1 = 0
            delimiter2 = self.dl
            while delimiter1 < len(cross1):
                cut1 = cross1[delimiter1:delimiter2]
                cut2 = cross2[delimiter1:delimiter2]
                """Gray codes to dec numbers"""
                num1.append((self.codes.index(cut1[1:])/self.acc)
                            * math.pow(-1, cut1[0] == '0'))
                num2.append((self.codes.index(cut2[1:])/self.acc)
                            * math.pow(-1, cut2[0] == '0'))
                delimiter1 = delimiter2
                delimiter2 += self.dl
            chromo_son1 = Chromosome(0, self.mx, self.acc,
                                     self.codes, cross1, num1)
            chromo_son2 = Chromosome(0, self.mx, self.acc,
                                     self.codes, cross2, num2)
            if chromo_son1.fitness > self.chromosomes[self.pts-1].fitness:
                j = 0
                while self.chromosomes[j].fitness > chromo_son1.fitness:
                    j += 1
                self.chromosomes.insert(j, chromo_son1)
            if chromo_son2.fitness > self.chromosomes[self.pts-1].fitness:
                j = 0
                while self.chromosomes[j].fitness > chromo_son2.fitness:
                    j += 1
                self.chromosomes.insert(j, chromo_son2)
            self.chromosomes = self.chromosomes[:self.pts]
            cross_num += 1

    def mutation(self):
        """Mutation of all chromosomes in population"""
        for chromo in self.chromosomes:
            chromo.mutation(0.9)

    def evolution(self):
        generation_num = 0
        while generation_num < 100:
            print("Generation %d" % generation_num)
            chromo = self.selection()
            self.crossing(chromo)
            self.mutation()
            generation_num += 1
            self.cout(generation_num)

if __name__ == "__main__":
    sys.exit(Genetic().evolution())
