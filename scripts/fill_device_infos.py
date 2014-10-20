#!/usr/bin/python3
"""
Created on Oct 8, 2014

Fills device_infos.json for the selected OpenCL device.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import argparse
import json
import logging
from multiprocessing import Process
import numpy
import opencl4py
import os
import pickle
import sys
import time
import traceback

from veles.config import root
from veles.distributable import Pickleable
from veles.formats import Vector
from veles.genetics import Chromosome, Population
from veles.logger import Logger
from veles.opencl import Device, DeviceInfo
import veles.opencl_types as opencl_types
import veles.prng as prng
from veles.tests import DummyWorkflow
from veles.znicz.all2all import All2All
from veles.znicz.gd import GradientDescent
from veles.znicz.conv import Conv
from veles.znicz.gd_conv import GradientDescentConv


class InProcessRun(Pickleable):
    def init_unpickled(self):
        super(InProcessRun, self).init_unpickled()
        self.create_pipe()

    def create_pipe(self):
        pread, pwrite = os.pipe()
        self.pread_ = os.fdopen(pread, "rb")
        self.pwrite_ = os.fdopen(pwrite, "wb")

    def run_in_process(self, target, *target_args):
        try:
            p = Process(target=self.process_target, args=(target, target_args))
            p.start()
            retval = pickle.load(self.pread_)
            p.join()
        except KeyboardInterrupt:
            p.terminate()
            raise
        return retval

    def process_target(self, target, target_args):
        retval = target(*target_args)
        pickle.dump(retval, self.pwrite_)
        self.pwrite_.flush()


class BenchmarkChromosome(Chromosome):
    def __init__(self, **kwargs):
        super(BenchmarkChromosome, self).__init__(**kwargs)
        self.max_wgs = kwargs["max_wgs"]
        self.max_localmem = kwargs["max_localmem"]
        self.itemsize = kwargs["itemsize"]
        self.population_ = None

    def evaluate(self):
        self.make_valid()
        key = tuple(self.numeric)
        vle = self.population_.computed_times.get(key)
        if vle is not None:
            self.warning("Already computed: %s => %.3f", str(key), vle)
            self.fitness = self.time_to_fitness(vle)
            return
        vle = self.population_.evaluate_in_process(
            self.numeric[0], self.numeric[1], self.numeric[2])
        self.population_.computed_times[key] = vle
        self.fitness = self.time_to_fitness(vle)

    @property
    def valid(self):
        a = self.numeric[0]
        b = self.numeric[1]
        c = self.numeric[2]
        return (a * b < self.max_wgs and
                    (a + b) * c * self.itemsize <= self.max_localmem)

    def make_valid(self):
        a = self.numeric[0]
        b = self.numeric[1]
        c = self.numeric[2]
        while a * b > self.max_wgs:
            if a > b:
                a = numpy.random.randint(2, a)
            else:
                b = numpy.random.randint(2, b)
        while (a + b) * c * self.itemsize > self.max_localmem:
            if a > b:
                if a > c:
                    a = numpy.random.randint(2, a)
                else:
                    c = numpy.random.randint(2, c)
            else:
                if b > c:
                    b = numpy.random.randint(2, b)
                else:
                    c = numpy.random.randint(2, c)
        if (a, b, c) != (self.numeric[0], self.numeric[1], self.numeric[2]):
            self.info("Adjusted parameters: %s => %s", str(self.numeric),
                      str((a, b, c)))
            self.numeric[0] = a
            self.numeric[1] = b
            self.numeric[2] = c

    def time_to_fitness(self, vle):
        return 100.0 / (1.0 + vle) if vle >= 0 else 0

    def fitness_to_time(self, vle):
        return 100.0 / vle - 1.0 if vle > 0 else 1000000000.0


class BenchmarkPopulation(Population):
    def __init__(self, **kwargs):
        self.filler = kwargs["filler"]
        self.computed_times = kwargs["computed_times"]
        self.evaluation_routine = kwargs["evaluation_routine"]
        self.evaluation_routine_args = kwargs["evaluation_routine_args"]
        max_ab = min(self.filler.max_wgs, 1024)
        self.itemsize = numpy.zeros(1, dtype=self.filler.dtype).itemsize
        max_c = min(self.filler.max_localmem // 2 // self.itemsize, 1024)
        super(BenchmarkPopulation, self).__init__(
            BenchmarkChromosome, optimization_size=3,
            min_values=[2, 2, 2], max_values=[max_ab, max_ab, max_c],
            population_size=70, accuracy=1.0 / 3.0)
        self.process = None
        self.run_count = 0
        pread, pwrite = os.pipe()
        self.out_pread = os.fdopen(pread, "rb")
        self.out_pwrite = os.fdopen(pwrite, "wb")
        pread, pwrite = os.pipe()
        self.in_pread = os.fdopen(pread, "rb")
        self.in_pwrite = os.fdopen(pwrite, "wb")

    def evaluate_in_process(self, a, b, c):
        try:
            if self.run_count <= 0:
                self.run_count = 100
                self.process = Process(target=self.evaluate_from_pipe)
                self.process.start()
            self.run_count -= 1
            pickle.dump((a, b, c), self.out_pwrite)
            self.out_pwrite.flush()
            vle = pickle.load(self.in_pread)
        except KeyboardInterrupt:
            if self.process is not None:
                self.process.terminate()
                self.process = None
            raise
        return vle

    def evaluate_from_pipe(self):
        try:
            while self.run_count > 0:
                self.run_count -= 1
                a, b, c = pickle.load(self.out_pread)
                vle = self.evaluation_routine(a, b, c,
                                              *self.evaluation_routine_args)
                pickle.dump(vle, self.in_pwrite)
                self.in_pwrite.flush()
        except Exception as e:
            self.error("evaluate_from_pipe() failed: %s\n%s", str(e),
                       traceback.format_exc())
            os._exit(1)

    def evaluate(self, callback):
        for chromo in self.chromosomes:
            chromo.population_ = self
        super(BenchmarkPopulation, self).evaluate(callback)

    def new(self, size, minvles, maxvles, accuracy, codes, binary=None,
            numeric=None):
        kv = {k: v for k, v in locals().items() if k != "self"}
        kv["max_wgs"] = self.filler.max_wgs
        kv["max_localmem"] = self.filler.max_localmem
        kv["itemsize"] = self.itemsize
        return BenchmarkChromosome(**kv)

    def on_after_evolution_step(self):
        chromo = self.chromosomes[0]
        self.info("Best configuration is: %s => %.6f",
                  str(tuple(chromo.numeric)),
                  chromo.fitness_to_time(chromo.fitness))
        n = 0
        for i in range(1, len(self.chromosomes)):
            if self.chromosomes[i].fitness is None:
                break
            n = i
        chromo = self.chromosomes[n >> 1]
        self.info("Median configuration is: %s => %.6f",
                  str(tuple(chromo.numeric)),
                  chromo.fitness_to_time(chromo.fitness))
        chromo = self.chromosomes[n]
        self.info("Worst configuration is: %s => %.6f",
                  str(tuple(chromo.numeric)),
                  chromo.fitness_to_time(chromo.fitness))
        return super(BenchmarkPopulation, self).on_after_evolution_step()

    def nothing(self):
        pass

    def evolve(self):
        while not self.generation or not self.on_after_evolution_step():
            self.do_evolution_step(self.nothing)
            self.generation += 1
        if self.process is not None:
            self.process.terminate()
            self.process = None


class DummyUnit(object):
    def __init__(self, device=None):
        self.device = device


class DeviceInfosFiller(InProcessRun, DeviceInfo, Logger):
    """Runs the benchmark to determine the optimal block sizes.
    """
    def __init__(self, **kwargs):
        self.a_block_size = 16
        self.b_block_size = 16
        self.common_block_size = 16

        self.device = None
        self.dtype = None
        self.device_info = None
        self.forward = None
        self.backward = None
        self.computed_times = {}
        self.device_infos = None

        self.create_pipe()
        di, max_wgs, max_localmem = self.run_in_process(self.get_info)
        kwargs["desc"] = di.desc
        kwargs["memsize"] = di.memsize
        kwargs["memalign"] = di.memalign
        kwargs["version"] = di.version
        super(DeviceInfosFiller, self).__init__(**kwargs)
        self.max_wgs = max_wgs
        self.max_localmem = max_localmem

    def do_the_job(self):
        self.device_infos = {}
        fnme = os.path.join(root.common.device_dir, "device_infos.json")
        try:
            with open(fnme, "r") as fin:
                self.device_infos = json.load(fin)
        except IOError:
            self.warning("%s was not found", fnme)

        self.conduct_tests()

    def get_info(self):
        self.device = Device()
        di = self.device.device_info
        max_wgs = self.device.queue_.device.max_work_group_size
        max_localmem = self.device.queue_.device.local_memsize
        return di, max_wgs, max_localmem

    def get_block_sizes(self, **kwargs):
        return self.a_block_size, self.b_block_size, self.common_block_size

    def conduct_tests(self):
        self.device_info = self.device_infos.get(self.desc, {})
        for ocl_dtype, dtype in sorted(opencl_types.dtypes.items()):
            for precision_level in (0, 1, 2):
                self.info("dtype is %s, precision level is %d",
                          ocl_dtype, precision_level)
                root.common.precision_type = ocl_dtype
                root.common.precision_level = precision_level
                self.dtype = dtype
                self.test_all2all(False)
                self.test_all2all(True)
                self.test_gd()
                self.test_conv(False)
                self.test_conv(True)
                self.test_deconv(False)
                self.test_deconv(True)
                self.test_gd_conv()

    def check_info(self, krnnme, access_type):
        krninfo = self.device_info.get(krnnme)
        if krninfo is None:
            return False
        accessinfo = krninfo.get(access_type)
        if accessinfo is None:
            return False
        typeinfo = accessinfo.get(root.common.precision_type)
        if typeinfo is None:
            return False
        return root.common.precision_level in typeinfo

    def update_info(self, krnnme, access_type, min_abc, min_dt):
        krninfo = self.device_info.get(krnnme, {})
        accessinfo = krninfo.get(access_type, {})
        typeinfo = accessinfo.get(root.common.precision_type, {})
        typeinfo[root.common.precision_level] = (min_abc, min_dt)
        accessinfo[root.common.precision_type] = typeinfo
        krninfo[access_type] = accessinfo
        self.device_info[krnnme] = krninfo
        self.device_infos[self.desc] = self.device_info
        fnme = os.path.join(root.common.device_dir, "device_infos.json")
        try:
            with open(fnme, "w") as fout:
                json.dump(self.device_infos, fout, indent=2)
        except IOError:
            self.error("Couldn't save device info to %s", fnme)

    def test_all2all(self, weights_transposed):
        krnnme = "matrix_multiplication"
        access_type = "row_x_col" if weights_transposed else "row_x_row"
        if self.check_info(krnnme, access_type):
            return
        self.evolve(krnnme, access_type,
                    self.test_all2all_partial, weights_transposed)

    def evolve(self, krnnme, access_type, target, *target_args):
        self.info("Will test kernel %s", krnnme)
        population = BenchmarkPopulation(
            filler=self, computed_times=self.computed_times,
            evaluation_routine=target, evaluation_routine_args=target_args)
        population.evolve()
        best = population.chromosomes[0]
        min_abc = best.numeric[0], best.numeric[1], best.numeric[2]
        min_dt = best.fitness_to_time(best.fitness)
        self.update_info(krnnme, access_type, min_abc, min_dt)

    def test_gd(self):
        krnnme = "matrix_multiplication"
        access_type = "col_x_col"
        if self.check_info(krnnme, access_type):
            return
        self.evolve(krnnme, access_type, self.test_gd_partial)

    def test_all2all_partial(self, a, b, c, weights_transposed):
        if self.device is None:
            self._init_all2all_forward(weights_transposed)

        return self._get_kernel_times(
            a, b, c, self.forward, self.forward.ocl_run)

    def test_gd_partial(self, a, b, c):
        if self.device is None:
            self._init_all2all_forward(False)
            self.forward.ocl_run()
            self.backward = GradientDescent(
                self.forward.workflow,
                learning_rate=0.0, learning_rate_bias=0.0,
                gradient_moment=0.0, gradient_moment_bias=0.0,
                store_gradient=False, apply_gradient=True)
            self._connect_backward_forward()

        return self._get_kernel_times(
            a, b, c, self.backward, self.backward.gpu_weights_update)

    def test_conv(self, weights_transposed):
        krnnme = "conv"
        access_type = "row_x_col" if weights_transposed else "row_x_row"
        if self.check_info(krnnme, access_type):
            return
        self.evolve(krnnme, access_type,
                    self.test_conv_partial, weights_transposed)

    def test_deconv(self, weights_transposed):
        krnnme = "deconv"
        access_type = "row_x_col" if weights_transposed else "row_x_row"
        if self.check_info(krnnme, access_type):
            return
        self.evolve(krnnme, access_type,
                    self.test_deconv_partial, weights_transposed)

    def test_gd_conv(self):
        krnnme = "conv"
        access_type = "col_x_col"
        if self.check_info(krnnme, access_type):
            return
        self.evolve(krnnme, access_type, self.test_gd_conv_partial)

    def _init_all2all_forward(self, weights_transposed):
        self.device = Device()
        self.device.device_info = self

        workflow = DummyWorkflow()

        self.forward = All2All(workflow, output_shape=3019,
                               weights_transposed=weights_transposed)
        self.forward.input = Vector(numpy.zeros([3019, 3019],
                                                dtype=self.dtype))
        prng.get().fill(self.forward.input.mem)
        self.forward.initialize(self.device)

    def _init_conv_forward(self, weights_transposed):
        self.device = Device()
        self.device.device_info = self

        self.forward = Conv(DummyWorkflow(), kx=7, ky=7, n_kernels=64,
                            weights_transposed=weights_transposed)
        self.forward.input = Vector(numpy.zeros([101, 47, 47, 32],
                                                dtype=self.dtype))
        prng.get().fill(self.forward.input.mem)
        self.forward.initialize(self.device)

    def _connect_backward_forward(self, need_err_input=False):
        self.backward.err_output = self.forward.output
        self.backward.output = self.forward.output
        self.backward.input = self.forward.input
        self.backward.weights = self.forward.weights
        self.backward.bias = self.forward.bias
        self.backward.need_err_input = need_err_input
        self.backward.initialize(self.device)

    def _get_kernel_times(self, a, b, c, unit, func):
        self.a_block_size, self.b_block_size, self.common_block_size = a, b, c
        try:
            unit.ocl_init(self.device)

            func()
            self.device.queue_.flush()
            self.device.queue_.finish()
            t0 = time.time()
            n = 3
            for _ in range(n):
                func()
            self.device.queue_.flush()
            self.device.queue_.finish()
            dt = (time.time() - t0) / n
        except opencl4py.CLRuntimeError as e:
            self.warning("OpenCL execution failed: %s", str(e))
            dt = 1000000000.0
        return dt

    def test_conv_partial(self, a, b, c, weights_transposed):
        if self.device is None:
            self._init_conv_forward(weights_transposed)

        return self._get_kernel_times(
            a, b, c, self.forward, self.forward.ocl_run)

    def test_gd_conv_partial(self, a, b, c):
        if self.device is None:
            self._init_conv_forward(False)
            self.forward.ocl_run()
            self.backward = GradientDescentConv(
                self.forward.workflow,
                kx=self.forward.kx, ky=self.forward.ky,
                n_kernels=self.forward.n_kernels,
                learning_rate=0.0, learning_rate_bias=0.0,
                gradient_moment=0.0, gradient_moment_bias=0.0,
                store_gradient=False, apply_gradient=True)
            self._connect_backward_forward()

        return self._get_kernel_times(
            a, b, c, self.backward, self.backward.gpu_weights_update)

    def test_deconv_partial(self, a, b, c, weights_transposed):
        if self.device is None:
            self._init_conv_forward(weights_transposed)
            self.forward.ocl_run()
            self.backward = GradientDescentConv(
                self.forward.workflow,
                kx=self.forward.kx, ky=self.forward.ky,
                n_kernels=self.forward.n_kernels,
                learning_rate=0.0, learning_rate_bias=0.0,
                gradient_moment=0.0, gradient_moment_bias=0.0,
                store_gradient=False, apply_gradient=True,
                weights_transposed=weights_transposed)
            self._connect_backward_forward(True)

        return self._get_kernel_times(
            a, b, c, self.backward, self.backward.gpu_err_input_update)


if __name__ == "__main__":
    Logger.setup(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="?",
                        help="OpenCL device to use.")
    args = parser.parse_args()
    if args.device == "?":
        logging.info("-d (--device) option must be specified\n"
                     "Devices available:\n%s",
                     opencl4py.Platforms().dump_devices())
        sys.exit(0)
    filler = DeviceInfosFiller()
    filler.do_the_job()
    sys.exit(0)
