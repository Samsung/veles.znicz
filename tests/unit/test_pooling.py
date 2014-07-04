"""
Created on Dec 4, 2013

Unit test for pooling layer forward propagation.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import logging
import numpy
import unittest

from veles.config import root
import veles.formats as formats
import veles.opencl as opencl
import veles.opencl_types as opencl_types
import veles.random_generator as prng
import veles.znicz.gd_pooling as gd_pooling
import veles.znicz.pooling as pooling
from veles.tests.dummy_workflow import DummyWorkflow
from veles.znicz.tests.unit.gd_numdiff import GDNumDiff


class TestMaxPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._input = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input.mem = numpy.array(
            [3, 4, 3, 1, -1, -2, 1, 3, 2, 3, 3, 0, 4, 1,
             (-2), 0, 4, 4, -2, 1, 3, -3, -3, 4, 1, -3, -2, -4,
             (-3), 2, -1, 4, 2, 0, -3, 3, 1, -3, -4, -3, 0, -3,
             (-1), 0, -2, 2, 2, -4, -1, -1, 0, -2, 1, 3, 1, 2,
             2, -2, 4, 0, -1, 0, 1, 0, 0, 3, -3, 3, -1, 1,
             4, 0, -1, -2, 3, 4, -4, -2, -4, 3, -2, -3, -1, -1,
             (-1), -3, 3, 3, -2, -1, 3, 2, -1, -2, 4, -1, 2, 4,
             (-2), -1, 1, 3, -2, -2, 0, -2, 0, 4, -1, -2, -2, -3,
             3, 2, -2, 3, 1, -3, -2, -1, 4, -2, 0, -3, -1, 2,
             2, -3, -1, -1, -3, -2, 2, 3, 0, -2, 1, 2, 0, -3,
             (-4), 1, -1, 2, -1, 0, 3, -2, 4, -3, 4, 4, 1, -4,
             0, -1, 1, 3, 0, 1, 3, 4, -3, 2, 4, 3, -1, 0,
             (-1), 0, 1, -2, -4, 0, -4, -4, 2, 3, 2, -3, 1, 1,
             1, -1, -4, 3, 1, -1, -3, -4, -4, 3, -1, -4, -1, 0,
             (-1), -3, 4, 1, 2, -1, -2, -3, 3, 1, 3, -3, 4, -2],
            dtype=self._dtype).reshape(3, 5, 7, 2)

        self._gold_output = numpy.array(
            [[[[4, 4], [3, 3], [3, 4], [4, -4]],
              [[-3, 4], [-3, -4], [-4, -3], [1, -3]],
              [[4, -2], [-1, 0], [-3, 3], [-1, 1]]],
             [[[4, -3], [-4, 4], [-4, 3], [2, 4]],
              [[3, 3], [-2, -3], [4, 4], [-2, -3]],
              [[2, -3], [-3, 3], [1, -2], [0, -3]]],
             [[[-4, 3], [3, 4], [4, 4], [1, -4]],
              [[-4, 3], [-4, -4], [-4, -4], [1, 1]],
              [[4, -3], [2, -3], [3, -3], [4, -2]]]], dtype=self._dtype)

        self._gold_offs = numpy.array(
            [[[[16, 1], [20, 7], [10, 23], [12, 27]],
              [[28, 31], [34, 47], [38, 37], [54, 41]],
              [[58, 57], [60, 61], [66, 65], [68, 69]]],
             [[[70, 85], [76, 75], [78, 79], [96, 97]],
              [[112, 101], [102, 117], [120, 107], [110, 111]],
              [[126, 127], [130, 133], [136, 135], [138, 139]]],
             [[[140, 157], [146, 161], [148, 151], [152, 153]],
              [[184, 185], [172, 175], [190, 193], [180, 181]],
              [[198, 197], [200, 203], [204, 207], [208, 209]]]],
            dtype=numpy.int32)

    def tearDown(self):
        pass

    def test_ocl(self):
        self.do_test(opencl.Device())

    def test_cpu(self):
        self.do_test(None)

    def do_test(self, device):
        c = pooling.MaxAbsPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._input
        c.initialize(device=device)
        c.run()
        c.output.map_read()

        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        c.input_offset.map_read()
        max_diff = numpy.fabs(self._gold_offs.ravel() -
                              c.input_offset.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in offs matrix"
                        " is %.6f" % (max_diff))


class TestStochasticPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._input = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input.mem = numpy.array(
            [3, 4, 3, 1, -1, -2, 1, 3, 2, 3, 3, 0, 4, 1,
             (-2), 0, 4, 4, -2, 1, 3, -3, -3, 4, 1, -3, -2, -4,
             (-3), 2, -1, 4, 2, 0, -3, 3, 1, -3, -4, -3, 0, -3,
             (-1), 0, -2, 2, 2, -4, -1, -1, 0, -2, 1, 3, 1, 2,
             2, -2, 4, 0, -1, 0, 1, 0, 0, 3, -3, 3, -1, 1,
             4, 0, -1, -2, 3, 4, -4, -2, -4, 3, -2, -3, -1, -1,
             (-1), -3, 3, 3, -2, -1, 3, 2, -1, -2, 4, -1, 2, 4,
             (-2), -1, 1, 3, -2, -2, 0, -2, 0, 4, -1, -2, -2, -3,
             3, 2, -2, 3, 1, -3, -2, -1, 4, -2, 0, -3, -1, 2,
             2, -3, -1, -1, -3, -2, 2, 3, 0, -2, 1, 2, 0, -3,
             (-4), 1, -1, 2, -1, 0, 3, -2, 4, -3, 4, 4, 1, -4,
             0, -1, 1, 3, 0, 1, 3, 4, -3, 2, 4, 3, -1, 0,
             (-1), 0, 1, -2, -4, 0, -4, -4, 2, 3, 2, -3, 1, 1,
             1, -1, -4, 3, 1, -1, -3, -4, -4, 3, -1, -4, -1, 0,
             (-1), -3, 4, 1, 2, -1, -2, -3, 3, 1, 3, -3, 4, -2],
            dtype=self._dtype).reshape(3, 5, 7, 2)

        self._states = numpy.array(
            [3452505065, 1917670807, 14243993, 1512088576, 2532679766,
             42032469, 3977686139, 3929345226, 3024591612, 1155847603,
             3900060178, 3286160407, 361773878, 342814451, 1858416936,
             2731888911, 3189328186, 2434551697, 612289036, 2743596499,
             2540923290, 188307719, 3905385961, 3134457802, 3188727044,
             2050931879, 2097552857, 348211195, 3264702066, 447911779,
             289583577, 3598088191, 2132064443, 3812201575, 2804774660,
             723685343, 4228537715, 597093719, 3019744505, 2660009287,
             1166333641, 2456771611, 3539929856, 559060760, 2173850399,
             1088060875, 678580882, 1545058217, 3241502370, 2643173893,
             1252748113, 844359074, 3709896490, 3114648666, 2105054106,
             1419983755, 3013046293, 440976707, 1767929016, 76917521,
             1346819809, 1283870274, 2705743502, 156205557, 1287517583,
             2360445272, 1209474356, 3434394441, 2647746397, 1865761943,
             1681576517, 2859016960, 580744684, 673884363, 2204248846,
             3798309126, 2178972171, 1671845884, 807915027, 1395473190,
             1096088425, 1372068722, 2705427973, 1248419244, 2197735460,
             2990280104, 2119769883, 265531583, 3230098254, 3559736340,
             3756622901, 2737860416, 1922165871, 665634094, 702602010,
             502720101, 2008253188, 98204791, 2750047608, 3702750544,
             3782029303, 3122572441, 307383361, 1264149544, 1862078429,
             2745903319, 4073664302, 1849412661, 3941907606, 2387632366,
             2583853611, 4285039735, 1051974468, 820103150, 2741334962,
             1631715778, 3134404123, 3527595460, 802266283, 1092863310,
             256369161, 2503154297, 4226816944, 9059174, 235604055,
             3046042587, 1786352474, 1968126611, 3418658467, 1753676186,
             3856234169, 2870767384, 1991099142, 911573286, 1231908658,
             2897151610, 2621783617, 3944519027, 1899203944, 288426671,
             297488599, 1293371268, 2480714873, 3913640929, 3962544283,
             3531063258, 1794841430, 4223350298, 3613778179, 4260346596,
             170226505, 1509606908, 2076607403, 771080312, 4159830419,
             4735119, 291125641, 1241721552, 3658928982, 57488753,
             4435458, 3592840520, 1522593780, 1554357898, 3921937144,
             4101921651, 2311696104, 1388539015, 2248629125, 3786067859,
             1868547614, 2039579690, 3270621266, 3016603279, 3285615362,
             2764015768, 3496840041, 3296982980, 2105866779, 3029227253,
             4185801341, 2271330637, 2082583249, 2617849172, 2935246913,
             539494873, 2398015936, 1452025780, 3031806872, 2160424801,
             400169657, 606432033, 451864914, 1731025825, 3446633570,
             3340480570, 727990338, 3551645703, 3519581492, 4108357658,
             949459417, 535336188, 1648872210, 3485034089, 2507027080,
             698095907, 1513748585, 114069755, 2854733246, 2274659843,
             2845342460, 247513226, 2342179698, 1153879810, 268750086,
             2680088585, 4005409592, 268592642, 3880338517, 4157540705,
             2109309577, 155656024, 758830446, 2009360145, 1151232494,
             2958679655, 1174324678, 1963668583, 2533708984, 4274601652,
             3533274927, 2261022573, 3239805725, 2181275000, 2086196084,
             2281002611, 2524647062, 3836990583, 3018052164, 1027652860,
             1794244924, 1049186556, 2372531915, 4092090677, 727653924,
             1116540842, 1939431431, 2973703302, 1762346179, 1758483243,
             1925818609, 4277537547, 2087746361, 2946514082, 4242795738,
             2596067795, 2850035489, 4229786224, 3750846106, 1332061414,
             3669644956, 1680572837, 250449266, 3881686823, 1661924000,
             1497236985, 4152430535, 526030183, 2894930856, 161480108,
             3615216497, 3082578720, 1813850386, 3560349165, 2647855913,
             1952541972, 1672157613, 1419036354, 1238835779, 2950921527,
             3573241343, 867641998, 904160752, 3007460164, 408447203,
             2943099117, 2778239326, 1726311673], dtype=numpy.uint32)

        self._gold_output = numpy.array(
            [[[[3, 4], [1, 3], [3, 4], [4, 1]],
              [[-3, 4], [2, 3], [1, 3], [1, 2]],
              [[4, -2], [1, 0], [-3, 3], [-1, 1]]],
             [[[3, 3], [3, 4], [4, 3], [2, 4]],
              [[3, 3], [1, -2], [4, 4], [-2, 2]],
              [[2, -3], [2, 3], [1, 2], [0, -3]]],
             [[[1, 1], [3, 4], [4, 3], [1, -4]],
              [[1, 3], [1, 0], [2, 3], [1, 1]],
              [[4, 1], [2, -1], [3, 1], [4, -2]]]],
            dtype=self._dtype)

        self._gold_offs = numpy.array(
            [[[[0, 1], [6, 7], [10, 23], [12, 13]],
              [[28, 31], [46, 35], [52, 53], [54, 55]],
              [[58, 57], [62, 61], [66, 65], [68, 69]]],
             [[[86, 87], [74, 75], [94, 79], [96, 97]],
              [[112, 115], [116, 105], [120, 107], [110, 125]],
              [[126, 127], [132, 133], [136, 137], [138, 139]]],
             [[[156, 141], [146, 161], [150, 165], [152, 153]],
              [[170, 185], [186, 173], [178, 191], [180, 181]],
              [[198, 199], [200, 201], [206, 205], [208, 209]]]],
            dtype=numpy.int32)

    def tearDown(self):
        pass

    def test_ocl(self):
        self.do_test(opencl.Device())

    def test_cpu(self):
        numpy.seterr(over='ignore')
        self.do_test(None)
        numpy.seterr(over='warn')

    def do_test(self, device):
        c = pooling.StochasticPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._input
        c.initialize(device=device, states=self._states)
        c.run()
        c.output.map_read()
        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        c.input_offset.map_read()
        max_diff = numpy.fabs(self._gold_offs.ravel() -
                              c.input_offset.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in offs matrix"
                        " is %.6f" % (max_diff))


class TestStochasticAbsPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._input = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input.mem = numpy.array(
            [3, 4, 3, 1, -1, -2, 1, 3, 2, 3, 3, 0, 4, 1,
             (-2), 0, 4, 4, -2, 1, 3, -3, -3, 4, 1, -3, -2, -4,
             (-3), 2, -1, 4, 2, 0, -3, 3, 1, -3, -4, -3, 0, -3,
             (-1), 0, -2, 2, 2, -4, -1, -1, 0, -2, 1, 3, 1, 2,
             2, -2, 4, 0, -1, 0, 1, 0, 0, 3, -3, 3, -1, 1,
             4, 0, -1, -2, 3, 4, -4, -2, -4, 3, -2, -3, -1, -1,
             (-1), -3, 3, 3, -2, -1, 3, 2, -1, -2, 4, -1, 2, 4,
             (-2), -1, 1, 3, -2, -2, 0, -2, 0, 4, -1, -2, -2, -3,
             3, 2, -2, 3, 1, -3, -2, -1, 4, -2, 0, -3, -1, 2,
             2, -3, -1, -1, -3, -2, 2, 3, 0, -2, 1, 2, 0, -3,
             (-4), 1, -1, 2, -1, 0, 3, -2, 4, -3, 4, 4, 1, -4,
             0, -1, 1, 3, 0, 1, 3, 4, -3, 2, 4, 3, -1, 0,
             (-1), 0, 1, -2, -4, 0, -4, -4, 2, 3, 2, -3, 1, 1,
             1, -1, -4, 3, 1, -1, -3, -4, -4, 3, -1, -4, -1, 0,
             (-1), -3, 4, 1, 2, -1, -2, -3, 3, 1, 3, -3, 4, -2],
            dtype=self._dtype).reshape(3, 5, 7, 2)

        self._states = numpy.array(
            [3452505065, 1917670807, 14243993, 1512088576, 2532679766,
             42032469, 3977686139, 3929345226, 3024591612, 1155847603,
             3900060178, 3286160407, 361773878, 342814451, 1858416936,
             2731888911, 3189328186, 2434551697, 612289036, 2743596499,
             2540923290, 188307719, 3905385961, 3134457802, 3188727044,
             2050931879, 2097552857, 348211195, 3264702066, 447911779,
             289583577, 3598088191, 2132064443, 3812201575, 2804774660,
             723685343, 4228537715, 597093719, 3019744505, 2660009287,
             1166333641, 2456771611, 3539929856, 559060760, 2173850399,
             1088060875, 678580882, 1545058217, 3241502370, 2643173893,
             1252748113, 844359074, 3709896490, 3114648666, 2105054106,
             1419983755, 3013046293, 440976707, 1767929016, 76917521,
             1346819809, 1283870274, 2705743502, 156205557, 1287517583,
             2360445272, 1209474356, 3434394441, 2647746397, 1865761943,
             1681576517, 2859016960, 580744684, 673884363, 2204248846,
             3798309126, 2178972171, 1671845884, 807915027, 1395473190,
             1096088425, 1372068722, 2705427973, 1248419244, 2197735460,
             2990280104, 2119769883, 265531583, 3230098254, 3559736340,
             3756622901, 2737860416, 1922165871, 665634094, 702602010,
             502720101, 2008253188, 98204791, 2750047608, 3702750544,
             3782029303, 3122572441, 307383361, 1264149544, 1862078429,
             2745903319, 4073664302, 1849412661, 3941907606, 2387632366,
             2583853611, 4285039735, 1051974468, 820103150, 2741334962,
             1631715778, 3134404123, 3527595460, 802266283, 1092863310,
             256369161, 2503154297, 4226816944, 9059174, 235604055,
             3046042587, 1786352474, 1968126611, 3418658467, 1753676186,
             3856234169, 2870767384, 1991099142, 911573286, 1231908658,
             2897151610, 2621783617, 3944519027, 1899203944, 288426671,
             297488599, 1293371268, 2480714873, 3913640929, 3962544283,
             3531063258, 1794841430, 4223350298, 3613778179, 4260346596,
             170226505, 1509606908, 2076607403, 771080312, 4159830419,
             4735119, 291125641, 1241721552, 3658928982, 57488753,
             4435458, 3592840520, 1522593780, 1554357898, 3921937144,
             4101921651, 2311696104, 1388539015, 2248629125, 3786067859,
             1868547614, 2039579690, 3270621266, 3016603279, 3285615362,
             2764015768, 3496840041, 3296982980, 2105866779, 3029227253,
             4185801341, 2271330637, 2082583249, 2617849172, 2935246913,
             539494873, 2398015936, 1452025780, 3031806872, 2160424801,
             400169657, 606432033, 451864914, 1731025825, 3446633570,
             3340480570, 727990338, 3551645703, 3519581492, 4108357658,
             949459417, 535336188, 1648872210, 3485034089, 2507027080,
             698095907, 1513748585, 114069755, 2854733246, 2274659843,
             2845342460, 247513226, 2342179698, 1153879810, 268750086,
             2680088585, 4005409592, 268592642, 3880338517, 4157540705,
             2109309577, 155656024, 758830446, 2009360145, 1151232494,
             2958679655, 1174324678, 1963668583, 2533708984, 4274601652,
             3533274927, 2261022573, 3239805725, 2181275000, 2086196084,
             2281002611, 2524647062, 3836990583, 3018052164, 1027652860,
             1794244924, 1049186556, 2372531915, 4092090677, 727653924,
             1116540842, 1939431431, 2973703302, 1762346179, 1758483243,
             1925818609, 4277537547, 2087746361, 2946514082, 4242795738,
             2596067795, 2850035489, 4229786224, 3750846106, 1332061414,
             3669644956, 1680572837, 250449266, 3881686823, 1661924000,
             1497236985, 4152430535, 526030183, 2894930856, 161480108,
             3615216497, 3082578720, 1813850386, 3560349165, 2647855913,
             1952541972, 1672157613, 1419036354, 1238835779, 2950921527,
             3573241343, 867641998, 904160752, 3007460164, 408447203,
             2943099117, 2778239326, 1726311673], dtype=numpy.uint32)

        self._gold_output = numpy.array(
            [[[[3, 4], [-1, 3], [-3, -3], [4, -4]],
              [[-3., 4.], [2., 3.], [1., -3.], [1., -3.]],
              [[4., -2.], [1., 0.], [-3., 3.], [-1., 1.]]],
             [[[3., -2.], [3., 4.], [-4., -2.], [2., 4.]],
              [[3., 3.], [-2., -2.], [4., 4.], [-2., -3.]],
              [[2., -3.], [2., 3.], [1., -2.], [0., -3.]]],
             [[[-4., 1.], [3., 4.], [-3., 3.], [-1., -4.]],
              [[1., 3.], [-4., -4.], [-4., -4.], [1., 1.]],
              [[4., -3.], [-2., -1.], [3., -3.], [4., -2.]]]],
            dtype=self._dtype)

        self._gold_offs = numpy.array(
            [[[[0, 1], [4, 7], [22, 25], [12, 27]],
              [[28, 31], [46, 35], [52, 37], [54, 41]],
              [[58, 57], [62, 61], [66, 65], [68, 69]]],
             [[[86, 73], [74, 75], [78, 93], [96, 97]],
              [[112, 115], [102, 105], [120, 107], [110, 111]],
              [[126, 127], [132, 133], [136, 135], [138, 139]]],
             [[[140, 141], [146, 161], [162, 165], [166, 153]],
              [[182, 185], [172, 175], [190, 193], [180, 181]],
              [[198, 197], [202, 201], [206, 207], [208, 209]]]],
            dtype=numpy.int32)

    def tearDown(self):
        pass

    def test_ocl(self):
        self.do_test(opencl.Device())

    def test_cpu(self):
        numpy.seterr(over='ignore')
        self.do_test(None)
        numpy.seterr(over='warn')

    def do_test(self, device):
        c = pooling.StochasticAbsPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._input
        c.initialize(device=device, states=self._states)
        c.run()
        c.output.map_read()
        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))
        c.input_offset.map_read()
        max_diff = numpy.fabs(self._gold_offs.ravel() -
                              c.input_offset.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in offs matrix"
                        " is %.6f" % (max_diff))


class TestGDMaxPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input = numpy.array(
            [[[3, 3, -1, 1, 2, 3, 4],
              [-2, 4, -2, 3, -3, 1, -2],
              [-3, -1, 2, -3, 1, -4, 0],
              [-1, -2, 2, -1, 0, 1, 1],
              [2, 4, -1, 1, 0, -3, -1]],
             [[4, -1, 3, -4, -4, -2, -1],
              [-1, 3, -2, 3, -1, 4, 2],
              [-2, 1, -2, 0, 0, -1, -2],
              [3, -2, 1, -2, 4, 0, -1],
              [2, -1, -3, 2, 0, 1, 0]],
             [[-4, -1, -1, 3, 4, 4, 1],
              [0, 1, 0, 3, -3, 4, -1],
              [-1, 1, -4, -4, 2, 2, 1],
              [1, -4, 1, -3, -4, -1, -1],
              [-1, 4, 2, -2, 3, 3, 4]]], dtype=self._dtype)
        self._input.shape = (3, 5, 7, 1)
        self._input_offset = numpy.array(
            [8, 10, 5, 6, 14, 17, 19, 27, 29, 30, 33, 34,
             35, 38, 39, 48, 56, 51, 60, 55, 63, 65, 68, 69,
             70, 73, 74, 76, 92, 86, 95, 90, 99, 100,
             102, 104], dtype=numpy.int32)
        self._err_output = numpy.array(
            [1, 3, 0.5, -4, 1, -2, -3, -1, -1, 3, -3, -0.5,
             4, -4, -0.3, -3, -1, -3, 2, -2, -4, 2, -1, -3,
             (-4), 2, 3, 2, -1, -1, -3, 4, -2, 2, 0.3, -4], dtype=self._dtype)
        self._gold_err_input = numpy.array(
            [[[0, 0, 0, 0, 0, 0.5, -4],
              [0, 1, 0, 3, 0, 0, 0],
              [1, 0, 0, -2, 0, -3, 0],
              [0, 0, 0, 0, 0, 0, -1],
              [0, -1, 3, 0, 0, -3, -0.5]],
             [[4, 0, 0, -4, -0.3, 0, 0],
              [0, 0, 0, 0, 0, 0, -3],
              [0, 0, -3, 0, 0, 0, -2],
              [-1, 0, 0, 0, 2, 0, 0],
              [-4, 0, 2, 0, 0, -1, -3]],
             [[-4, 0, 0, 2, 3, 0, 2],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, -1, 0, 0, 0, 4],
              [0, -1, 0, 0, -3, 0, 0],
              [0, -2, 2, 0, 0.3, 0, -4]]], dtype=self._dtype)

        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def test_fixed_gpu(self):
        return self._test_fixed(self.device)

    def test_fixed_cpu(self):
        return self._test_fixed(None)

    def _test_fixed(self, device):
        logging.info('starting OpenCL max pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDMaxPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = formats.Vector()
        c.input.mem = self._input.copy()
        c.input_offset = formats.Vector()
        c.input_offset.mem = self._input_offset.copy()
        c.err_output = formats.Vector()
        c.err_output.mem = self._err_output.copy()
        c.initialize(device=device)
        c.err_input.map_invalidate()
        c.err_input.mem[:] = 1.0e30
        c.run()
        c.err_input.map_read()  # get results back
        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001,
                        "max difference in err_input is %.6f" % (max_diff))
        logging.info("test passed")

        # We cannot check by numeric differentiation here
        # 'cause of the non-differentiable function "max".


class TestAvgPooling(unittest.TestCase):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._input = formats.Vector()
        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input.mem = numpy.array(
            [3, 4, 3, 1, -1, -2, 1, 3, 2, 3, 3, 0, 4, 1,
             (-2), 0, 4, 4, -2, 1, 3, -3, -3, 4, 1, -3, -2, -4,
             (-3), 2, -1, 4, 2, 0, -3, 3, 1, -3, -4, -3, 0, -3,
             (-1), 0, -2, 2, 2, -4, -1, -1, 0, -2, 1, 3, 1, 2,
             2, -2, 4, 0, -1, 0, 1, 0, 0, 3, -3, 3, -1, 1,
             4, 0, -1, -2, 3, 4, -4, -2, -4, 3, -2, -3, -1, -1,
             (-1), -3, 3, 3, -2, -1, 3, 2, -1, -2, 4, -1, 2, 4,
             (-2), -1, 1, 3, -2, -2, 0, -2, 0, 4, -1, -2, -2, -3,
             3, 2, -2, 3, 1, -3, -2, -1, 4, -2, 0, -3, -1, 2,
             2, -3, -1, -1, -3, -2, 2, 3, 0, -2, 1, 2, 0, -3,
             (-4), 1, -1, 2, -1, 0, 3, -2, 4, -3, 4, 4, 1, -4,
             0, -1, 1, 3, 0, 1, 3, 4, -3, 2, 4, 3, -1, 0,
             (-1), 0, 1, -2, -4, 0, -4, -4, 2, 3, 2, -3, 1, 1,
             1, -1, -4, 3, 1, -1, -3, -4, -4, 3, -1, -4, -1, 0,
             (-1), -3, 4, 1, 2, -1, -2, -3, 3, 1, 3, -3, 4, -2],
            dtype=self._dtype).reshape(3, 5, 7, 2)

        self._gold_output = numpy.array(
            [[[[2, 2.25], [0.25, -0.25], [0.75, 1], [1, -1.5]],
              [[-1.75, 2], [0, -0.5], [-0.5, -1.25], [0.5, -0.5]],
              [[3, -1], [0, 0], [-1.5, 3], [-1, 1]]],
             [[[1.25, -0.5], [0, 0.75], [-0.75, -0.75], [0.5, 1.5]],
              [[0, 1.75], [-0.75, -2], [0.75, -0.75], [-1.5, -0.5]],
              [[0.5, -2], [-0.5, 0.5], [0.5, 0], [0, -3]]],
             [[[-1, 1.25], [1.25, 0.75], [2.25, 1.5], [0, -2]],
              [[-0.75, 0], [-2.5, -2.25], [-0.25, -0.25], [0, 0.5]],
              [[1.5, -1], [0, -2], [3, -1], [4, -2]]]], dtype=self._dtype)

    def tearDown(self):
        pass

    def test_ocl(self):
        self.do_test(opencl.Device())

    def test_cpu(self):
        self.do_test(None)

    def do_test(self, device):
        c = pooling.AvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = self._input
        cur_device = opencl.Device()
        c.initialize(device=cur_device)
        c.run()
        c.output.map_read()
        max_diff = numpy.fabs(self._gold_output.ravel() -
                              c.output.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in output matrix"
                        " is %.6f" % (max_diff))


class TestGDAvgPooling(unittest.TestCase, GDNumDiff):
    def setUp(self):
        root.common.unit_test = True
        root.common.plotters_disabled = True

        self._dtype = opencl_types.dtypes[root.common.precision_type]
        self._input = numpy.array([
            [[[3, 6], [3, 6], [-1, -2], [1, 2], [2, 4], [3, 6], [4, 8]],
             [[-2, -4], [4, 8], [-2, -4], [3, 6], [-3, -6], [1, 2], [-2, -4]],
             [[-3, -6], [-1, -2], [2, 4], [-3, -6], [1, 2], [-4, -8], [0, 0]],
             [[-1, -2], [-2, -4], [2, 4], [-1, -2], [0, 0], [1, 2], [1, 2]],
             [[2, 4], [4, 8], [-1, -2], [1, 2], [0, 0], [-3, -6], [-1, -2]]],
            [[[4, 8], [-1, -2], [3, 6], [-4, -8], [-4, -8], [-2, -4],
              [-1, -2]],
             [[-1, -2], [3, 6], [-2, -4], [3, 6], [-1, -2], [4, 8], [2, 4]],
             [[-2, -4], [1, 2], [-2, -4], [0, 0], [0, 0], [-1, -2], [-2, -4]],
             [[3, 6], [-2, -4], [1, 2], [-2, -4], [4, 8], [0, 0], [-1, -2]],
             [[2, 4], [-1, -2], [-3, -6], [2, 4], [0, 0], [1, 2], [0, 0]]],
            [[[-4, -8], [-1, -2], [-1, -2], [3, 6], [4, 8], [4, 8], [1, 2]],
             [[0, 0], [1, 2], [0, 0], [3, 6], [-3, -6], [4, 8], [-1, -2]],
             [[-1, -2], [1, 2], [-4, -8], [-4, -8], [2, 4], [2, 4], [1, 2]],
             [[1, 2], [-4, -8], [1, 2], [-3, -6], [-4, -8], [-1, -2],
              [-1, -2]],
             [[-1, -2], [4, 8], [2, 2], [-2, -4],
              [3, 6], [3, 6], [4, 8]]]], dtype=self._dtype)
        self._err_output = numpy.array(
            [[[[1, 2], [3, 6], [0.5, 1], [-4, -8]],
              [[1, 2], [-2, -4], [-3, -6], [-1, -2]],
              [[-1, -2], [3, 6], [-3, -6], [-0.5, -1]]],
             [[[4, 8], [-4, -8], [-0.3, -0.6], [-3, -6]],
              [[-1, -2], [-3, -6], [2, 4], [-2, -4]],
              [[-4, -8], [2, 4], [-1, -2], [-3, -6]]],
             [[[-4, -8], [2, 4], [3, 6], [2, 4]],
              [[-1, -2], [-1, -2], [-3, -6], [4, 8]],
              [[-2, -4], [2, 4], [0.3, 0.6], [-4, -8]]]], dtype=self._dtype)
        self._gold_err_input = numpy.array([
            [[[0.25, 0.5], [0.25, 0.5], [0.75, 1.5], [0.75, 1.5],
              [0.125, 0.25], [0.125, 0.25], [-2, -4]],
             [[0.25, 0.5], [0.25, 0.5], [0.75, 1.5], [0.75, 1.5],
              [0.125, 0.25], [0.125, 0.25], [-2, -4]],
             [[0.25, 0.5], [0.25, 0.5], [-0.5, -1], [-0.5, -1],
              [-0.75, -1.5], [-0.75, -1.5], [-0.5, -1]],
             [[0.25, 0.5], [0.25, 0.5], [-0.5, -1], [-0.5, -1],
              [-0.75, -1.5], [-0.75, -1.5], [-0.5, -1]],
             [[-0.5, -1], [-0.5, -1], [1.5, 3], [1.5, 3],
              [-1.5, -3], [-1.5, -3], [-0.5, -1]]],
            [[[1, 2], [1, 2], [-1, -2], [-1, -2],
              [-0.075, -0.15], [-0.075, -0.15], [-1.5, -3]],
             [[1, 2], [1, 2], [-1, -2], [-1, -2],
              [-0.075, -0.15], [-0.075, -0.15], [-1.5, -3]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.75, -1.5], [-0.75, -1.5],
              [0.5, 1], [0.5, 1], [-1, -2]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.75, -1.5], [-0.75, -1.5],
              [0.5, 1], [0.5, 1], [-1, -2]],
             [[-2, -4], [-2, -4], [1, 2], [1, 2],
              [-0.5, -1], [-0.5, -1], [-3, -6]]],
            [[[-1, -2], [-1, -2], [0.5, 1], [0.5, 1],
              [0.75, 1.5], [0.75, 1.5], [1, 2]],
             [[-1, -2], [-1, -2], [0.5, 1], [0.5, 1],
              [0.75, 1.5], [0.75, 1.5], [1, 2]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5],
              [-0.75, -1.5], [-0.75, -1.5], [2, 4]],
             [[-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5], [-0.25, -0.5],
              [-0.75, -1.5], [-0.75, -1.5], [2, 4]],
             [[-1, -2], [-1, -2], [1, 2], [1, 2],
              [0.15, 0.3], [0.15, 0.3], [-4, -8]]]], dtype=self._dtype)

        if not hasattr(self, "device"):
            self.device = opencl.Device()

    def _test_fixed_gpu(self):
        return self._test_fixed(self.device)

    def _test_fixed_cpu(self):
        return self._test_fixed(None)

    def _test_fixed(self, device):
        logging.info('starting OpenCL avg pooling layer gradient descent '
                     'test...')
        c = gd_pooling.GDAvgPooling(DummyWorkflow(), kx=2, ky=2)
        c.input = formats.Vector()
        c.input.mem = self._input.copy()
        c.err_output = formats.Vector()
        c.err_output.mem = self._err_output.copy()
        c.initialize(device=device)
        c.err_input.map_invalidate()
        c.err_input.mem[:] = 1.0e30
        c.run()
        c.err_input.map_read()  # get results back

        max_diff = numpy.fabs(self._gold_err_input.ravel() -
                              c.err_input.mem.ravel()).max()
        self.assertLess(max_diff, 0.0001, "max difference in err_input matrix"
                        " is %.6f" % (max_diff))
        logging.info("test passed")

        logging.info("reverifying by numeric differentiation...")

        forward = pooling.AvgPooling(DummyWorkflow(), kx=c.kx, ky=c.ky)
        forward.input = formats.Vector()
        forward.input.mem = self._input.copy()
        forward.initialize(device=self.device)
        forward.run()
        forward.output.map_read()
        target = forward.output.mem.ravel() - self._err_output.ravel()

        self.numdiff_check_gd(forward, self._input, None, None, target,
                              c.err_input.mem, None, None,
                              logging.info, self.assertLess)

        logging.info("test passed")

    def test_random_numeric_gpu(self):
        self._test_random_numeric(self.device, (3, 3))
        self._test_random_numeric(self.device, (1, 1))

    def test_random_numeric_cpu(self):
        self._test_random_numeric(None, (3, 3))
        self._test_random_numeric(None, (1, 1))

    def _test_random_numeric(self, device, sliding):
        logging.info("Will test AvgPooling layer forward-backward "
                     "via numeric differentiation")

        inp = numpy.zeros([2, 6, 6, 3], dtype=self._dtype)
        prng.get().fill(inp)
        forward = pooling.AvgPooling(DummyWorkflow(), kx=3, ky=3,
                                     sliding=sliding)
        forward.input = formats.Vector()
        forward.input.mem = inp.copy()
        forward.initialize(device=self.device)
        forward.run()

        forward.output.map_read()
        target = numpy.zeros_like(forward.output.mem)
        prng.get().fill(target)
        err_output = forward.output.mem - target

        c = gd_pooling.GDAvgPooling(
            DummyWorkflow(), kx=forward.kx, ky=forward.ky,
            sliding=forward.sliding)
        c.err_output = formats.Vector()
        c.err_output.mem = err_output.copy()
        c.input = formats.Vector()
        c.input.mem = inp.copy()
        c.output = formats.Vector()
        c.output.mem = c.err_output.mem.copy()
        c.initialize(device=device)
        c.run()
        c.err_input.map_read()

        err_input = c.err_input.mem.ravel()

        self.numdiff_check_gd(forward, inp, None, None, target,
                              err_input, None, None,
                              logging.info, self.assertLess,
                              error_function_averaged=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
