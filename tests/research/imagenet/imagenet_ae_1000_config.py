"""
Created on Jule 18, 2014

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""

import os

from veles.config import root


LR = 0.000003
WD = 0.004
GM = 0.9
L1_VS_L2 = 0.0

LRFT = 0.001
LRFTB = LRFT * 2

LRAA = 0.03
LRBAA = LRAA * 2
WDAA = 0.0005
WDBAA = 0
GMAA = 0.9
GMBAA = GM

FILLING = "gaussian"
STDDEV_CONV = 0.01
STDDEV_AA = 0.005

root.common.precision_type = "float"

root.defaults = {
    "decision": {"fail_iterations": 100,
                 "max_epochs": 50,
                 "use_dynamic_alpha": False,
                 "do_export_weights": True},
    "loader": {"year": "2014",
               "series": "img",
               "minibatch_size": 360,
               "path": "/export/home/imagenet/",
               "sx": 192,
               "sy": 192},
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir,
                                  "tmp_imagenet/test"),
                     os.path.join(root.common.cache_dir,
                                  "tmp_imagenet/validation"),
                     os.path.join(root.common.cache_dir,
                                  "tmp_imagenet/train")]},
    "snapshotter": {"prefix": "imagenet_ae"},
    "imagenet": {"from_snapshot_add_layer": True,
                 "fine_tuning_noise": 0,
                 "layers":
                 [{"type": "ae_begin"},  # 216
                  {"type": "conv", "n_kernels": 96,
                   "kx": 9, "ky": 9, "sliding": (3, 3),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "ae_begin"},  # 72
                  {"type": "conv", "n_kernels": 192,
                   "kx": 6, "ky": 6, "sliding": (2, 2),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "ae_begin"},  # 36
                  {"type": "conv", "n_kernels": 224,
                   "kx": 6, "ky": 6, "sliding": (2, 2),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "ae_begin"},  # 18
                  {"type": "conv", "n_kernels": 256,
                   "kx": 6, "ky": 6, "sliding": (2, 2),
                   "learning_rate": LR,
                   "learning_rate_ft": LRFT,
                   "weights_decay": WD,
                   "gradient_moment": GM,
                   "weights_filling": FILLING,
                   "weights_stddev": STDDEV_CONV,
                   "l1_vs_l2": L1_VS_L2},
                  {"type": "ae_end"},

                  {"type": "activation_mul"},
                  {"type": "all2all_tanh", "output_shape": 4096,
                   "learning_rate": LRAA, "learning_rate_bias": LRBAA,
                   "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
                   "weights_decay": WDAA, "weights_decay_bias": WDBAA,
                   "gradient_moment": GMAA, "gradient_moment_bias": GMBAA,
                   "weights_filling": "gaussian", "bias_filling": "gaussian",
                   "weights_stddev": STDDEV_AA, "bias_stddev": STDDEV_AA,
                   "l1_vs_l2": L1_VS_L2},
                  #{"type": "dropout", "dropout_ratio": 0.5},

                  {"type": "all2all_tanh", "output_shape": 4096,
                   "learning_rate": LRAA, "learning_rate_bias": LRBAA,
                   "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
                   "weights_decay": WDAA, "weights_decay_bias": WDBAA,
                   "gradient_moment": GMAA, "gradient_moment_bias": GMBAA,
                   "weights_filling": "gaussian", "bias_filling": "gaussian",
                   "weights_stddev": STDDEV_AA, "bias_stddev": STDDEV_AA,
                   "l1_vs_l2": L1_VS_L2},
                  #{"type": "dropout", "dropout_ratio": 0.5},

                  {"type": "softmax", "output_shape": 1001,
                   "learning_rate": LRAA, "learning_rate_bias": LRBAA,
                   "learning_rate_ft": LRFT, "learning_rate_ft_bias": LRFTB,
                   "weights_decay": WDAA, "weights_decay_bias": WDBAA,
                   "gradient_moment": GMAA, "gradient_moment_bias": GMBAA,
                   "weights_filling": "gaussian", "bias_filling": "gaussian",
                   "bias_stddev": 0.01, "weights_stddev": 0.01,
                   "l1_vs_l2": L1_VS_L2}]}}
