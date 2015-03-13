#!/usr/bin/python3 -O
"""
Created on Mart 21, 2014

Configuration file for kanji.
Model – fully-connected Neural Network with MSE loss function.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


import os

from veles.config import root


# optional parameters

train_path = os.path.join(root.common.test_dataset_root, "new_kanji/train")

target_path = os.path.join(root.common.test_dataset_root, "new_kanji/target")


root.kanji.update({
    "decision": {"fail_iterations": 1000,
                 "max_epochs": 10000},
    "loss_function": "mse",
    "loader_name": "full_batch_auto_label_file_image_mse",
    "add_plotters": True,
    "image_saver": {"out_dirs":
                    [os.path.join(root.common.cache_dir, "tmp/test"),
                     os.path.join(root.common.cache_dir, "tmp/validation"),
                     os.path.join(root.common.cache_dir, "tmp/train")]},
    "loader": {"minibatch_size": 50,
               "force_cpu": False,
               "filename_types": ["png"],
               "train_paths": [train_path],
               "target_paths": [target_path],
               "color_space": "GRAY",
               "normalization_type": "linear",
               "target_normalization_type": "linear",
               "targets_shape": (24, 24),
               "background_color": (0,),
               "validation_ratio": 0.15},
    "snapshotter": {"prefix": "kanji"},
    "weights_plotter": {"limit": 16},
    "layers": [{"type": "all2all_tanh",
                "->": {"output_sample_shape": 250},
                "<-": {"learning_rate": 0.0001, "weights_decay": 0.00005}},
               {"type": "all2all_tanh",
                "->": {"output_sample_shape": 250},
                "<-": {"learning_rate": 0.0001, "weights_decay": 0.00005}},
               {"type": "all2all_tanh",
                "->": {"output_sample_shape": 24 * 24},
                "<-": {"learning_rate": 0.0001, "weights_decay": 0.00005}}]})
