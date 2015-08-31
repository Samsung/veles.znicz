#!/usr/bin/python3


import os
import veles
from veles.znicz.labels_printer import ResultWriter


launcher = veles(
    "veles/znicz/samples/MNIST/mnist.py",
    debug="MnistLoader", dry_run="init",
    snapshot="https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv"
             "_validation_0.70_train_0.23.4.pickle",
    stealth=True, device=0)

reversed_labels_mapping = launcher.workflow.loader.reversed_labels_mapping

data_dir = os.path.dirname(__file__)

data_path = "/usr/share/veles/data"
if not os.path.exists(data_path):
    print("Changed data path to %s" % data_dir)
    data_path = data_dir

launcher.testing = True

fwd = launcher.workflow.extract_forward_workflow(
    loader_name="file_list_image",
    loader_config={"minibatch_size": 1, "scale": (28, 28), "shuffle_limit": 0,
                   "background_color": (0,), "color_space": "GRAY",
                   "normalization_type": "linear",
                   "path_to_folder": os.path.join(data_path, "mnist_test"),
                   "path_to_test_text_file":
                   [os.path.join(data_dir, "mnist_test.txt")]},
    result_unit_factory=ResultWriter,
    result_unit_config={"output_path": os.path.join(data_dir, "result.txt")},
    cyclic=True)
fwd.result_unit.link_attrs(fwd.loader, "minibatch_size")
fwd.result_unit.link_attrs(
    fwd.loader, ("input_path", "path_to_test_text_file"))
fwd.result_unit.reversed_labels_mapping = reversed_labels_mapping
launcher.boot()
