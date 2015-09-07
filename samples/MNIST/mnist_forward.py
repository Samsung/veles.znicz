#!/usr/bin/python3


import os

from veles.loader import Loader
from veles.loader.file_image import FileListImageLoader
import veles


kwargs = {
    "debug": "Conv,All2AllRELU,All2AllSoftmax,MaxPooling",
    "dry_run": "init",
    "snapshot":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_"
        "validation_0.73_train_0.23.4.pickle",
    "stealth": True, "device": 0}

path = "veles/znicz/samples/MNIST/mnist.py"

data_dir = os.path.dirname(__file__)
data_path = "/usr/share/veles/data"
base_directory = os.path.join(data_path, "mnist_test")
if not os.path.exists(base_directory):
    print("Changed data path to %s" % data_dir)
    data_path = data_dir

launcher = veles(path, **kwargs)  # pylint: disable=E1102

# Find loader unit
for unit in launcher.workflow.units:
    if isinstance(unit, Loader):
        loader = unit

normalizer = loader.normalizer
labels_mapping = loader.labels_mapping

launcher.testing = True
launcher.workflow.plotters_are_enabled = False

# Delete and unlink avatar
launcher.workflow.loader.unlink_all()
launcher.workflow.del_ref(launcher.workflow.loader)

# Delete and unlink loader
loader.unlink_all()
launcher.workflow.del_ref(loader)

# Create new loader
launcher.workflow.loader = FileListImageLoader(
    launcher.workflow,
    minibatch_size=10, scale=(28, 28), shuffle_limit=0,
    background_color=(0,), color_space="GRAY",
    normalization_type="linear",
    base_directory=os.path.join(data_path, "mnist_test"),
    path_to_test_text_file=[os.path.join(data_dir, "mnist_test.txt")])
launcher.workflow.loader._normalizer = normalizer

# Link new loader
launcher.workflow.loader.link_from(launcher.workflow.repeater)
launcher.workflow.forwards[0].link_from(launcher.workflow.loader)
launcher.workflow.forwards[0].link_attrs(
    launcher.workflow.loader, ("input", "minibatch_data"))

launcher.workflow.evaluator.link_attrs(
    launcher.workflow.loader,
    "class_keys",
    ("indices", "minibatch_indices"),
    ("batch_size", "minibatch_size"),
    ("labels", "minibatch_labels"),
    ("max_samples_per_epoch", "total_samples"),
    "class_lengths", ("offset", "minibatch_offset"))
launcher.workflow.decision.link_attrs(
    launcher.workflow.loader, "minibatch_class", "last_minibatch",
    "minibatch_size", "class_lengths", "epoch_ended", "epoch_number")

launcher.workflow.evaluator.labels_mapping = labels_mapping

launcher.boot()
launcher.workflow.write_results(file=os.path.join(data_dir, "result.txt"))
