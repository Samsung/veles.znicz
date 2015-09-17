#!/usr/bin/python3


import os

from veles.loader import Loader
from veles.loader.file_image import FileListImageLoader
import veles


if __name__ == "__main__":
    kwargs = {
        "dry_run": "init",
        "snapshot":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_"
        "validation_0.75_train_0.04.4.pickle",
        "stealth": True, "device": 0}

    path_to_model = "veles/znicz/samples/MNIST/mnist.py"
    data_path = os.path.dirname(__file__)

    launcher = veles(path_to_model, **kwargs)  # pylint: disable=E1102

    # Find loader unit
    for unit in launcher.workflow.units:
        if isinstance(unit, Loader):
            loader = unit

    normalizer = loader.normalizer
    labels_mapping = loader.labels_mapping

    launcher.testing = True
    launcher.workflow.plotters_are_enabled = False

    new_loader = launcher.workflow.change_unit(
        loader.name,
        FileListImageLoader(
            launcher.workflow,
            minibatch_size=10, scale=(28, 28), shuffle_limit=0,
            background_color=(0,), color_space="GRAY",
            normalization_type="linear",
            base_directory=os.path.join(data_path, "mnist_test"),
            path_to_test_text_file=[os.path.join(data_path,
                                                 "mnist_test.txt")]))

    new_loader._normalizer = normalizer
    launcher.workflow.evaluator.labels_mapping = labels_mapping

    launcher.boot()
    launcher.workflow.write_results(file=os.path.join(data_path, "result.txt"))
