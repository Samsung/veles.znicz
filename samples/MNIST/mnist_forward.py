#!/usr/bin/python3


import os

import veles
from veles.config import root
from veles.downloader import Downloader
from veles.loader.file_image import FileListImageLoader


def create_forward(workflow, normalizer, labels_mapping, loader_config):
    # Disable plotters:
    workflow.plotters_are_enabled = False

    # Link downloader
    workflow.start_point.unlink_after()
    workflow.downloader = Downloader(
        workflow,
        url="https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/"
            "mnist_test.tar",
        directory=root.common.dirs.datasets,
        files=["mnist_test"])
    workflow.downloader.link_from(workflow.start_point)
    workflow.repeater.link_from(workflow.downloader)

    # Cnanging MnistLoader for another Loader:
    new_loader = workflow.change_unit(
        workflow.loader.name,
        FileListImageLoader(workflow, **loader_config))

    # Link attributes:
    # TODO: remove link attributes after adding in change_unit() function
    # TODO: data links transmission
    workflow.forwards[0].link_attrs(
        new_loader, ("input", "minibatch_data"))

    workflow.evaluator.link_attrs(
        new_loader,
        "class_keys",
        ("batch_size", "minibatch_size"),
        ("labels", "minibatch_labels"),
        ("max_samples_per_epoch", "total_samples"),
        "class_lengths", ("offset", "minibatch_offset"))
    workflow.decision.link_attrs(
        new_loader, "minibatch_class", "last_minibatch",
        "minibatch_size", "class_lengths", "epoch_ended", "epoch_number")

    # Set normalizer from previous Loader to new one:
    new_loader._normalizer = normalizer

    # Set labels_mapping and class_keys in Evaluator to correct writting the
    # results:
    workflow.evaluator.labels_mapping = labels_mapping
    workflow.evaluator.class_keys = new_loader.class_keys


if __name__ == "__main__":
    kwargs = {
        "dry_run": "init",
        "snapshot":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_"
        "validation_0.78_train_0.16.4.pickle",
        "stealth": True,
        "device": 0}
    path_to_model = "veles/znicz/samples/MNIST/mnist.py"
    data_path = os.path.join(root.common.dirs.datasets, "mnist_test")

    # Load workflow from snapshot
    launcher = veles(path_to_model, **kwargs)  # pylint: disable=E1102

    # Swith to testing mode:
    launcher.testing = True
    loader_conf = {"minibatch_size": 10,
                   "scale": (28, 28),
                   "background_color": (0,),
                   "color_space": "GRAY",
                   "normalization_type": "linear",
                   "base_directory": os.path.join(data_path, "pictures"),
                   "path_to_test_text_file":
                   [os.path.join(data_path, "mnist_test.txt")]}
    create_forward(
        launcher.workflow, normalizer=launcher.workflow.loader.normalizer,
        labels_mapping=launcher.workflow.loader.labels_mapping,
        loader_config=loader_conf)

    # Initialize and run new workflow:
    launcher.boot()

    # Write results:
    launcher.workflow.write_results(file=os.path.join(data_path, "result.txt"))
