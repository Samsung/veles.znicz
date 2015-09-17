#!/usr/bin/python3


import os

from veles.loader.file_image import FileListImageLoader
import veles


if __name__ == "__main__":
    kwargs = {
        "dry_run": "init",
        "snapshot":
        "https://s3-eu-west-1.amazonaws.com/veles.forge/MNIST/mnist_conv_"
        "validation_0.78_train_0.16.4.pickle",
        "stealth": True,
        "device": 0}

    path_to_model = "veles/znicz/samples/MNIST/mnist.py"
    data_path = os.path.dirname(__file__)

    # Load workflow from snapshot
    launcher = veles(path_to_model, **kwargs)  # pylint: disable=E1102

    # Save normalizer and labels_mapping from old Loader:
    normalizer = launcher.workflow.loader.normalizer
    labels_mapping = launcher.workflow.loader.labels_mapping

    # Swith to testing mode:
    launcher.testing = True

    # Disable plotters:
    launcher.workflow.plotters_are_enabled = False

    # Cnanging MnistLoader for another Loader:
    new_loader = launcher.workflow.change_unit(
        launcher.workflow.loader.name,
        FileListImageLoader(
            launcher.workflow,
            minibatch_size=10, scale=(28, 28), background_color=(0,),
            color_space="GRAY", normalization_type="linear",
            base_directory=os.path.join(data_path, "mnist_test"),
            path_to_test_text_file=[os.path.join(data_path,
                                                 "mnist_test.txt")]))

    # Link attributes:
    # TODO: remove link attributes after adding in change_unit() function
    # TODO: data links transmission
    launcher.workflow.forwards[0].link_attrs(
        new_loader, ("input", "minibatch_data"))

    launcher.workflow.evaluator.link_attrs(
        new_loader,
        "class_keys",
        ("batch_size", "minibatch_size"),
        ("labels", "minibatch_labels"),
        ("max_samples_per_epoch", "total_samples"),
        "class_lengths", ("offset", "minibatch_offset"))
    launcher.workflow.decision.link_attrs(
        new_loader, "minibatch_class", "last_minibatch",
        "minibatch_size", "class_lengths", "epoch_ended", "epoch_number")

    # Set normalizer from previous Loader to new one:
    new_loader._normalizer = normalizer

    # Set labels_mapping and class_keys in Evaluator to correct writting the
    # results:
    launcher.workflow.evaluator.labels_mapping = labels_mapping
    launcher.workflow.evaluator.class_keys = new_loader.class_keys

    # Initialize and run new workflow:
    launcher.boot()

    # Write results:
    launcher.workflow.write_results(file=os.path.join(data_path, "result.txt"))
