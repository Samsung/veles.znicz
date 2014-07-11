from veles.config import root
from veles.znicz.tests.research.imagenet.imagenet_ae import (LR, LRB,
                                                             WD, WDB,
                                                             GM, GMB)


root.update = {
    "snapshotter": {"prefix": "imagenet_ae_6_2"},
    "imagenet": {"layers":
                 [{"type": "conv", "n_kernels": 48,
                   "kx": 6, "ky": 6, "sliding": (2, 2),
                   "learning_rate": LR,
                   "weights_decay": WD,
                   "gradient_moment": GM},

                  {"type": "softmax", "output_shape": 5,
                   "learning_rate": LR, "learning_rate_bias": LRB,
                   "weights_decay": WD, "weights_decay_bias": WDB,
                   "gradient_moment": GM, "gradient_moment_bias": GMB}]}
}
