Feed-forward units
******************

* All-to-All perceptron layers (:mod:`veles.znicz.all2all`).
* Activation functions (:mod:`veles.znicz.activation`).
* Convolutional layers (:mod:`veles.znicz.conv`).
* Pooling layers (:mod:`veles.znicz.pooling`).
* Evaluators (:mod:`veles.znicz.evaluator`), softmax and MSE are implemented.


Gradient descent units
**********************

This units calculate gradient descent via back-propagation of gradient signals.
For **each** feed-forward layer should be a coupled gradient descent unit.

* GD for perceptron layers (:mod:`veles.znicz.gd`).
* GD for activation functions (:class:`veles.znicz.activation.ActivationBackward`).
* GD for convolutional layers (:mod:`veles.znicz.gd_conv`).
* GD for pooling layers (:mod:`veles.znicz.gd_pooling`).