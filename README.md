
## R interface to useful extra functionality for TensorFlow 2.x by SIG-addons


The `tfaddons` package provides R wrappers to [TensorFlow Addons](https://www.tensorflow.org/addons).

__TensorFlow Addons__ is a repository of contributions that conform to well-established API patterns, but implement new functionality not available in core TensorFlow. TensorFlow natively supports a large number of operators, layers, metrics, losses, and optimizers. However, in a fast moving field like ML, there are many interesting new developments that cannot be integrated into core TensorFlow (because their broad applicability is not yet clear, or it is mostly used by a smaller subset of the community).

[![Actions Status](https://github.com/henry090/tfaddons/workflows/R-CMD/badge.svg)](https://github.com/henry090/tfaddons)

<img src="images/tfaddons.png" width=200 align=right style="margin-left: 15px;" alt="Keras Tuner"/>

Addons provide the following features which are compatible with ```keras``` library.

- activations
- callbacks
- image
- layers
- losses
- metrics
- optimizers
- rnn
- seq2seq
- text

Currently, the package is under development. But it could be installed from github:

```
devtools::install_github('henry090/tfaddons')

```





