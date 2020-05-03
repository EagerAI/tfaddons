
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


## Usage: the basics

Here's how to build a sequential model with ```keras``` using additional features from ```tfaddons``` package.

```

library(keras)
library(tfaddons)

mnist = dataset_mnist()

x_train <- mnist$train$x
y_train <- mnist$train$y

# reshape the dataset
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))

# Transform RGB values into [0,1] range
x_train <- x_train / 255

y_train <- to_categorical(y_train, 10)

# Build a sequential model
model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                #apply activation gelu
                activation = activation_gelu) %>% 
  # apply group normalization layer
  layer_group_normalization(groups = 5, axis = 3) %>% 
  layer_flatten() %>% 
  layer_dense(10, activation='softmax')

model %>% compile(
  # apply rectified adam
  optimizer = optimizer_radam(),
  # apply sparse max loss
  loss = loss_sparsemax(),
  # choose cohen kappa metric
  metrics = metric_cohen_kappa(10)
)

model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 1,
  validation_split = 0.2
)

```


```
Train on 48000 samples, validate on 12000 samples
48000/48000 [==============================] - 24s 510us/sample - loss: 0.1193 - cohen_kappa: 0.8074 - 
val_loss: 0.0583 - val_cohen_kappa: 0.9104
```

## Callbacks

One can stop training after certain time. For this purpose, ```seconds``` parameter should be set in ```callback_time_stopping``` function:

```
model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 1,
  validation_split = 0.2
)
```

```
Timed stopping at epoch 1 after training for 0:00:06
```




