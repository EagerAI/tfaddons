

## R interface to useful extra functionality for TensorFlow 2.x by SIG-addons

The `tfaddons` package provides R wrappers to [TensorFlow Addons](https://www.tensorflow.org/addons).

__TensorFlow Addons__ is a repository of contributions that conform to well-established API patterns, but implement new functionality not available in core TensorFlow. TensorFlow natively supports a large number of operators, layers, metrics, losses, and optimizers. However, in a fast moving field like ML, there are many interesting new developments that cannot be integrated into core TensorFlow (because their broad applicability is not yet clear, or it is mostly used by a smaller subset of the community).

[![Actions Status](https://github.com/eagerai/tfaddons/workflows/TFA_stable/badge.svg)](https://github.com/eagerai/tfaddons)
[![CRAN](https://www.r-pkg.org/badges/version/tfaddons?color=darkgreen)](https://cran.r-project.org/package=tfaddons)
[![Last month downloads](http://cranlogs.r-pkg.org/badges/last-month/tfaddons?color=green)](https://cran.r-project.org/package=tfaddons)
[![Last commit](https://img.shields.io/github/last-commit/eagerai/tfaddons.svg)](https://github.com/eagerai/tfaddons/commits/master)

<img src="images/tfaddons.png" width=200 align=right style="margin-left: 15px;" alt="TF-addons"/>

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

## Installation

Requirements:

- TensorFlow 2.X

The dev version:

```
devtools::install_github('eagerai/tfaddons')
```

Later, you need to install the python module *tensorflow-addons*:

```
tfaddons::install_tfaddons()
```

## Usage: the basics

Here's how to build a sequential model with ```keras``` using additional features from ```tfaddons``` package.

Import and prepare MNIST dataset.

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
```

Using the Sequential API, define the model architecture.

```
# Build a sequential model
model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                #apply activation gelu
                activation = activation_gelu) %>% 
  # apply group normalization layer
  layer_group_normalization(groups = 5, axis = 3) %>% 
  layer_flatten() %>% 
  layer_dense(10, activation='softmax')

# Compile
model %>% compile(
  # apply rectified adam
  optimizer = optimizer_radam(),
  # apply sparse max loss
  loss = loss_sparsemax(),
  # choose cohen kappa metric
  metrics = metric_cohen_kappa(10)
)
```

Train the Keras model.

```
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

Let's apply ```Weight Normalization```, a Simple Reparameterization technique to Accelerate Training of Deep Neural Networks:

> Note: We only change the model architecture and then train our model.

```
# Build a sequential model
model = keras_model_sequential() %>% 
  layer_weight_normalization(input_shape = c(28L,28L,1L),
                             layer_conv_2d(filters = 10, kernel_size = c(3,3))) %>% 
  layer_flatten() %>% 
  layer_weight_normalization(layer_dense(units = 10, activation='softmax'))
```

```
Train on 48000 samples, validate on 12000 samples
48000/48000 [==============================] - 12s 253us/sample - loss: 0.1276 - cohen_kappa: 0.7920 - 
val_loss: 0.0646 - val_cohen_kappa: 0.9044
```

We can see that the training process has finished in *12 seconds*. But without this method, 1 epoch required *24 seconds*.

## Callbacks

One can stop training after certain time. For this purpose, ```seconds``` parameter should be set in ```callback_time_stopping``` function:

```
model %>% fit(
  x_train, y_train,
  batch_size = 128,
  epochs = 4,
  validation_split = 0.2,
  verbose = 0,
  callbacks = callback_time_stopping(seconds = 6, verbose = 1)
)
```

```
Timed stopping at epoch 1 after training for 0:00:06
```

## Losses

```TripletLoss``` can be applied in the following form:

First task is to create a Keras model.

```
model = keras_model_sequential() %>% 
  layer_conv_2d(filters = 64, kernel_size = 2, padding='same', input_shape=c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size=2) %>% 
  layer_flatten() %>% 
  layer_dense(256, activation= NULL) %>% 
  layer_lambda(f = function(x) tf$math$l2_normalize(x, axis = 1L))

model %>% compile(
  optimizer = optimizer_lazy_adam(),
  # apply triplet semihard loss
  loss = loss_triplet_semihard())
```

With ```tfdatasets``` package we can *cast* our dataset and then ```fit```.

```
library(tfdatasets)

train = tensor_slices_dataset(list(tf$cast(x_train,'uint8'),tf$cast( y_train,'int64'))) %>% 
  dataset_shuffle(1024) %>% dataset_batch(32)
  
# fit
model %>% fit(
  train,
  epochs = 1
)
```

```
Train for 1875 steps
1875/1875 [==============================] - 74s 39ms/step - loss: 0.4227
```






