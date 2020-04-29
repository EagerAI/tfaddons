#' @title LSTM cell with layer normalization and recurrent dropout.
#'
#'
#' @details This class adds layer normalization and recurrent dropout to a LSTM unit. Layer
#' normalization implementation is based on: https://arxiv.org/abs/1607.06450.
#' "Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton and is
#' applied before the internal nonlinearities.
#' Recurrent dropout is based on: https://arxiv.org/abs/1603.05118
#' "Recurrent Dropout without Memory Loss" Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
#'
#' @param object Model or layer object
#'
#' @param units Positive integer, dimensionality of the output space.
#' @param activation Activation function to use. Default: hyperbolic tangent (`tanh`). If
#' you pass `NULL`, no activation is applied (ie. "linear" activation: `a(x) = x`).
#' @param recurrent_activation Activation function to use for the recurrent step.
#' Default: sigmoid (`sigmoid`). If you pass `NULL`, no activation is applied
#' (ie. "linear" activation: `a(x) = x`).
#' @param use_bias Boolean, whether the layer uses a bias vector.
#' @param kernel_initializer Initializer for the `kernel` weights matrix, used for the
#' linear transformation of the inputs.
#' @param recurrent_initializer Initializer for the `recurrent_kernel` weights matrix,
#' used for the linear transformation of the recurrent state.
#' @param bias_initializer Initializer for the bias vector.
#' @param unit_forget_bias Boolean. If True, add 1 to the bias of the forget gate at initialization.
#' Setting it to true will also force `bias_initializer="zeros"`. This is
#' recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
#' @param kernel_regularizer Regularizer function applied to the `kernel` weights matrix.
#' @param recurrent_regularizer Regularizer function applied to the `recurrent_kernel` weights matrix.
#' @param bias_regularizer Regularizer function applied to the bias vector.
#' @param kernel_constraint Constraint function applied to the `kernel` weights matrix.
#' @param recurrent_constraint Constraint function applied to the `recurrent_kernel` weights matrix.
#' @param bias_constraint Constraint function applied to the bias vector.
#' @param dropout Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
#' @param recurrent_dropout Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
#' @param norm_gamma_initializer Initializer for the layer normalization gain initial value.
#' @param norm_beta_initializer Initializer for the layer normalization shift initial value.
#' @param norm_epsilon Float, the epsilon value for normalization layers.
#'
#'
#'
#'
#' @param ... List, the other keyword arguments for layer creation.
#'
#'
#'
#'
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_norm_lstm_cell <- function(object,
                                 units,
                                 activation = 'tanh',
                                 recurrent_activation = 'sigmoid',
                                 use_bias = TRUE,
                                 kernel_initializer = 'glorot_uniform',
                                 recurrent_initializer = 'orthogonal',
                                 bias_initializer = 'zeros',
                                 unit_forget_bias = TRUE,
                                 kernel_regularizer = NULL,
                                 recurrent_regularizer = NULL,
                                 bias_regularizer = NULL,
                                 kernel_constraint = NULL,
                                 recurrent_constraint = NULL,
                                 bias_constraint = NULL,
                                 dropout = 0.,
                                 recurrent_dropout = 0.,
                                 norm_gamma_initializer = 'ones',
                                 norm_beta_initializer = 'zeros',
                                 norm_epsilon = 1e-3,
                                 ...) {
  args = list(units = as.integer(units),
              activation = activation,
              recurrent_activation = recurrent_activation,
              use_bias = use_bias,
              kernel_initializer = kernel_initializer,
              recurrent_initializer = recurrent_initializer,
              bias_initializer = bias_initializer,
              unit_forget_bias = unit_forget_bias,
              kernel_regularizer = kernel_regularizer,
              recurrent_regularizer = recurrent_regularizer,
              bias_regularizer = bias_regularizer,
              kernel_constraint = kernel_constraint,
              recurrent_constraint = recurrent_constraint,
              bias_constraint = bias_constraint,
              dropout = dropout,
              recurrent_dropout = recurrent_dropout,
              norm_gamma_initializer = norm_gamma_initializer,
              norm_beta_initializer = norm_beta_initializer,
              norm_epsilon = norm_epsilon,
              ...)

  create_layer(tfa$rnn$LayerNormLSTMCell, object, args)


}


#' @title Neural Architecture Search (NAS) recurrent network cell.
#'
#' @details This implements the recurrent cell from the paper: https://arxiv.org/abs/1611.01578
#' Barret Zoph and Quoc V. Le. "Neural Architecture Search with Reinforcement Learning"
#' Proc. ICLR 2017. The class uses an optional projection layer.
#'
#' @param object Model or layer object
#' @param units int, The number of units in the NAS cell.
#' @param projection (optional) int, The output dimensionality for the projection matrices.
#' If None, no projection is performed.
#' @param use_bias (optional) bool, If `TRUE` then use biases within the cell.
#' This is `FALSE` by default.
#' @param kernel_initializer Initializer for kernel weight.
#' @param recurrent_initializer Initializer for recurrent kernel weight.
#' @param projection_initializer Initializer for projection weight, used when projection
#' is not `NULL`.
#' @param bias_initializer Initializer for bias, used when `use_bias` is `TRUE`.
#'
#'
#' @param ... Additional keyword arguments.
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_nas_cell <- function(object,
                           units,
                           projection = NULL,
                           use_bias = FALSE,
                           kernel_initializer = 'glorot_uniform',
                           recurrent_initializer = 'glorot_uniform',
                           projection_initializer = 'glorot_uniform',
                           bias_initializer = 'zeros',
                           ...) {
  args = list(
    units = as.integer(units),
    projection = projection,
    use_bias = use_bias,
    kernel_initializer = kernel_initializer,
    recurrent_initializer = recurrent_initializer,
    projection_initializer = projection_initializer,
    bias_initializer = bias_initializer,
    ...
  )

  create_layer(tfa$rnn$NASCell, object, args)


}










