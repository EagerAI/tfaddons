#' @title Gelu
#'
#' @description Gaussian Error Linear Unit.
#'
#' @details Computes gaussian error linear:
#' `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` or
#' `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`, where P(X) ~ N(0, 1),
#' depending on whether approximation is enabled.
#' See [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
#' and [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805).
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#' @param approximate bool, whether to enable approximation. Returns: A `Tensor`. Has the same type as `x`.
#'
#' @return A `Tensor`. Has the same type as `x`.
#'
#' @section Computes gaussian error linear:
#' `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))` or `x * P(X <= x) = 0.5 * x * (1 + erf(x / sqrt(2)))`,
#' where P(X) ~ N(0, 1), depending on whether approximation is enabled.
#'
#' @examples
#'
#' \dontrun{
#' library(keras)
#' library(tfaddons)
#' model = keras_model_sequential() %>%
#' layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
#'               activation = activation_gelu)
#' }
#'
#'
#' @export
activation_gelu <- function(x, approximate = TRUE) {

  args <- list(
    x = x,
    approximate = approximate
  )

  do.call(tfa$activations$gelu, args)

}

attr(activation_gelu, "py_function_name") <- "gelu"

#' @title Hardshrink
#'
#' @description Hard shrink function.
#'
#' @details Computes hard shrink function:
#' `x if x < lower or x > upper else 0`.
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#' @param lower `float`, lower bound for setting values to zeros.
#' @param upper `float`, upper bound for setting values to zeros. Returns: A `Tensor`. Has the same type as `x`.
#'
#' @return A `Tensor`. Has the same type as `x`.
#'
#' @section Computes hard shrink function:
#' `x if x < lower or x > upper else 0`.
#'
#' @examples
#'
#' \dontrun{
#' library(keras)
#' library(tfaddons)
#' model = keras_model_sequential() %>%
#' layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
#'               activation = activation_hardshrink)
#' }
#'
#' @export
activation_hardshrink <- function(x, lower = -0.5, upper = 0.5) {

  args <- list(
    x = x,
    lower = lower,
    upper = upper
  )

  do.call(tfa$activations$hardshrink, args)

}

attr(activation_hardshrink, "py_function_name") <- "hardshrink"

#' @title Lisht
#'
#' @description LiSHT: Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function.
#'
#' @details Computes linearly scaled hyperbolic tangent (LiSHT): `x * tanh(x)`
#' See [LiSHT: Non-Parameteric Linearly Scaled Hyperbolic Tangent Activation Function for Neural Networks](https://arxiv.org/abs/1901.05894).
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#'
#' @return A `Tensor`. Has the same type as `x`.
#'
#' @examples
#'
#' \dontrun{
#' library(keras)
#' library(tfaddons)
#' model = keras_model_sequential() %>%
#' layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
#'               activation = activation_lisht)
#' }
#'
#' @export
activation_lisht <- function(x) {

  args <- list(
    x = x
  )

  do.call(tfa$activations$lisht, args)

}

attr(activation_lisht, "py_function_name") <- "lisht"

#' @title Mish
#'
#' @description Mish: A Self Regularized Non-Monotonic Neural Activation Function.
#'
#' @details Computes mish activation: x * tanh(softplus(x))
#' See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#' Returns: A `Tensor`. Has the same type as `x`.
#'
#' @return A `Tensor`. Has the same type as `x`.
#'
#' @export
activation_mish <- function(x) {

  args <- list(
    x = x
  )

  do.call(tfa$activations$mish, args)

}

attr(activation_mish, "py_function_name") <- "mish"


#' @title Rrelu
#'
#' @description rrelu function.
#'
#' @details Computes rrelu function:
#' `x if x > 0 else random(lower, upper) * x` or
#' `x if x > 0 else x * (lower + upper) / 2`
#' depending on whether training is enabled.
#' See [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853).
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#' @param lower `float`, lower bound for random alpha.
#' @param upper `float`, upper bound for random alpha.
#' @param training `bool`, indicating whether the `call` is meant for training or inference.
#' @param seed `int`, this sets the operation-level seed. Returns:
#'
#' @return A `Tensor`. Has the same type as `x`.
#'
#' @section Computes rrelu function:
#' `x if x > 0 else random(lower, upper) * x` or `x if x > 0 else x * (lower + upper) / 2` depending on
#' whether training is enabled.
#'
#' @export
activation_rrelu <- function(x, lower = 0.125, upper = 0.3333333333333333, training = NULL, seed = NULL) {

  args <- list(
    x = x,
    lower = lower,
    upper = upper,
    training = training,
    seed = seed
  )

  if (!is.null(args$seed)) {
    args$seed <- as.integer(args$seed)
  }

  do.call(tfa$activations$rrelu, args)

}

attr(activation_rrelu, "py_function_name") <- "rrelu"


#' @title Softshrink
#'
#' @description Soft shrink function.
#'
#' @details Computes soft shrink function:
#' `x - lower if x < lower, x - upper if x > upper else 0`.
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#' @param lower `float`, lower bound for setting values to zeros.
#' @param upper `float`, upper bound for setting values to zeros. Returns: A `Tensor`. Has the same type as `x`.
#'
#' @return A `Tensor`. Has the same type as `x`.
#'
#' @section Computes soft shrink function:
#' `x - lower if x < lower, x - upper if x > upper else 0`.
#'
#' @export
activation_softshrink <- function(x, lower = -0.5, upper = 0.5) {

  args <- list(
    x = x,
    lower = lower,
    upper = upper
  )

  do.call(tfa$activations$softshrink, args)

}

attr(activation_softshrink, "py_function_name") <- "softshrink"

#' @title Sparsemax
#'
#' @description Sparsemax activation function [1].
#'
#' @details For each batch `i` and class `j` we have $$sparsemax[i, j] =
#' max(logits[i, j] - tau(logits[i, :]), 0)$$ [1]: https://arxiv.org/abs/1602.02068
#'
#' @param logits Input tensor.
#' @param axis Integer, axis along which the sparsemax operation is applied.
#'
#' @return Tensor, output of sparsemax transformation. Has the same type and shape as
#' `logits`. Raises: ValueError: In case `dim(logits) == 1`.
#'
#' @section Raises:
#' ValueError: In case `dim(logits) == 1`.
#'
#' @export
activation_sparsemax <- function(logits, axis = -1L) {

  args <- list(
    logits = logits,
    axis = axis
  )

  do.call(tfa$activations$sparsemax, args)

}

attr(activation_sparsemax, "py_function_name") <- "sparsemax"

#' @title Tanhshrink
#'
#' @description Applies the element-wise function: x - tanh(x)
#'
#'
#' @param x A `Tensor`. Must be one of the following types: `float16`, `float32`, `float64`.
#'
#' @return A `Tensor`. Has the same type as `features`.
#'
#' @export
activation_tanhshrink <- function(x) {

  args <- list(
    x = x
  )

  do.call(tfa$activations$tanhshrink, args)

}

attr(activation_tanhshrink, "py_function_name") <- "tanhshrink"





