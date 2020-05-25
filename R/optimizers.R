#' @title Lazy Adam
#'
#' @param learning_rate A Tensor or a floating point value. or a schedule that is a tf.keras.optimizers.schedules.LearningRateSchedule The learning rate.
#' @param beta_1 A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
#' @param beta_2 A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
#' @param epsilon A small constant for numerical stability. This epsilon is "epsilon hat" in Adam: A Method for Stochastic Optimization. Kingma et al., 2014 (in the formula just before Section 2.1), not the epsilon in Algorithm 1 of the paper.
#' @param amsgrad boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond". Note that this argument is currently not supported and the argument can only be False.
#' @param name Optional name for the operations created when applying gradients. Defaults to "LazyAdam".
#' @param clipnorm is clip gradients by norm;
#' @param clipvalue is clip gradients by value,
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#' @return Optimizer for use with `keras::compile()`
#'
#' @examples
#'
#' \dontrun{
#' keras_model_sequential() %>%
#'   layer_dense(32, input_shape = c(784)) %>%
#'   compile(
#'     optimizer = optimizer_lazy_adam(),
#'     loss='binary_crossentropy',
#'     metrics='accuracy'
#'   )
#' }
#'
#'
#' @export
optimizer_lazy_adam <- function(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon  = 1e-7,
                          amsgrad = FALSE, name = "LazyAdam", clipnorm = NULL, clipvalue = NULL,
                          decay = NULL, lr = NULL) {

  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    amsgrad = amsgrad,
    name = name,
    clipnorm = clipnorm,
    clipvalue = clipvalue,
    decay = decay,
    lr = lr
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  args$decay <- decay
  args$lr <- lr

  do.call(tfa$optimizers$LazyAdam, args)
}

attr(optimizer_lazy_adam, "py_function_name") <- "lazy_adam"

#' @title Conditional Gradient
#'
#' @param learning_rate A Tensor or a floating point value, or a schedule that is a tf$keras$optimizers$schedules$LearningRateSchedule The learning rate.
#' @param lambda_ A Tensor or a floating point value. The constraint.
#' @param epsilon A Tensor or a floating point value. A small constant for numerical stability when handling the case of norm of gradient to be zero.
#' @param use_locking If True, use locks for update operations.
#' @param name Optional name prefix for the operations created when applying gradients. Defaults to 'ConditionalGradient'.
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#' @return Optimizer for use with `keras::compile()`
#' @export
optimizer_conditional_gradient <- function(learning_rate, lambda_, epsilon = 1e-07, use_locking = FALSE,
                          name = 'ConditionalGradient',
                          clipnorm = NULL, clipvalue = NULL,
                          decay = NULL, lr = NULL) {

  args <- list(
    learning_rate = learning_rate,
    lambda_ = lambda_,
    epsilon = epsilon,
    use_locking = use_locking,
    name = name,
    clipnorm = clipnorm,
    clipvalue = clipvalue,
    decay = decay,
    lr = lr
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  args$decay <- decay
  args$lr <- lr

  do.call(tfa$optimizers$ConditionalGradient, args)
}

attr(optimizer_conditional_gradient, "py_function_name") <- "conditional_gradient"


#' @title Layer-wise Adaptive Moments
#'
#' @param learning_rate A `Tensor` or a floating point value. or a schedule that is a `tf$keras$optimizers$schedules$LearningRateSchedule` The learning rate.
#' @param beta_1 A `float` value or a constant `float` tensor. The exponential decay rate for the 1st moment estimates.
#' @param beta_2 A `float` value or a constant `float` tensor. The exponential decay rate for the 2nd moment estimates.
#' @param epsilon A small constant for numerical stability.
#' @param weight_decay_rate weight decay rate.
#' @param exclude_from_weight_decay List of regex patterns of variables excluded from weight decay. Variables whose name contain a substring matching the pattern will be excluded.
#' @param exclude_from_layer_adaptation List of regex patterns of variables excluded from layer adaptation. Variables whose name contain a substring matching the pattern will be excluded.
#' @param name Optional name for the operations created when applying gradients. Defaults to "LAMB".
#'
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#' @return Optimizer for use with `keras::compile()`
#'
#' @examples
#'
#' \dontrun{
#' keras_model_sequential() %>%
#'   layer_dense(32, input_shape = c(784)) %>%
#'   compile(
#'     optimizer = optimizer_lamb(),
#'     loss='binary_crossentropy',
#'     metrics='accuracy'
#'   )
#' }
#'
#'
#' @export
optimizer_lamb <- function(learning_rate = 0.001,
                           beta_1 = 0.9,
                           beta_2 = 0.999,
                           epsilon = 1e-6,
                           weight_decay_rate = 0.0,
                           exclude_from_weight_decay = NULL,
                           exclude_from_layer_adaptation = NULL,
                           name = "LAMB",
                           clipnorm = NULL, clipvalue = NULL,
                           decay = NULL, lr = NULL) {

  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    weight_decay_rate = weight_decay_rate,
    exclude_from_weight_decay = exclude_from_weight_decay,
    exclude_from_layer_adaptation = exclude_from_layer_adaptation,
    name = name,

    clipnorm = clipnorm,
    clipvalue = clipvalue,
    decay = decay,
    lr = lr
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  args$decay <- decay
  args$lr <- lr

  do.call(tfa$optimizers$LAMB, args)
}

attr(optimizer_lamb, "py_function_name") <- "lamb"


#' @title NovoGrad
#'
#' @param learning_rate A `Tensor` or a floating point value. or a schedule that is a `tf$keras$optimizers$schedules$LearningRateSchedule` The learning rate.
#' @param beta_1 A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
#' @param beta_2 A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
#' @param epsilon A small constant for numerical stability.
#' @param weight_decay A floating point value. Weight decay for each param.
#' @param amsgrad boolean. Whether to apply AMSGrad variant of this algorithm from the paper "On the Convergence of Adam and beyond"
#' @param grad_averaging determines whether to use Adam style exponential moving averaging for the first order moments.
#' @param name Optional name for the operations created when applying gradients. Defaults to "NovoGrad".
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#' @return Optimizer for use with `keras::compile()`
#'
#' @examples
#'
#' \dontrun{
#' keras_model_sequential() %>%
#'   layer_dense(32, input_shape = c(784)) %>%
#'   compile(
#'     optimizer = optimizer_novograd(),
#'     loss='binary_crossentropy',
#'     metrics='accuracy'
#'   )
#' }
#'
#' @export
optimizer_novograd <- function(learning_rate = 0.001,
                               beta_1 = 0.9,
                               beta_2 = 0.999,
                               epsilon = 1e-7,
                               weight_decay = 0.0,
                               grad_averaging = FALSE,
                               amsgrad = FALSE,
                               name = "NovoGrad",
                               clipnorm = NULL, clipvalue = NULL,
                               decay = NULL, lr = NULL) {

  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    weight_decay = weight_decay,
    grad_averaging = grad_averaging,
    amsgrad = amsgrad,
    name = name,

    clipnorm = clipnorm,
    clipvalue = clipvalue,
    decay = decay,
    lr = lr
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  args$decay <- decay
  args$lr <- lr

  do.call(tfa$optimizers$NovoGrad, args)
}

attr(optimizer_novograd, "py_function_name") <- "novograd"

#' @title Rectified Adam (a.k.a. RAdam)
#'
#' @param learning_rate A `Tensor` or a floating point value. or a schedule that is
#' a `tf$keras$optimizers$schedules$LearningRateSchedule` The learning rate.
#' @param beta_1 A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
#' @param beta_2 A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
#' @param epsilon A small constant for numerical stability.
#' @param weight_decay A floating point value. Weight decay for each param.
#' @param amsgrad boolean. Whether to apply AMSGrad variant of this algorithm from the paper
#' "On the Convergence of Adam and beyond".
#' @param sma_threshold A float value. The threshold for simple mean average.
#' @param total_steps An integer. Total number of training steps. Enable warmup by setting a positive value.
#' @param warmup_proportion A floating point value. The proportion of increasing steps.
#' @param min_lr A floating point value. Minimum learning rate after warmup.
#' @param name Optional name for the operations created when applying gradients. Defaults to "RectifiedAdam".
#'
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#'
#' @return Optimizer for use with `keras::compile()`
#' @export
optimizer_radam <- function(learning_rate = 0.001,
                               beta_1 = 0.9,
                               beta_2 = 0.999,
                               epsilon = 1e-7,
                               weight_decay = 0.0,
                               amsgrad = FALSE,
                               sma_threshold = 5.0,
                               # float for total_steps is here to be able to load models created before
                               # https://github.com/tensorflow/addons/pull/1375 was merged. It should be
                               # removed for Addons 0.11.
                               total_steps = 0,
                               warmup_proportion = 0.1,
                               min_lr = 0.0,
                               name = "RectifiedAdam",
                               clipnorm = NULL, clipvalue = NULL,
                               decay = NULL, lr = NULL) {

  args <- list(
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    weight_decay = weight_decay,
    amsgrad = amsgrad,
    sma_threshold = sma_threshold,
    # float for total_steps is here to be able to load models created before
    # https://github.com/tensorflow/addons/pull/1375 was merged. It should be
    # removed for Addons 0.11.
    total_steps = total_steps,
    warmup_proportion = warmup_proportion,
    min_lr = min_lr,
    name = name,

    clipnorm = clipnorm,
    clipvalue = clipvalue,
    decay = decay,
    lr = lr
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  args$decay <- decay
  args$lr <- lr

  do.call(tfa$optimizers$RectifiedAdam, args)
}

attr(optimizer_radam, "py_function_name") <- "radam"



#' @title Yogi
#'
#' @param learning_rate A Tensor or a floating point value. The learning rate.
#' @param beta1 A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
#' @param beta2 A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
#' @param epsilon A constant trading off adaptivity and noise.
#' @param l1_regularization_strength A float value, must be greater than or equal to zero.
#' @param l2_regularization_strength A float value, must be greater than or equal to zero.
#' @param initial_accumulator_value The starting value for accumulators. Only positive values are allowed.
#' @param activation Use hard sign or soft tanh to determin sign.
#' @param name Optional name for the operations created when applying gradients. Defaults to "Yogi".
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#'
#' @return Optimizer for use with `keras::compile()`
#' @export
optimizer_yogi <- function(learning_rate = 0.01,
                           beta1 = 0.9,
                           beta2 = 0.999,
                           epsilon = 1e-3,
                           l1_regularization_strength = 0.0,
                           l2_regularization_strength = 0.0,
                           initial_accumulator_value = 1e-6,
                           activation = "sign",
                           name = "Yogi",
                           clipnorm = NULL, clipvalue = NULL,
                           decay = NULL, lr = NULL) {

  args <- list(
    learning_rate = learning_rate,
    beta1 = beta1,
    beta2 = beta2,
    epsilon = epsilon,
    l1_regularization_strength = l1_regularization_strength,
    l2_regularization_strength = l2_regularization_strength,
    initial_accumulator_value = initial_accumulator_value,
    activation = activation,
    name = name,

    clipnorm = clipnorm,
    clipvalue = clipvalue,
    decay = decay,
    lr = lr
  )
  args$clipnorm <- clipnorm
  args$clipvalue <- clipvalue
  args$decay <- decay
  args$lr <- lr

  do.call(tfa$optimizers$Yogi, args)
}

attr(optimizer_yogi, "py_function_name") <- "yogi"

