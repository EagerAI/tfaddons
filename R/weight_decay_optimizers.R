#' @title Optimizer that implements the Adam algorithm with weight decay
#' @description This is an implementation of the AdamW optimizer described in "Decoupled Weight Decay Regularization"
#' by Loshchilov & Hutter (https://arxiv.org/abs/1711.05101) ([pdf])(https://arxiv.org/pdf/1711.05101.pdf). It computes
#' the update step of tf.keras.optimizers.Adam and additionally decays the variable. Note that this is different
#' from adding L2 regularization on the variables to the loss: it regularizes variables with large gradients more than
#' L2 regularization would, which was shown to yield better training loss and generalization error in the paper above.
#'
#'
#' @param weight_decay A Tensor or a floating point value. The weight decay.
#' @param learning_rate A Tensor or a floating point value. The learning rate.
#' @param beta_1 A float value or a constant float tensor. The exponential decay rate for the 1st moment estimates.
#' @param beta_2 A float value or a constant float tensor. The exponential decay rate for the 2nd moment estimates.
#' @param epsilon A small constant for numerical stability. This epsilon is "epsilon hat" in
#' the Kingma and Ba paper (in the formula just before Section 2.1),
#' not the epsilon in Algorithm 1 of the paper.
#' @param amsgrad boolean. Whether to apply AMSGrad variant of this algorithm from the paper
#' "On the Convergence of Adam and beyond".
#' @param name Optional name for the operations created when applying
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#' @return Optimizer for use with `keras::compile()`
#'
#' @examples
#'
#' \dontrun{
#'
#' step = tf$Variable(0L, trainable = FALSE)
#' schedule = tf$optimizers$schedules$PiecewiseConstantDecay(list(c(10000, 15000)),
#' list(c(1e-0, 1e-1, 1e-2)))
#  lr and wd can be a function or a tensor
#' lr = 1e-1 * schedule(step)
#' wd = lambda: 1e-4 * schedule(step)
#'
#' }
#'
#' @export
optimizer_decay_adamw <- function(weight_decay,
                            learning_rate = 0.001,
                            beta_1 = 0.9,
                            beta_2 = 0.999,
                            epsilon=  1e-07,
                            amsgrad = FALSE,
                            name = "AdamW",
                            clipnorm = NULL, clipvalue = NULL,
                            decay = NULL, lr = NULL) {

  args = list(
    weight_decay = weight_decay,
    learning_rate = learning_rate,
    beta_1 = beta_1,
    beta_2 = beta_2,
    epsilon = epsilon,
    amsgrad = FALSE,
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

  do.call(tfa$optimizers$AdamW, args)

}

attr(optimizer_decay_adamw, "py_function_name") <- "decay_adamw"

#' @title Optimizer that implements the Momentum algorithm with weight_decay
#' @description This is an implementation of the SGDW optimizer described in "Decoupled Weight Decay Regularization"
#' by Loshchilov & Hutter (https://arxiv.org/abs/1711.05101) ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
#' It computes the update step of tf.keras.optimizers.SGD and additionally decays the variable. Note that this
#' is different from adding L2 regularization on the variables to the loss. Decoupling the weight decay from other
#' hyperparameters (in particular the learning rate) simplifies hyperparameter search. For further information see
#' the documentation of the SGD Optimizer.
#'
#' @param weight_decay weight decay rate.
#' @param learning_rate float hyperparameter >= 0. Learning rate.
#' @param momentum float hyperparameter >= 0 that accelerates SGD in the relevant direction and dampens oscillations.
#' @param nesterov boolean. Whether to apply Nesterov momentum.
#' @param name Optional name prefix for the operations created when applying gradients. Defaults to 'SGD'.
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#' @return Optimizer for use with `keras::compile()`
#'
#' @examples
#'
#' \dontrun{
#'
#' step = tf$Variable(0L, trainable = FALSE)
#' schedule = tf$optimizers$schedules$PiecewiseConstantDecay(list(c(10000, 15000)),
#' list(c(1e-0, 1e-1, 1e-2)))
#  lr and wd can be a function or a tensor
#' lr = 1e-1 * schedule(step)
#' wd = lambda: 1e-4 * schedule(step)
#'
#' }
#'
#' @export
optimizer_decay_sgdw <- function(weight_decay,
                            learning_rate = 0.001,
                            momentum = 0.0,
                            nesterov = FALSE,
                            name = 'SGDW',
                            clipnorm = NULL, clipvalue = NULL,
                            decay = NULL, lr = NULL) {

  args = list(
    weight_decay = weight_decay,
    learning_rate = learning_rate,
    momentum = momentum,
    nesterov = nesterov,
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

  do.call(tfa$optimizers$SGDW, args)

}

attr(optimizer_decay_sgdw, "py_function_name") <- "decay_sgdw"


#' @title Factory function returning an optimizer class with decoupled weight decay
#'
#' @details The API of the new optimizer class slightly differs from the API of the base optimizer:
#'
#' - The first argument to the constructor is the weight decay rate.
#' - minimize and apply_gradients accept the optional keyword argument decay_var_list,
#' which specifies the variables that should be decayed. If NULLs, all variables that are optimized are decayed.
#'
#' @param base_optimizer An optimizer class that inherits from tf$optimizers$Optimizer.
#'
#'
#' @note Note: this extension decays weights BEFORE applying the update based
#' on the gradient, i.e. this extension only has the desired behaviour for
#' optimizers which do not depend on the value of 'var' in the update step!
#' Note: when applying a decay to the learning rate, be sure to manually apply
#' the decay to the `weight_decay` as well.
#'
#' @return A new optimizer class that inherits from DecoupledWeightDecayExtension and base_optimizer.
#'
#' @examples
#'
#' \dontrun{
#'
#' ### MyAdamW is a new class
#' MyAdamW = extend_with_decoupled_weight_decay(tf$keras$optimizers$Adam)
#' ### Create a MyAdamW object
#' optimizer = MyAdamW(weight_decay = 0.001, learning_rate = 0.001)
#' #### update var1, var2 but only decay var1
#' optimizer$minimize(loss, var_list = list(var1, var2), decay_variables = list(var1))
#'
#' }
#'
#' @export
extend_with_decoupled_weight_decay <- function(base_optimizer) {
  tfa$optimizers$extend_with_decoupled_weight_decay(base_optimizer = base_optimizer)
}


