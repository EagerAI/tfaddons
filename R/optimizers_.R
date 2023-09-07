#' @title Lookahead mechanism
#' @details The mechanism is proposed by Michael R. Zhang et.al in the paper
#' [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610v1).
#' The optimizer iteratively updates two sets of weights: the search directions for weights
#' are chosen by the inner optimizer, while the "slow weights" are updated each k steps based
#' on the directions of the "fast weights" and the two sets of weights are synchronized.
#' This method improves the learning stability and lowers the variance of its inner optimizer.
#'
#'
#' @param optimizer The original optimizer that will be used to compute and apply the gradients.
#' @param sync_period An integer. The synchronization period of lookahead. Enable lookahead mechanism
#' by setting it with a positive value.
#' @param slow_step_size A floating point value. The ratio for updating the slow weights.
#' @param name Optional name for the operations created when applying gradients. Defaults to "Lookahead".
#'
#' @param clipnorm is clip gradients by norm.
#' @param clipvalue is clip gradients by value.
#' @param decay is included for backward compatibility to allow time inverse decay of learning rate.
#' @param lr is included for backward compatibility, recommended to use learning_rate instead.
#'
#' @examples
#'
#' \dontrun{
#'
#' opt = tf$keras$optimizers$SGD(learning_rate)
#' opt = lookahead_mechanism(opt)
#'
#' }
#' @return Optimizer for use with `keras::compile()`
#'
#' @export
lookahead_mechanism <- function(optimizer,
                                sync_period = 6,
                                slow_step_size = 0.5,
                                name = "Lookahead",
                                clipnorm = NULL, clipvalue = NULL,
                                decay = NULL, lr = NULL) {

  args = list(
    optimizer = optimizer,
    sync_period = as.integer(sync_period),
    slow_step_size = slow_step_size,
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

  do.call(tfa$optimizers$Lookahead, args)

}


#' @title Moving Average
#' @details Optimizer that computes a moving average of the variables.
#' Empirically it has been found that using the moving average of the
#' trained parameters of a deep network is better than using its trained
#' parameters directly. This optimizer allows you to compute this moving
#' average and swap the variables at save time so that any code outside
#' of the training loop will use by default the average values
#' instead of the original ones.
#'
#'
#' @param optimizer str or tf$keras$optimizers$Optimizer that will be used to compute
#' and apply gradients.
#' @param average_decay float. Decay to use to maintain the moving averages of trained variables.
#' @param num_updates Optional count of the number of updates applied to variables.
#' @param name Optional name for the operations created when applying gradients.
#' Defaults to "MovingAverage".
#'
#' @param dynamic_decay bool. Whether to change the decay based on the number of optimizer updates. Decay will start at 0.1 and gradually increase up to average_decay after each optimizer update.
#' @param start_step int. What step to start the moving average.
#' @param ... keyword arguments. Allowed to be {clipnorm, clipvalue, lr, decay}. clipnorm is clip gradients by norm; clipvalue is clip gradients by value, decay is included for backward compatibility to allow time inverse decay of learning rate. lr is included for backward compatibility, recommended to use learning_rate instead.
#' @examples
#'
#' \dontrun{
#'
#' opt = tf$keras$optimizers$SGD(learning_rate)
#' opt = moving_average(opt)
#'
#' }
#'
#' @return Optimizer for use with `keras::compile()`
#' @export
optimizer_moving_average <- function(optimizer,
                                     average_decay = 0.99,
                                     num_updates = NULL,
                                     start_step = 0,
                                     dynamic_decay = FALSE,
                                     name = 'MovingAverage',
                                     ...) {

  args = list(
    optimizer = optimizer,
    average_decay = average_decay,
    num_updates = num_updates,
    start_step = as.integer(start_step),
    dynamic_decay = dynamic_decay,
    name = name,
    ...

  )

  if(is.null(num_updates)) {
    args$num_updates <- NULL
  } else {
    args$num_updates <- as.integer(args$num_updates)
  }

  do.call(tfa$optimizers$MovingAverage, args)


}



#' @title Stochastic Weight Averaging
#'
#' @details The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov et. al
#' in the paper [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407). The
#' optimizer implements averaging of multiple points along the trajectory of SGD. The optimizer
#' expects an inner optimizer which will be used to apply the gradients to the variables and
#' itself computes a running average of the variables every k steps (which generally corresponds
#' to the end of a cycle when a cyclic learning rate is employed). We also allow the specification
#' of the number of steps averaging should first happen after. Let's say, we want averaging
#' to happen every k steps after the first m steps. After step m we'd take a snapshot of
#' the variables and then average the weights appropriately at step m + k, m + 2k and so on.
#' The assign_average_vars function can be called at the end of training to obtain the
#' averaged_weights from the optimizer.
#'
#'
#' @param optimizer The original optimizer that will be used to compute and apply the gradients.
#' @param start_averaging An integer. Threshold to start averaging using SWA. Averaging only occurs
#' at start_averaging iters, must be >= 0. If start_averaging = m, the first snapshot will be taken
#' after the mth application of gradients (where the first iteration is iteration 0).
#' @param average_period An integer. The synchronization period of SWA. The averaging occurs every
#' average_period steps. Averaging period needs to be >= 1.
#' @param name Optional name for the operations created when applying gradients. Defaults to 'SWA'.
#' @param ... keyword arguments. Allowed to be {clipnorm, clipvalue, lr, decay}. clipnorm is clip gradients by norm; clipvalue is clip gradients by value, decay is included for backward compatibility to allow time inverse decay of learning rate. lr is included for backward compatibility, recommended to use learning_rate instead.
#'
#' @examples
#'
#' \dontrun{
#' opt = tf$keras$optimizers$SGD(learning_rate)
#' opt = optimizer_swa(opt, start_averaging=m, average_period=k)
#' }
#'
#' @return Optimizer for use with `keras::compile()`
#' @export
optimizer_swa <- function(optimizer,
                          start_averaging = 0,
                          average_period = 10,
                          name = 'SWA',
                          ...) {

  args = list(
    optimizer = optimizer,
    start_averaging = as.integer(start_averaging),
    average_period = as.integer(average_period),
    name = name,
    ...
  )

  do.call(tfa$optimizers$SWA, args)

}







