#' @title Bernoulli sample
#'
#' @description Samples from Bernoulli distribution.
#'
#'
#' @param probs probabilities
#' @param logits logits
#' @param dtype the data type
#' @param sample_shape a list/vector of integers
#' @param seed integer, random seed
#' @return a Tensor
#' @export
sample_bernoulli <- function(probs = NULL, logits = NULL,
                             dtype = tf$int32,
                             sample_shape = list(),
                             seed = NULL) {

  args <- list(
    probs = probs,
    logits = logits,
    dtype = dtype,
    sample_shape = sample_shape,
    seed = seed
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  do.call(tfa$seq2seq$sampler$bernoulli_sample, args)

}

#' @title Categorical sample
#'
#' @description Samples from categorical distribution.
#'
#'
#' @param logits logits
#' @param dtype dtype
#' @param sample_shape sample_shape
#' @param seed seed
#'
#' @export
sample_categorical <- function(logits, dtype = tf$int32, sample_shape = list(), seed = NULL) {

  args <- list(
    logits = logits,
    dtype = dtype,
    sample_shape = sample_shape,
    seed = seed
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  do.call(tfa$seq2seq$sampler$categorical_sample, args)

}





