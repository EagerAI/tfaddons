#' @title Time Stopping
#' @details Stop training when a specified amount of time has passed.
#'
#'
#'
#'
#'
#' @param seconds maximum amount of time before stopping. Defaults to 86400 (1 day).
#' @param verbose verbosity mode. Defaults to 0.
#'
#'
#'
#'
#'
#'
#' @return None
#'
#' @export
callback_time_stopping <- function(seconds = 86400, verbose = 0){

  args = list(seconds = as.integer(seconds), verbose = as.integer(verbose))

  do.call(tfa$callbacks$TimeStopping, args)

}


#' @title TQDM Progress Bar
#'
#' @details TQDM Progress Bar for Tensorflow Keras.
#'
#'
#'
#' @param metrics_separator (string) Custom separator between metrics. Defaults to ' - '
#' @param overall_bar_format (string format) Custom bar format for overall (outer) progress
#' bar, see https://github.com/tqdm/tqdm#parameters for more detail.
#' @param epoch_bar_format (string format) Custom bar format for epoch (inner) progress bar,
#' see https://github.com/tqdm/tqdm#parameters for more detail.
#' @param update_per_second (int) Maximum number of updates in the epochs bar per second, this
#' is to prevent small batches from slowing down training. Defaults to 10.
#' @param leave_epoch_progress (bool) TRUE to leave epoch progress bars
#' @param leave_overall_progress (bool) TRUE to leave overall progress bar
#' @param show_epoch_progress (bool) FALSE to hide epoch progress bars
#' @param show_overall_progress (bool) FALSE to hide overall progress bar
#'
#'
#'
#'
#'
#'
#'
#' @return None
#'
#'
#' @export
callback_tqdm_progress_bar <- function(metrics_separator = ' - ',
                                       overall_bar_format = '{l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}',
                                       epoch_bar_format = '{n_fmt}/{total_fmt}{bar} ETA: {remaining}s - {desc}',
                                       update_per_second = 10,
                                       leave_epoch_progress = TRUE,
                                       leave_overall_progress = TRUE,
                                       show_epoch_progress = TRUE,
                                       show_overall_progress = TRUE) {

  args = list(
    metrics_separator = metrics_separator,
    overall_bar_format = overall_bar_format,
    epoch_bar_format = epoch_bar_format,
    update_per_second = as.integer(update_per_second),
    leave_epoch_progress = leave_epoch_progress,
    leave_overall_progress = leave_overall_progress,
    show_epoch_progress = show_epoch_progress,
    show_overall_progress = show_overall_progress
  )

  do.call(tfa$callbacks$TQDMProgressBar, args)

}



#' @title Average Model Checkpoint
#'
#' @description Save the model after every epoch.
#'
#' @details The callback that should be used with optimizers that extend
#' AverageWrapper, i.e., MovingAverage and StochasticAverage optimizers.
#' It saves and, optionally, assigns the averaged weights.
#'
#' @param update_weights bool, wheteher to update weights or not
#' @param filepath string, path to save the model file.
#' @param monitor quantity to monitor.
#' @param verbose verbosity mode, 0 or 1.
#' @param save_best_only if `save_best_only=TRUE`, the latest best model according
#' to the quantity monitored will not be overwritten. If `filepath` doesn't contain
#' formatting options like `{epoch}` then `filepath` will be overwritten by each new
#' better model.
#' @param save_weights_only if TRUE, then only the model's weights will be saved
#' (`model$save_weights(filepath)`), else the full model is saved (`model$save(filepath)`).
#' @param mode one of {auto, min, max}. If `save_best_only=TRUE`, the decision to
#' overwrite the current save file is made based on either the maximization or the
#' minimization of the monitored quantity. For `val_acc`, this should be `max`, for
#' `val_loss` this should be `min`, etc. In `auto` mode, the direction is automatically
#' inferred from the name of the monitored quantity.
#' @param save_freq `'epoch'` or integer. When using `'epoch'`, the callback saves the
#' model after each epoch. When using integer, the callback saves the model at end of a
#' batch at which this many samples have been seen since last saving. Note that if the
#' saving isn't aligned to epochs, the monitored metric may potentially be less reliable
#' (it could reflect as little as 1 batch, since the metrics get reset every epoch).
#' Defaults to `'epoch'`
#' @param ... Additional arguments for backwards compatibility. Possible key is `period`.
#'
#' @section For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,:
#' then the model checkpoints will be saved with the epoch number and the validation loss in the filename.
#' @return None
#' @export
callback_average_model_checkpoint <- function(filepath, update_weights,
                                              monitor = "val_loss", verbose = 0,
                                              save_best_only = FALSE,save_weights_only = FALSE,
                                              mode = "auto", save_freq = "epoch", ...) {

  args <- list(
    filepath = filepath,
    update_weights = update_weights,
    monitor = monitor,
    verbose = as.integer(verbose),
    save_best_only = save_best_only,
    save_weights_only = save_weights_only,
    mode = mode,
    save_freq = save_freq,
    ...
  )

  do.call(tf$keras$callbacks$ModelCheckpoint, args)

}












