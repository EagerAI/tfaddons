#' @title Interface to 'TensorFlow SIG Addons'
#' @description 'TensorFlow SIG Addons' <https://www.tensorflow.org/addons> is a repository
#' of community contributions that conform to well-established API patterns,
#' but implement new functionality not available in core TensorFlow.
#' TensorFlow natively supports a large number of operators, layers, metrics,
#' losses, optimizers, and more. However, in a fast moving field like ML,
#' there are many interesting new developments that cannot be integrated into
#' core TensorFlow (because their broad applicability is not yet clear, or
#' it is mostly used by a smaller subset of the community).
tfa <- NULL

.onLoad <- function(libname, pkgname) {

  tfa <<- reticulate::import("tensorflow_addons", delay_load = list(
    priority = 10,
    environment = "r-tensorflow"
  ))

}
