#' @title Version of TensorFlow SIG Addons
#' @description Get the current version of TensorFlow SIG Addons
#' @return prints the version.
#' @export
tfaddons_version = function() {
  tfa$`__version__`
}
