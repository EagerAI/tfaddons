#' @title Install TensorFlow SIG Addons
#'
#' @description This function is used to install the `TensorFlow SIG Addons` python module

#' @param version for specific version of `TensorFlow SIG Addons`, e.g. "0.10.0"
#' @param ... other arguments passed to [reticulate::py_install()].
#' @param restart_session Restart R session after installing (note this will only occur within RStudio).
#' @return a python module `tensorflow_addons`
#' @importFrom reticulate py_config py_install
#' @export
install_tfaddons <- function(version = NULL, ..., restart_session = TRUE) {

  tfaddons_py_install = function(pkgs) {
    system(paste(reticulate::py_discover_config()[['python']],'-m pip install',
                 paste(pkgs,collapse = ' ')
    ))
  }

  if (is.null(version))
    module_string <- paste0("tensorflow-addons==", '0.16.1')
  else
    module_string <- paste0("tensorflow-addons==", version)

  invisible(py_config())
  #py_install(packages = paste(module_string), pip = TRUE, ...)
  tfaddons_py_install(module_string)


  if (restart_session && rstudioapi::hasFun("restartSession"))
    rstudioapi::restartSession()
}
