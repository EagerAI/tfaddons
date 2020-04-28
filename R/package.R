
tfa <- NULL

.onLoad <- function(libname, pkgname) {

  tfa <<- reticulate::import("tensorflow_addons", delay_load = list(
    priority = 10,
    environment = "r-tensorflow"
  ))

}
