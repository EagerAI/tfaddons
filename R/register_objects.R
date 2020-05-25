#' @title Register all
#'
#' @description Register TensorFlow Addons' objects in TensorFlow global dictionaries.
#'
#' @details When loading a Keras model that has a TF Addons' function, it is needed
#' for this function to be known by the Keras deserialization process. There are two ways
#' to do this, either do
#' ```
#' tf$keras$models$load_model( "my_model.tf", custom_objects=list("LAMB": tfaddons::optimizer_lamb)
#' )
#' ```
#' or you can do:
#' ```python
#' register_all()
#' tf$keras$models$load_model("my_model.tf")
#' ``` If the model contains custom ops (compiled ops) of TensorFlow Addons,
#' and the graph is loaded with `tf$saved_model$load`, then custom ops need
#' to be registered before to avoid an error of the type:
#' ```
#' tensorflow.python.framework.errors_impl.NotFoundError: Op type not registered
#' '...' in binary running on ... Make sure the Op and Kernel are
#' registered in the binary running in this process.
#' ```
#' In this case, the only way to make sure that the ops are registered is to call
#' this function:
#' ```
#' register_all()
#' tf$saved_model$load("my_model.tf")
#' ```
#' Note that you can call this function multiple times in the same process,
#' it only has an effect the first time. Afterward, it's just a no-op.
#'
#' @param keras_objects boolean, `TRUE` by default. If `TRUE`, register all Keras
#' objects with `tf$keras$utils$register_keras_serializable(package="Addons")` If
#' set to FALSE, doesn't register any Keras objects of Addons in TensorFlow.
#' @param custom_kernels boolean, `TRUE` by default. If `TRUE`, loads all custom
#' kernels of TensorFlow Addons with `tf.load_op_library("path/to/so/file.so")`.
#' Loading the SO files register them automatically. If `FALSE` doesn't load and
#' register the shared objects files. Not that it might be useful to turn it off
#' if your installation of Addons doesn't work well with custom ops.
#'
#' @return None
#'
#'
#' @export
register_all <- function(keras_objects = TRUE, custom_kernels = TRUE) {

  args <- list(
    keras_objects = keras_objects,
    custom_kernels = custom_kernels
  )

  do.call(tfa$register_all, args)

}

#' @title Register keras objects
#'
#' @param ... parameters to pass
#' @return None
#' @export
register_keras_objects <- function(...) {

  args = list(...)

  do.call(tfa$register$register_keras_objects,args)
}

#' @title Register custom kernels
#'
#' @param ... parameters to pass
#' @return None
#' @export
register_custom_kernels <- function(...) {

  args = list(...)

  do.call(tfa$register$register_custom_kernels,args)
}



