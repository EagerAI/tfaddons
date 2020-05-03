context("activations")

source("utils.R")

test_activation <- function(name) {
  test_succeeds(paste("use activation", name), {
    activation_fn <- eval(parse(text = name))
    test_succeeds(name, {
      keras_model_sequential() %>%
        layer_dense(32, input_shape = 784) %>%
        layer_activation(activation = activation_fn)
    })
    tensor <- k_constant(matrix(runif(100), nrow = 10, ncol = 10), shape = c(10, 10))
    activation_fn(tensor)
  })
}


test_activation("activation_gelu")
test_activation("activation_hardshrink")
test_activation("activation_lisht")
test_activation("activation_mish")
test_activation("activation_softshrink")
test_activation("activation_sparsemax")
test_activation("activation_tanhshrink")



test_succeeds('activation rrelu', {
  layer_activation(activation = activation_rrelu)
  tensor <- k_constant(matrix(runif(100), nrow = 10, ncol = 10), shape = c(10, 10))
  activation_rrelu(tensor)
})








