context("optimizers")

source("utils.R")

test_optimizer <- function(name) {
  optimizer_fn <- eval(parse(text = name))
  test_succeeds(name, {
    keras_model_sequential() %>%
      layer_dense(32, input_shape = c(784)) %>%
      compile(
        optimizer = optimizer_fn(),
        loss='binary_crossentropy',
        metrics='accuracy'
      )
  })
}


test_optimizer("optimizer_lamb")
test_optimizer("optimizer_lazy_adam")
test_optimizer("optimizer_novograd")
test_optimizer("optimizer_radam")
test_optimizer("optimizer_yogi")


test_succeeds('optimizer_conditional_gradient', {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    compile(
      optimizer = optimizer_conditional_gradient(learning_rate = 1e-3, lambda_ = 0.04),
      loss='binary_crossentropy',
      metrics='accuracy'
    )
})


test_succeeds('optimizer_decay_adamw', {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    compile(
      optimizer = optimizer_decay_adamw(weight_decay = 1e-3),
      loss='binary_crossentropy',
      metrics='accuracy'
    )
})

test_succeeds('optimizer_decay_sgdw', {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    compile(
      optimizer = optimizer_decay_sgdw(weight_decay = 1e-3),
      loss='binary_crossentropy',
      metrics='accuracy'
    )
})



test_succeeds('optimizer_moving_average', {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    compile(
      optimizer = optimizer_moving_average(optimizer_decay_sgdw(weight_decay = 1e-3)),
      loss='binary_crossentropy',
      metrics='accuracy'
    )
})


test_succeeds('optimizer_swa', {
  keras_model_sequential() %>%
    layer_dense(32, input_shape = c(784)) %>%
    compile(
      optimizer = optimizer_swa(optimizer_decay_sgdw(weight_decay = 1e-3)),
      loss='binary_crossentropy',
      metrics='accuracy'
    )
})




