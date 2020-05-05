context("losses")

source("utils.R")

test_loss <- function(name) {
  loss_fn <- eval(parse(text = name))
  test_succeeds(name, {
    keras_model_sequential() %>%
      layer_dense(4, input_shape = c(784)) %>%
      compile(
        optimizer = 'sgd',
        loss=loss_fn(),
        metrics='accuracy'
      )
  })
}


test_loss("loss_contrastive")
test_loss("loss_pinball") # passes
test_loss("loss_sigmoid_focal_crossentropy")
test_loss("loss_triplet_hard")
test_loss("loss_triplet_semihard")
test_loss("loss_giou")
test_loss("loss_lifted_struct")
test_loss("loss_sparsemax")

sys = switch(Sys.info()[['sysname']],
             Windows= {paste("windows.")},
             Linux  = {paste("linux.")},
             Darwin = {paste("mac")})

# TensorFlow 2.1 does not support "loss_npairs" and "loss_npairs_multilabel" losses! Skip them.
if (!(sys == 'windows') & !(tensorflow::tf_version() == "2.1")) {
  test_loss("loss_npairs")
  test_loss("loss_npairs_multilabel")
}





