context("metrics")

source("utils.R")

data("iris")

build_model <- function(metric) {
  model = keras_model_sequential() %>%
    layer_dense(units = 10, input_shape = ncol(iris) - 1,activation = activation_lisht) %>%
    layer_dense(units = 3)

  model %>% compile(loss = 'categorical_crossentropy',
                    optimizer = optimizer_radam(),
                    metrics = metric)

  history = model %>% fit(as.matrix(iris[1:4]),
                          tf$keras$utils$to_categorical(iris[,4]),
                          epochs = 2,
                          validation_split = 0.2,
                          verbose = 1 )
}


test_metrics <- function(name, metric) {
  test_succeeds(paste(name), {
    build_model(metric)
  })
}

test_metrics("cohen_kappa",metric_cohen_kappa(num_classes = 3))
test_metrics("fbetascore",metric_fbetascore(num_classes = 3))
test_metrics("MatthewsCorrelationCoefficient",metric_mcc(num_classes = 3))
test_metrics("multilabel_confusion_matrix",metric_multilabel_confusion_matrix(num_classes = 3))
test_metrics("f1score",tfaddons::metrics_f1score(num_classes = 3))
#test_metrics("R^2",metric_rsquare())


