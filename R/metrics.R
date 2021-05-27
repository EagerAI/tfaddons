#' @title Computes Kappa score between two raters
#'
#' @details The score lies in the range [-1, 1]. A score of -1 represents complete
#' disagreement between two raters whereas a score of 1 represents complete agreement
#' between the two raters. A score of 0 means agreement by chance.
#'
#' @param num_classes Number of unique classes in your dataset.
#' @param weightage (optional) Weighting to be considered for calculating kappa statistics.
#' A valid value is one of [None, 'linear', 'quadratic']. Defaults to `NULL`
#' @param sparse_labels (bool) Valid only for multi-class scenario. If True, ground truth
#' labels are expected tp be integers and not one-hot encoded
#' @param regression (bool) If set, that means the problem is being treated as a regression
#' problem where you are regressing the predictions. **Note:** If you are regressing for the
#' values, the the output layer should contain a single unit.
#' @param name (optional) String name of the metric instance
#' @param dtype (optional) Data type of the metric result. Defaults to `NULL`
#'
#' @examples
#'
#' \dontrun{
#' model = keras_model_sequential() %>%
#'   layer_dense(units = 10, input_shape = ncol(iris) - 1,activation = activation_lisht) %>%
#'   layer_dense(units = 3)
#'
#' model %>% compile(loss = 'categorical_crossentropy',
#'                   optimizer = optimizer_radam(),
#'                   metrics = metric_cohen_kappa(3))
#' }
#'
#'
#'
#' @return Input tensor or list of input tensors.
#' @export
metric_cohen_kappa = function( num_classes,
                               name = "cohen_kappa",
                               weightage = NULL,
                               sparse_labels = FALSE,
                               regression = FALSE,
                               dtype = NULL) {
  args = list(
    num_classes = num_classes,
    name = name,
    weightage = weightage,
    sparse_labels = sparse_labels,
    regression = regression,
    dtype = dtype
  )

  if(!is.null(num_classes)) {
    args$num_classes <- as.integer(num_classes)
  }

  do.call(tfa$metrics$CohenKappa, args)

}

#attr(metric_cohen_kappa, "py_function_name") <- "cohen_kappa"

#' @title F1Score
#'
#' @description Computes F-1 Score.
#'
#' @details It is the harmonic mean of precision and recall.
#' Output range is [0, 1]. Works for both multi-class
#' and multi-label classification. F-1 = 2 * (precision * recall) / (precision + recall)
#'
#' @param num_classes Number of unique classes in the dataset.
#' @param average Type of averaging to be performed on data. Acceptable values are NULL,
#' micro, macro and weighted. Default value is NULL.
#' - None: Scores for each class are returned
#' - micro: True positivies, false positives and false negatives are computed globally.
#' - macro: True positivies, false positives and
#' - false negatives are computed for each class and their unweighted mean is returned.
#' - weighted: Metrics are computed for each class and returns the mean weighted by the number of
#' true instances in each class.
#' @param threshold Elements of y_pred above threshold are considered to be 1, and the rest 0.
#' If threshold is NULL, the argmax is converted to 1, and the rest 0.
#' @param dtype (optional) Data type of the metric result. Defaults to `tf$float32`.
#' @param name (optional) String name of the metric instance.
#' @return F-1 Score: float
#'
#' @examples
#'
#' \dontrun{
#' model = keras_model_sequential() %>%
#'   layer_dense(units = 10, input_shape = ncol(iris) - 1,activation = activation_lisht) %>%
#'   layer_dense(units = 3)
#'
#' model %>% compile(loss = 'categorical_crossentropy',
#'                   optimizer = optimizer_radam(),
#'                   metrics = metrics_f1score(3))
#' }
#'
#' @section Raises:
#' ValueError: If the `average` has values other than [NULL, micro, macro, weighted].
#'
#' @export
metrics_f1score <- function(num_classes,
                            average = NULL,
                            threshold = NULL,
                            name = 'f1_score',
                            dtype = tf$float32) {

  args <- list(
    num_classes = num_classes,
    average = NULL,
    threshold = NULL,
    name = name,
    dtype = dtype
  )

  if(!is.null(num_classes)) {
    args$num_classes <- as.integer(num_classes)
  }

  do.call(tfa$metrics$F1Score, args)

}




#' @title FBetaScore
#'
#' @description Computes F-Beta score.
#'
#' @details It is the weighted harmonic mean of precision
#' and recall. Output range is [0, 1]. Works for
#' both multi-class and multi-label classification.
#' F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)
#'
#' @param num_classes Number of unique classes in the dataset.
#' @param average Type of averaging to be performed on data. Acceptable
#' values are None, micro, macro and weighted. Default value is NULL.
#' micro, macro and weighted. Default value is NULL.
#' - None: Scores for each class are returned
#' - micro: True positivies, false positives and false negatives are computed globally.
#' - macro: True positivies, false positives and
#' - false negatives are computed for each class and their unweighted mean is returned.
#' - weighted: Metrics are computed for each class and returns the mean weighted by the number of
#' true instances in each class.-
#' @param beta Determines the weight of precision and recall in harmonic mean.
#' Determines the weight given to the precision and recall. Default value is 1.
#' @param threshold Elements of y_pred greater than threshold are converted to be 1,
#' and the rest 0. If threshold is None, the argmax is converted to 1, and the rest 0.
#' @param dtype (optional) Data type of the metric result. Defaults to `tf$float32`.
#' @param name (optional) String name of the metric instance.
#' @param ... additional parameters to pass
#' @return F-Beta Score: float
#'
#' @section Raises:
#' ValueError: If the `average` has values other than [NULL, micro, macro, weighted].
#'
#' @export
metric_fbetascore <- function(num_classes,
                       average = NULL,
                       beta = 1.0,
                       threshold = NULL,
                       name = "fbeta_score",
                       dtype = tf$float32,
                       ...) {

  args <- list(
    num_classes = num_classes,
    average = average,
    beta = beta,
    threshold = threshold,
    name = name,
    dtype = dtype,
    ...
  )

  if(!is.null(num_classes)) {
    args$num_classes <- as.integer(num_classes)
  }

  do.call(tfa$metrics$FBetaScore, args)

}

#' @title Hamming distance
#'
#' @description Computes hamming distance.
#'
#' @details Hamming distance is for comparing two binary strings.
#' It is the number of bit positions in which two bits
#' are different.
#'
#' @param actuals actual value
#' @param predictions predicted value
#'
#' @examples
#'
#' \dontrun{
#'
#' actuals = tf$constant(as.integer(c(1, 1, 0, 0, 1, 0, 1, 0, 0, 1)), dtype=tf$int32)
#' predictions = tf$constant(as.integer(c(1, 0, 0, 0, 1, 0, 0, 1, 0, 1)),dtype=tf$int32)
#' result = metric_hamming_distance(actuals, predictions)
#' paste('Hamming distance: ', result$numpy())
#'
#' }
#'
#' @return hamming distance: float
#'
#' @export
metric_hamming_distance <- function(actuals, predictions) {

  args <- list(
    actuals = actuals,
    predictions = predictions
  )

  do.call(tfa$metrics$hamming_distance, args)

}

attr(metric_hamming_distance, "py_function_name") <- "hamming_distance"



#' @title Hamming loss
#'
#' @description Computes hamming loss.
#'
#' @details Hamming loss is the fraction of wrong labels to the total number of labels.
#' In multi-class classification, hamming loss is calculated as the hamming distance
#' between `actual` and `predictions`. In multi-label classification, hamming loss
#' penalizes only the individual labels.
#'
#' @param threshold Elements of `y_pred` greater than threshold are converted to be 1,
#' and the rest 0. If threshold is None, the argmax is converted to 1, and the rest 0.
#' @param mode multi-class or multi-label
#' @param dtype (optional) Data type of the metric result. Defaults to `tf$float32`.
#' @param name (optional) String name of the metric instance.
#' @param ... additional arguments that are passed on to function `fn`.
#'
#' @examples
#' \dontrun{
#'
#' # multi-class hamming loss
#' hl = loss_hamming(mode='multiclass', threshold=0.6)
#' actuals = tf$constant(list(as.integer(c(1, 0, 0, 0)),as.integer(c(0, 0, 1, 0)),
#'                        as.integer(c(0, 0, 0, 1)),as.integer(c(0, 1, 0, 0))),
#'                       dtype=tf$float32)
#' predictions = tf$constant(list(c(0.8, 0.1, 0.1, 0),
#'                            c(0.2, 0, 0.8, 0),
#'                            c(0.05, 0.05, 0.1, 0.8),
#'                            c(1, 0, 0, 0)),
#'                           dtype=tf$float32)
#' hl$update_state(actuals, predictions)
#' paste('Hamming loss: ', hl$result()$numpy()) # 0.25
#' # multi-label hamming loss
#' hl = loss_hamming(mode='multilabel', threshold=0.8)
#' actuals = tf$constant(list(as.integer(c(1, 0, 1, 0)),as.integer(c(0, 1, 0, 1)),
#'                        as.integer(c(0, 0, 0,1))), dtype=tf$int32)
#' predictions = tf$constant(list(c(0.82, 0.5, 0.90, 0),
#'                            c(0, 1, 0.4, 0.98),
#'                            c(0.89, 0.79, 0, 0.3)),
#'                           dtype=tf$float32)
#' hl$update_state(actuals, predictions)
#' paste('Hamming loss: ', hl$result()$numpy()) # 0.16666667
#'
#' }
#'
#' @return  hamming loss: float
#' @export
loss_hamming <- function(mode,
                         name = 'hamming_loss',
                         threshold = NULL,
                         dtype = tf$float32,
                         ...) {

  args <- list(
    mode,
    name = 'hamming_loss',
    threshold = NULL,
    dtype = tf$float32,
    ...
  )

  do.call(tfa$metrics$HammingLoss, args)

}



#' @title MatthewsCorrelationCoefficient
#'
#' @description Computes the Matthews Correlation Coefficient.
#'
#' @details The statistic is also known as the phi coefficient.
#' The Matthews correlation coefficient (MCC) is used in
#' machine learning as a measure of the quality of binary
#' and multiclass classifications. It takes into account
#' true and false positives and negatives and is generally
#' regarded as a balanced measure which can be used even
#' if the classes are of very different sizes. The correlation
#' coefficient value of MCC is between -1 and +1. A
#' coefficient of +1 represents a perfect prediction,
#' 0 an average random prediction and -1 an inverse
#' prediction. The statistic is also known as
#' the phi coefficient. MCC = (TP * TN) - (FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2) Usage:
#'
#' @param num_classes Number of unique classes in the dataset.
#' @param name (Optional) String name of the metric instance.
#' @param dtype (Optional) Data type of the metric result. Defaults to `tf$float32`.
#'
#' @examples
#'
#' \dontrun{
#'
#' actuals = tf$constant(list(1, 1, 1, 0), dtype=tf$float32)
#' preds = tf$constant(list(1,0,1,1), dtype=tf$float32)
#' # Matthews correlation coefficient
#' mcc = metric_mcc(num_classes=1)
#' mcc$update_state(actuals, preds)
#' paste('Matthews correlation coefficient is:', mcc$result()$numpy())
#' # Matthews correlation coefficient is : -0.33333334
#'
#' }
#'
#'
#' @return Matthews correlation coefficient: float
#'
#' @export
metric_mcc <- function(num_classes = NULL,
                       name = 'MatthewsCorrelationCoefficient',
                       dtype = tf$float32) {

  args <- list(
    num_classes = num_classes,
    name = name,
    dtype = dtype
  )

  if(!is.null(num_classes)) {
    args$num_classes <- as.integer(num_classes)
  }

  do.call(tfa$metrics$MatthewsCorrelationCoefficient, args)

}



#' @title MultiLabelConfusionMatrix
#'
#' @description Computes Multi-label confusion matrix.
#'
#' @details Class-wise confusion matrix is computed for the
#' evaluation of classification. If multi-class input is provided, it will be treated
#' as multilabel data. Consider classification problem with two classes
#' (i.e num_classes=2). Resultant matrix `M` will be in the shape of (num_classes, 2, 2).
#' Every class `i` has a dedicated 2*2 matrix that contains: - true negatives for class i in M(0,0)
#' - false positives for class i in M(0,1)
#' - false negatives for class i in M(1,0)
#' - true positives for class i in M(1,1) ```python
#' # multilabel confusion matrix
#' y_true = tf$constant(list(as.integer(c(1, 0, 1)), as.integer(c(0, 1, 0))), dtype=tf$int32)
#' y_pred = tf$constant(list(as.integer(c(1, 0, 0)), as.integer(c(0, 1, 1))), dtype=tf$int32)
#' output = metric_multilabel_confusion_matrix(num_classes=3)
#' output$update_state(y_true, y_pred)
#' paste('Confusion matrix:', output$result())
#' # Confusion matrix: [[[1 0] [0 1]] [[1 0] [0 1]] [[0 1] [1 0]]] # if multiclass input is provided
#' y_true = tf$constant(list(as.integer(c(1, 0, 0)), as.integer(c(0, 1, 0))), dtype=tf$int32)
#' y_pred = tf$constant(list(as.integer(c(1, 0, 0)), as.integer(c(0, 0, 1))), dtype=tf$int32)
#' output = metric_multilabel_confusion_matrix(num_classes=3)
#' output$update_state(y_true, y_pred)
#' paste('Confusion matrix:', output$result())
#' # Confusion matrix: [[[1 0] [0 1]] [[1 0] [1 0]] [[1 1] [0 0]]]
#' ```
#'
#' @param num_classes Number of unique classes in the dataset.
#' @param name (Optional) String name of the metric instance.
#' @param dtype (Optional) Data type of the metric result. Defaults to `tf$int32`.
#' @return MultiLabelConfusionMatrix: float
#' @export
metric_multilabel_confusion_matrix <- function(num_classes,
                                               name = 'Multilabel_confusion_matrix',
                                               dtype = tf$int32) {

  args <- list(
    num_classes = num_classes,
    name = name,
    dtype = dtype
  )

  if(!is.null(num_classes)) {
    args$num_classes <- as.integer(num_classes)
  }

  do.call(tfa$metrics$MultiLabelConfusionMatrix, args)

}


#' @title RSquare
#'
#' This is also called as coefficient of determination. It tells how close
#' are data to the fitted regression line. Highest score can be 1.0 and it
#' indicates that the predictors perfectly accounts for variation in the target.
#' Score 0.0 indicates that the predictors do not account for variation in the
#' target. It can also be negative if the model is worse.
#'
#' @param name (Optional) String name of the metric instance.
#' @param dtype (Optional) Data type of the metric result. Defaults to `tf$float32`.
#' @param ... additional arguments to pass
#' @param multioutput one of the following: "raw_values", "uniform_average", "variance_weighted"
#' @param shape output tensor shape
#'
#'
#' @examples
#'
#' \dontrun{
#'
#' actuals = tf$constant(c(1, 4, 3), dtype=tf$float32)
#' preds = tf$constant(c(2, 4, 4), dtype=tf$float32)
#' result = metric_rsquare()
#' result$update_state(actuals, preds)
#' paste('R^2 score is: ', result$result()$numpy()) # 0.57142866
#'
#' }
#' @return r squared score: float
#' @export
metric_rsquare <- function(name = 'r_square',
                           dtype = tf$float32,
                           multioutput = 'uniform_average',
                           y_shape = 1,
                           ...) {

  tfa$metrics$RSquare(name = name,
                      dtype = dtype,
                      multioutput = multioutput,
                      y_shape = reticulate::tuple(as.integer(y_shape)),
                      ...)

}







