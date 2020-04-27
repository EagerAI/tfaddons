#' @title Computes Kappa score between two raters
#'
#' @details The score lies in the range [-1, 1]. A score of -1 represents complete
#' disagreement between two raters whereas a score of 1 represents complete agreement
#' between the two raters. A score of 0 means agreement by chance.
#'
#' @param num_classes Number of unique classes in your dataset.
#' @param weightage (optional) Weighting to be considered for calculating kappa statistics.
#' A valid value is one of [None, 'linear', 'quadratic']. Defaults to `NULL`
#' @param sparse_lables (bool) Valid only for multi-class scenario. If True, ground truth
#' labels are expected tp be integers and not one-hot encoded
#' @param regression (bool) If set, that means the problem is being treated as a regression
#' problem where you are regressing the predictions. **Note:** If you are regressing for the
#' values, the the output layer should contain a single unit.
#' @param name (optional) String name of the metric instance
#' @param dtype (optional) Data type of the metric result. Defaults to `NULL`
#'
#'
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
#' @param dtype Represents the type of the elements in a `Tensor`.
#' @param name (optional) String name of the metric instance.
#' @return F-1 Score: float
#'
#' @section Raises:
#' ValueError: If the `average` has values other than [NULL, micro, macro, weighted].
#'
#' @export
metrics_f1score <- function(num_classes,
                            average = None,
                            threshold = None,
                            name = 'f1_score',
                            dtype = tf$float32) {

  args <- list(
    num_classes = num_classes,
    average = NULL,
    threshold = NULL,
    name = name,
    dtype = dtype
  )

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
#' @param dtype Represents the type of the elements in a `Tensor`.
#' @param name (optional) String name of the metric instance.
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

  do.call(tfa$metrics$FBetaScore, args)

}







