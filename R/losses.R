#' @title Contrastive loss
#'
#' @description Computes the contrastive loss between `y_true` and `y_pred`.
#' @param margin Float, margin term in the loss definition. Default value is 1.0.
#' @param reduction (Optional) Type of tf$keras$losses$Reduction to apply.
#' Default value is SUM_OVER_BATCH_SIZE.
#' @param name (Optional) name for the loss.
#' @details This loss encourages the embedding to be close to each other for
#' the samples of the same label and the embedding to be far apart at least
#' by the margin constant for the samples of different labels.
#' The euclidean distances `y_pred` between two embedding matrices
#' `a` and `b` with shape [batch_size, hidden_size] can be computed
#' as follows: ```python
#' # y_pred = `\\sqrt` (`\\sum_i` (a[:, i] - b[:, i])^2)
#' y_pred = tf$linalg.norm(a - b, axis=1)
#' ``` See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#'
#' @return contrastive_loss: 1-D float `Tensor` with shape [batch_size].
#'
#' @export
loss_contrastive <- function(margin = 1.0,
                             reduction = tf$keras$losses$Reduction$SUM_OVER_BATCH_SIZE,
                             name = 'contrasitve_loss') {

  args = list(margin = margin,
              reduction = reduction,
              name = name)
  do.call(tfa$losses$ContrastiveLoss, args)

}

attr(loss_contrastive, "py_function_name") <- "contrastive"


#' @title Implements the GIoU loss function.
#' @description GIoU loss was first introduced in the [Generalized Intersection over Union:
#' A Metric and A Loss for Bounding Box Regression](https://giou.stanford.edu/GIoU.pdf).
#' GIoU is an enhancement for models which use IoU in object detection.
#'
#' @param mode one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
#' @param name A name for the operation (optional).
#' @param reduction (Optional) Type of tf$keras$losses$Reduction to apply.
#' Default value is SUM_OVER_BATCH_SIZE.
#'
#' @return GIoU loss float `Tensor`.
#'
#' @export
loss_giou <- function(mode = 'giou',
                      reduction = tf$keras$losses$Reduction$AUTO,
                      name = 'giou_loss') {

  args <- list(
    mode = mode,
    reduction = reduction,
    name = name
  )

  do.call(tfa$losses$GIoULoss, args)

}

attr(loss_giou, "py_function_name") <- "giou"

#' @title Lifted structured loss
#'
#' @description Computes the lifted structured loss.
#' @details The loss encourages the positive distances (between a pair of embeddings
#' with the same labels) to be smaller than any negative distances (between a pair of
#' embeddings with different labels) in the mini-batch in a way that is differentiable
#' with respect to the embedding vectors. See: https://arxiv.org/abs/1511.06452
#' @param margin Float, margin term in the loss definition.
#' @param name Optional name for the op.
#' @param ... additional parameters to pass
#' @return lifted_loss: tf$float32 scalar.
#'
#' @export
loss_lifted_struct <- function(margin = 1.0,
                               name = NULL,
                               ...) {

  args <- list(margin = margin,
               name = name,
               ...)

  do.call(tfa$losses$LiftedStructLoss, args)

}

attr(loss_lifted_struct, "py_function_name") <- "lifted_struct"


#' @title Npairs loss
#'
#' @description Computes the npairs loss between `y_true` and `y_pred`.
#'
#' @details Npairs loss expects paired data where a pair is composed of samples from
#' the same labels and each pairs in the minibatch have different labels.
#' The loss takes each row of the pair-wise similarity matrix, `y_pred`,
#' as logits and the remapped multi-class labels, `y_true`, as labels. The
#' similarity matrix `y_pred` between two embedding matrices `a` and `b`
#' with shape `[batch_size, hidden_size]` can be computed as follows:
#' ```
#' # y_pred = a * b^T
#' y_pred = tf$matmul(a, b, transpose_a=FALSE, transpose_b=TRUE)
#' ```
#' See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#' @param name Optional name for the op.
#' @return npairs_loss: float scalar.
#'
#' @export
loss_npairs <- function(name = 'npairs_loss') {

  args <- list(name = name
  )

  do.call(tfa$losses$NpairsLoss, args)

}

attr(loss_npairs, "py_function_name") <- "npairs"


#' @title Npairs multilabel loss
#'
#' @description Computes the npairs loss between multilabel data `y_true` and `y_pred`.
#'
#' @param name Optional name for the op.
#' @details Npairs loss expects paired data where a pair is composed of samples from
#' the same labels and each pairs in the minibatch have different labels.
#' The loss takes each row of the pair-wise similarity matrix, `y_pred`,
#' as logits and the remapped multi-class labels, `y_true`, as labels. To deal with
#' multilabel inputs, the count of label intersection
#' is computed as follows:
#' ```
#' L_{i,j} = | set_of_labels_for(i) `\\cap` set_of_labels_for(j) |
#' ```
#' Each row of the count based label matrix is further normalized so that
#' each row sums to one. `y_true` should be a binary indicator for classes.
#' That is, if `y_true[i, j] = 1`, then `i`th sample is in `j`th class;
#' if `y_true[i, j] = 0`, then `i`th sample is not in `j`th class. The similarity matrix
#' `y_pred` between two embedding matrices `a` and `b`
#' with shape `[batch_size, hidden_size]` can be computed as follows:
#' ```
#' # y_pred = a * b^T
#' y_pred = tf.matmul(a, b, transpose_a=FALSE, transpose_b=TRUE)
#' ```
#'
#' @section See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#'
#'
#' @return npairs_multilabel_loss: float scalar.
#'
#' @export
loss_npairs_multilabel <- function( name = 'npairs_multilabel_loss') {

  args <- list(
    name = name
  )

  do.call(tfa$losses$NpairsMultilabelLoss, args)

}

attr(loss_npairs_multilabel, "py_function_name") <- "npairs_multilabel"


#' @title Pinball loss
#'
#' @description Computes the pinball loss between `y_true` and `y_pred`.
#'
#' @details `loss = maximum(tau * (y_true - y_pred), (tau - 1) * (y_true - y_pred))` In the context of regression this, loss yields an estimator of the tau
#' conditional quantile. See: https://en.wikipedia.org/wiki/Quantile_regression Usage:
#' ```python
#' loss = pinball_loss([0., 0., 1., 1.], [1., 1., 1., 0.], tau=.1) # loss = max(0.1 * (y_true - y_pred), (0.1 - 1) * (y_true - y_pred))
#' # = (0.9 + 0.9 + 0 + 0.1) / 4 print('Loss: ', loss$numpy()) # Loss: 0.475
#' ```
#'
#' @return pinball_loss: 1-D float `Tensor` with shape [batch_size].
#'
#' @section Usage:
#' ```python_loss = pinball_loss([0., 0., 1., 1.], [1., 1., 1., 0.], tau=.1) ````
#' @param tau (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
#' shape = [d0,..., dn]. It defines the slope of the pinball loss. In the context
#' of quantile regression, the value of tau determines the conditional quantile
#' level. When tau = 0.5, this amounts to l1 regression, an estimator of the
#' conditional median (0.5 quantile).
#' @param reduction (Optional) Type of tf.keras.losses.Reduction to apply to loss.
#' Default value is AUTO. AUTO indicates that the reduction option will be determined
#' by the usage context. For almost all cases this defaults to SUM_OVER_BATCH_SIZE.
#' When used with tf.distribute.Strategy, outside of built-in training loops such as
#' tf$keras compile and fit, using AUTO or SUM_OVER_BATCH_SIZE will raise an error.
#' Please see https://www.tensorflow.org/alpha/tutorials/distribute/training_loops
#' for more details on this.
#' @param name Optional name for the op.
#' @section References:
#' - https://en.wikipedia.org/wiki/Quantile_regression - https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
#' @return pinball_loss: 1-D float `Tensor` with shape [batch_size].
#' @export
loss_pinball <- function(tau = 0.5,
                         reduction = tf$keras$losses$Reduction$AUTO,
                         name = 'pinball_loss') {

  args <- list(
    tau = tau,
    reduction = reduction,
    name = name
  )

  do.call(tfa$losses$PinballLoss, args)

}

attr(loss_pinball, "py_function_name") <- "pinball"


#' @title Sigmoid focal crossentropy loss
#'
#'
#' @param name (Optional) name for the loss.
#' @param alpha balancing factor.
#' @param gamma modulating factor.
#' @param reduction (Optional) Type of tf$keras$losses$Reduction to apply.
#' Default value is SUM_OVER_BATCH_SIZE.
#' @return Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the same shape as `y_true`;
#' otherwise, it is scalar.
#' @param from_logits If logits are provided then convert the predictions into probabilities
#' @export
loss_sigmoid_focal_crossentropy <- function(from_logits = FALSE,
                                            alpha = 0.25,
                                            gamma = 2.0,
                                            reduction = tf$keras$losses$Reduction$NONE,
                                            name = 'sigmoid_focal_crossentropy') {

  args <- list(
    from_logits = from_logits,
    alpha = alpha,
    gamma = gamma,
    reduction = reduction,
    name = name
  )

  do.call(tfa$losses$SigmoidFocalCrossEntropy, args)

}

attr(loss_sigmoid_focal_crossentropy, "py_function_name") <- "sigmoid_focal_crossentropy"


#' @title Sparsemax loss
#'
#' @description Sparsemax loss function [1].
#'
#' @details Computes the generalized multi-label classification loss for the sparsemax
#' function. The implementation is a reformulation of the original loss
#' function such that it uses the sparsemax properbility output instead of the
#' internal au variable. However, the output is identical to the original
#' loss function. [1]: https://arxiv.org/abs/1602.02068
#'
#' @param from_logits Whether y_pred is expected to be a logits tensor.
#' Default is True, meaning y_pred is the logits.
#' @param reduction (Optional) Type of tf$keras$losses$Reduction to apply
#' to loss. Default value is SUM_OVER_BATCH_SIZE.
#' @param name Optional name for the op.
#'
#' @return A `Tensor`. Has the same type as `logits`.
#'
#' @export
loss_sparsemax <- function(from_logits = TRUE,
                           reduction = tf$keras$losses$Reduction$SUM_OVER_BATCH_SIZE,
                           name = 'sparsemax_loss') {

  args <- list(
    from_logits = from_logits,
    reduction = reduction,
    name = name
  )

  do.call(tfa$losses$SparsemaxLoss, args)

}

attr(loss_sparsemax, "py_function_name") <- "sparsemax"


#' @title Triplet hard loss
#'
#' @description Computes the triplet loss with hard negative and hard positive mining.
#' @param margin Float, margin term in the loss definition. Default value is 1.0.
#' @param soft Boolean, if set, use the soft margin version. Default value is False.
#' @param name Optional name for the op.
#' @param ... additional arguments to pass
#'
#' @return triplet_loss: float scalar with dtype of y_pred.
#' @export
loss_triplet_hard <- function(margin = 1.0,
                              soft = FALSE,
                              name = NULL,
                              ...) {

  args <- list(
    margin = margin,
    soft = soft,
    name = name,
    ...
  )

  do.call(tfa$losses$TripletHardLoss, args)

}

attr(loss_triplet_hard, "py_function_name") <- "triplet_hard"


#' @title Triplet semihard loss
#'
#' @description Computes the triplet loss with semi-hard negative mining.
#' @param margin Float, margin term in the loss definition. Default value is 1.0.
#' @param name Optional name for the op.
#' @param ... additional arguments to pass
#' @return triplet_loss: float scalar with dtype of y_pred.
#' @export
loss_triplet_semihard <- function(margin = 1.0,
                                  name = NULL,
                                  ...) {

  args <- list(
    margin = margin,
    name = name,
    ...
  )

  do.call(tfa$losses$TripletSemiHardLoss, args)

}

attr(loss_triplet_semihard, "py_function_name") <- "triplet_semihard"





