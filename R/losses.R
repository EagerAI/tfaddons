#' @title Contrastive loss
#'
#' @description Computes the contrastive loss between `y_true` and `y_pred`.
#' @param y_true 1-D integer Tensor with shape [batch_size] of binary labels indicating positive vs negative pair.
#' @param y_pred 1-D float Tensor with shape [batch_size] of distances between two embedding matrices.
#' @param margin margin term in the loss definition.
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
loss_contrastive <- function(y_true, y_pred, margin = 1.0) {

  args = list(y_true = y_true,
              y_pred = y_pred,
              margin = margin)
  do.call(tfa$losses$contrastive_loss, args)

}

attr(loss_contrastive, "py_function_name") <- "contrastive"


#' @title Giou loss
#'
#'
#' @param y_true true targets tensor. The coordinates of the each bounding box in boxes are
#' encoded as [y_min, x_min, y_max, x_max].
#' @param y_pred predictions tensor. The coordinates of the each bounding box in boxes are
#' encoded as [y_min, x_min, y_max, x_max].
#' @param mode one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
#'
#' @return GIoU loss float `Tensor`.
#'
#' @export
loss_giou <- function(y_true, y_pred, mode = "giou") {

  args <- list(
    y_true = y_true,
    y_pred = y_pred,
    mode = mode
  )

  do.call(tfa$losses$giou_loss, args)

}

attr(loss_giou, "py_function_name") <- "giou"

#' @title Lifted structured loss
#'
#' @description Computes the lifted structured loss.
#' @param labels 1-D tf.int32 Tensor with shape [batch_size] of multiclass integer labels.
#' @param embeddings 2-D float Tensor of embedding vectors. Embeddings should not be l2 normalized.
#' @param margin Float, margin term in the loss definition.
#' @return lifted_loss: tf.float32 scalar.
#'
#' @export
loss_lifted_struct <- function(labels, embeddings, margin = 1.0) {

  args <- list(labels = labels,
               embeddings = embeddings,
               margin = margin)

  do.call(tfa$losses$lifted_struct_loss, args)

}

attr(loss_lifted_struct, "py_function_name") <- "lifted_struct"


#' @title npairs_loss
#'
#' @description Computes the npairs loss between `y_true` and `y_pred`.
#'
#' @details Npairs loss expects paired data where a pair is composed of samples from
#' the same labels and each pairs in the minibatch have different labels.
#' The loss takes each row of the pair-wise similarity matrix, `y_pred`,
#' as logits and the remapped multi-class labels, `y_true`, as labels. The similarity matrix `y_pred` between two embedding matrices `a` and `b`
#' with shape `[batch_size, hidden_size]` can be computed as follows: ```python
#' # y_pred = a * b^T
#' y_pred = tf$matmul(a, b, transpose_a=FALSE, transpose_b=TRUE)
#' ``` See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#' @param y_true 1-D integer Tensor with shape [batch_size] of multi-class labels.
#' @param y_pred 2-D float Tensor with shape [batch_size, batch_size] of similarity matrix between embedding matrices.
#'
#' @return npairs_loss: float scalar.
#'
#' @export
loss_npairs <- function(y_true, y_pred) {

  args <- list(y_true = y_true,
               y_pred = y_pred
  )

  do.call(tfa$losses$npairs_loss, args)

}

attr(loss_npairs, "py_function_name") <- "npairs"


#' @title Npairs multilabel loss
#'
#' @description Computes the npairs loss between multilabel data `y_true` and `y_pred`.
#'
#' @param y_true Either 2-D integer Tensor with shape [batch_size, num_classes], or
#' SparseTensor with dense shape [batch_size, num_classes]. If y_true is a SparseTensor,
#' then it will be converted to Tensor via tf$sparse$to_dense first.
#' @param y_pred 2-D float Tensor with shape [batch_size, batch_size] of similarity matrix between embedding matrices.
#'
#' @details Npairs loss expects paired data where a pair is composed of samples from
#' the same labels and each pairs in the minibatch have different labels.
#' The loss takes each row of the pair-wise similarity matrix, `y_pred`,
#' as logits and the remapped multi-class labels, `y_true`, as labels. To deal with multilabel inputs, the count of label intersection
#' is computed as follows: ```
#' L_{i,j} = | set_of_labels_for(i) `\\cap` set_of_labels_for(j) |
#' ``` Each row of the count based label matrix is further normalized so that
#' each row sums to one. `y_true` should be a binary indicator for classes.
#' That is, if `y_true[i, j] = 1`, then `i`th sample is in `j`th class;
#' if `y_true[i, j] = 0`, then `i`th sample is not in `j`th class. The similarity matrix `y_pred` between two embedding matrices `a` and `b`
#' with shape `[batch_size, hidden_size]` can be computed as follows: ```python
#' # y_pred = a * b^T
#' y_pred = tf.matmul(a, b, transpose_a=FALSE, transpose_b=TRUE)
#' ``` See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf
#'
#'
#' @return npairs_multilabel_loss: float scalar.
#'
#' @export
loss_npairs_multilabel <- function(y_true, y_pred) {

  args <- list(
    y_true = y_true,
    y_pred = y_pred
  )

  do.call(tfa$losses$npairs_multilabel_loss, args)

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
#' ```python_loss = pinball_loss([0., 0., 1., 1.], [1., 1., 1., 0.], tau=.1)
#' @param y_true Ground truth values. shape = `[batch_size, d0, .. dN]`
#' @param y_pred The predicted values. shape = `[batch_size, d0, .. dN]`
#' @param tau (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
#' shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
#' the context of quantile regression, the value of tau determines the
#' conditional quantile level. When tau = 0.5, this amounts to l1
#' regression, an estimator of the conditional median (0.5 quantile).
#' tau: (Optional) Float in [0, 1] or a tensor taking values in [0, 1] and
#' shape = `[d0,..., dn]`.  It defines the slope of the pinball loss. In
#' the context of quantile regression, the value of tau determines the
#' conditional quantile level. When tau = 0.5, this amounts to l1
#' regression, an estimator of the conditional median (0.5 quantile).
#' @section References:
#' - https://en.wikipedia.org/wiki/Quantile_regression - https://projecteuclid.org/download/pdfview_1/euclid.bj/1297173840
#' @return pinball_loss: 1-D float `Tensor` with shape [batch_size].
#' @export
loss_pinball <- function(y_true, y_pred, tau = 0.1) {

  args <- list(
    y_true = y_true,
    y_pred = y_pred,
    tau = tau
  )

  do.call(tfa$losses$pinball_loss, args)

}

attr(loss_pinball, "py_function_name") <- "pinball"


#' @title Sigmoid focal crossentropy loss
#'
#'
#' @param y_true true targets tensor.
#' @param y_pred predictions tensor.
#' @param alpha balancing factor.
#' @param gamma modulating factor.
#' @return Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the same shape as `y_true`;
#' otherwise, it is scalar.
#' @param from_logits If logits are provided then convert the predictions into probabilities
#' @export
loss_sigmoid_focal_crossentropy <- function(y_true, y_pred, alpha = 0.25, gamma = 2.0, from_logits = FALSE) {

  args <- list(
    y_true = y_true,
    y_pred = y_pred,
    alpha = alpha,
    gamma = gamma,
    from_logits = from_logits
  )

  do.call(tfa$losses$sigmoid_focal_crossentropy, args)

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
#' @param logits A `Tensor`. Must be one of the following types: `float32`, `float64`.
#' @param sparsemax A `Tensor`. Must have the same type as `logits`.
#' @param labels A `Tensor`. Must have the same type as `logits`.
#' @param name A name for the operation (optional). Returns: A `Tensor`. Has the same type as `logits`.
#'
#' @return A `Tensor`. Has the same type as `logits`.
#'
#' @export
loss_sparsemax <- function(logits, sparsemax, labels, name = NULL) {

  args <- list(
    logits = logits,
    sparsemax = sparsemax,
    labels = as.integer(labels),
    name = name
  )

  do.call(tfa$losses$sparsemax_loss, args)

}

attr(loss_sparsemax, "py_function_name") <- "sparsemax"


#' @title Triplet hard loss
#'
#' @description Computes the triplet loss with hard negative and hard positive mining.
#' @param y_true 1-D integer `Tensor` with shape [batch_size] of
#' multiclass integer labels.
#' @param y_pred 2-D float `Tensor` of embedding vectors. Embeddings should
#' be l2 normalized.
#' @param margin Float, margin term in the loss definition.
#' @param soft Boolean, if set, use the soft margin version.
#'
#' @return triplet_loss: float scalar with dtype of y_pred.
#' @export
loss_triplet_hard <- function(y_true, y_pred, margin = 1.0, soft = FALSE) {

  args <- list(
    y_true = y_true,
    y_pred = y_pred,
    margin = margin,
    soft = soft
  )

  do.call(tfa$losses$triplet_hard_loss, args)

}

attr(loss_triplet_hard, "py_function_name") <- "triplet_hard"


#' @title Triplet semihard loss
#'
#' @description Computes the triplet loss with semi-hard negative mining.
#' @param y_true 1-D integer Tensor with shape [batch_size] of multiclass integer labels.
#' @param y_pred 2-D float Tensor of embedding vectors. Embeddings should be l2 normalized.
#' @param margin Float, margin term in the loss definition.
#' @return triplet_loss: float scalar with dtype of y_pred.
#' @export
loss_triplet_semihard <- function(y_true, y_pred, margin = 1.0) {

  args <- list(
    y_true = y_true,
    y_pred = y_pred,
    margin = margin
  )

  do.call(tfa$losses$triplet_semihard_loss, args)

}

attr(loss_triplet_semihard, "py_function_name") <- "triplet_semihard"





