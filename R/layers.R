#' @title Instance normalization layer
#'
#' @details Instance Normalization is an specific case of ```GroupNormalizationsince```
#' it normalizes all features of one channel. The Groupsize is equal to the channel
#' size. Empirically, its accuracy is more stable than batch norm in a wide range of
#' small batch sizes, if learning rate is adjusted linearly with batch sizes.
#'
#' @param object Model or layer object
#' @param groups Integer, the number of groups for Group Normalization. Can be in the
#' range [1, N] where N is the input dimension. The input dimension must be divisible
#' by the number of groups.
#' @param axis Integer, the axis that should be normalized.
#' @param epsilon Small float added to variance to avoid dividing by zero.
#' @param center If TRUE, add offset of `beta` to normalized tensor. If FALSE, `beta` is ignored.
#' @param scale If TRUE, multiply by `gamma`. If FALSE, `gamma` is not used.
#' @param beta_initializer Initializer for the beta weight.
#' @param gamma_initializer Initializer for the gamma weight.
#' @param beta_regularizer Optional regularizer for the beta weight.
#' @param gamma_regularizer Optional regularizer for the gamma weight.
#' @param beta_constraint Optional constraint for the beta weight.
#' @param gamma_constraint Optional constraint for the gamma weight.
#' @param ... additional parameters to pass
#'
#' @references [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
#'
#'
#'
#' @importFrom keras create_layer
#'
#'
#'
#'
#' @return A tensor
#'
#' @export
layer_normalization <- function(object,
                                groups = 2,
                                axis = -1,
                                epsilon = 1e-3,
                                center = TRUE,
                                scale = TRUE,
                                beta_initializer = 'zeros',
                                gamma_initializer = 'ones',
                                beta_regularizer = NULL,
                                gamma_regularizer = NULL,
                                beta_constraint = NULL,
                                gamma_constraint = NULL,
                                ...) {
  args = list(
    groups = as.integer(groups),
    axis = as.integer(axis),
    epsilon = epsilon,
    center = center,
    scale = scale,
    beta_initializer = beta_initializer,
    gamma_initializer = gamma_initializer,
    beta_regularizer = beta_regularizer,
    gamma_regularizer = gamma_regularizer,
    beta_constraint = beta_constraint,
    gamma_constraint = gamma_constraint,
    ...

  )

  create_layer(tfa$layers$InstanceNormalization, object, args)
}





