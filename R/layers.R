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
layer_instance_normalization <- function(object,
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





#' @title Keras-based multi head attention layer
#'
#' @description MultiHead Attention layer.
#'
#' @details Defines the MultiHead Attention operation as defined in
#' [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes
#' in a `query`, `key` and `value` tensors returns the dot-product attention
#' between them.
#'
#' @examples
#'
#' \dontrun{
#'
#' ```
#' mha = layer_multi_head_attention(head_size=128, num_heads=128)
#' query = tf$random$uniform(list(32L, 20L, 200L)) # (batch_size, query_elements, query_depth)
#' key = tf$random$uniform(list(32L, 15L, 300L)) # (batch_size, key_elements, key_depth)
#' value = tf$random$uniform(list(32L, 15L, 400L)) # (batch_size, key_elements, value_depth)
#' attention = mha(list(query, key, value)) # (batch_size, query_elements, value_depth)
#' ```
#'
#' # If `value` is not given then internally `value = key` will be used:
#' ```
#' mha = layer_multi_head_attention(head_size=128, num_heads=128)
#' query = tf$random$uniform(list(32L, 20L, 200L)) # (batch_size, query_elements, query_depth)
#' key = tf$random$uniform(list(32L, 15L, 300L)) # (batch_size, key_elements, key_depth)
#' attention = mha(list(query, key)) # (batch_size, query_elements, value_depth)
#' ```
#'
#' }
#'
#' @param object Model or layer object
#' @param head_size int, dimensionality of the `query`, `key` and `value` tensors after the linear transformation.
#' @param num_heads int, number of attention heads.
#' @param output_size int, dimensionality of the output space, if `NULL` then the input dimension of `value` or `key` will be used, default `NULL`.
#' @param dropout float, `rate` parameter for the dropout layer that is applied to attention after softmax, default `0`.
#' @param use_projection_bias bool, whether to use a bias term after the linear output projection.
#' @param return_attn_coef bool, if `TRUE`, return the attention coefficients as an additional output argument.
#' @param kernel_initializer initializer, initializer for the kernel weights.
#' @param kernel_regularizer regularizer, regularizer for the kernel weights.
#' @param kernel_constraint constraint, constraint for the kernel weights.
#' @param bias_initializer initializer, initializer for the bias weights.
#' @param bias_regularizer regularizer, regularizer for the bias weights.
#' @param bias_constraint constraint, constraint for the bias weights.
#' @param ... additional parameters to pass
#' @importFrom keras create_layer
#'
#' @export
layer_multi_head_attention <- function(object, head_size, num_heads, output_size = NULL, dropout = 0.0,
                                       use_projection_bias = TRUE, return_attn_coef = FALSE,
                                       kernel_initializer = "glorot_uniform", kernel_regularizer = NULL,
                                       kernel_constraint = NULL, bias_initializer = "zeros",
                                       bias_regularizer = NULL, bias_constraint = NULL, ...) {

  args = list(
    head_size = as.integer(head_size),
    num_heads = as.integer(num_heads),
    output_size = output_size,
    dropout = dropout,
    use_projection_bias = use_projection_bias,
    return_attn_coef = return_attn_coef,
    kernel_initializer = kernel_initializer,
    kernel_regularizer = kernel_regularizer,
    kernel_constraint = kernel_constraint,
    bias_initializer = bias_initializer,
    bias_regularizer = bias_regularizer,
    bias_constraint = bias_constraint,
    ...
  )

  if(!is.null(output_size)) {
    args$output_size <- as.integer(output_size)
  }

  create_layer(tfa$layers$MultiHeadAttention, object, args)

}






