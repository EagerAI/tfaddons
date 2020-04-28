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
#' @return A tensor
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





#' @title Correlation Cost Layer.
#'
#' @details This layer implements the correlation operation from FlowNet
#' Learning Optical Flow with Convolutional Networks (Fischer et al.):
#' https://arxiv.org/abs/1504.06
#'
#'
#'
#' @param object Model or layer object
#' @param kernel_size An integer specifying the height and width of the
#' patch used to compute the per-patch costs.
#' @param max_displacement An integer specifying the maximum search radius
#' for each position.
#' @param stride_1 An integer specifying the stride length in the input.
#' @param stride_2 An integer specifying the stride length in the patch.
#' @param pad An integer specifying the paddings in height and width.
#' @param data_format Specifies the data format. Possible values are:
#' "channels_last" float [batch, height, width, channels] "channels_first"
#' float [batch, channels, height, width] Defaults to "channels_last".
#' @param ... additional parameters to pass
#' @importFrom keras create_layer
#' @return A tensor
#' @export
layer_correlation_cost <- function(object,
                                   kernel_size,
                                   max_displacement,
                                   stride_1,
                                   stride_2,
                                   pad,
                                   data_format,
                                   ...) {

  args = list(
    kernel_size = as.integer(kernel_size),
    max_displacement = as.integer(max_displacement),
    stride_1 = as.integer(stride_1),
    stride_2 = as.integer(stride_2),
    pad = as.integer(pad),
    data_format = data_format,
    ...
  )

  create_layer(tfa$layers$CorrelationCost, object, args)

}


#' @title FilterResponseNormalization
#'
#' @description Filter response normalization layer.
#'
#' @details Filter Response Normalization (FRN), a normalization
#' method that enables models trained with per-channel
#' normalization to achieve high accuracy. It performs better than
#' all other normalization techniques for small batches and is par
#' with Batch Normalization for bigger batch sizes.
#' @param object Model or layer object
#' @param epsilon Small positive float value added to variance to avoid dividing by zero.
#' @param axis List of axes that should be normalized. This should represent the spatial dimensions.
#' @param beta_initializer Initializer for the beta weight.
#' @param gamma_initializer Initializer for the gamma weight.
#' @param beta_regularizer Optional regularizer for the beta weight.
#' @param gamma_regularizer Optional regularizer for the gamma weight.
#' @param beta_constraint Optional constraint for the beta weight.
#' @param gamma_constraint Optional constraint for the gamma weight.
#' @param learned_epsilon (bool) Whether to add another learnable epsilon parameter or not.
#' @param learned_epsilon_constraint learned_epsilon_constraint
#' @param name Optional name for the layer
#' @note Input shape Arbitrary. Use the keyword argument `input_shape` (list of integers,
#' does not include the samples axis) when using this layer as the first layer in a model.
#' This layer, as of now, works on a 4-D tensor where the tensor should have the
#' shape [N X H X W X C] TODO: Add support for NCHW data format and FC layers. Output shape
#' Same shape as input. References - [Filter Response Normalization Layer: Eliminating Batch
#' Dependence in the training of Deep Neural Networks] (https://arxiv.org/abs/1911.09737)
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_filter_response_normalization <- function(object, epsilon = 1e-06,
                                                axis = c(1, 2),
                                                beta_initializer = "zeros",
                                                gamma_initializer = "ones",
                                                beta_regularizer = NULL,
                                                gamma_regularizer = NULL,
                                                beta_constraint = NULL,
                                                gamma_constraint = NULL,
                                                learned_epsilon = FALSE,
                                                learned_epsilon_constraint = NULL,
                                                name = NULL) {

  args <- list(
    epsilon = epsilon,
    axis = as.integer(axis),
    beta_initializer = beta_initializer,
    gamma_initializer = gamma_initializer,
    beta_regularizer = beta_regularizer,
    gamma_regularizer = gamma_regularizer,
    beta_constraint = beta_constraint,
    gamma_constraint = gamma_constraint,
    learned_epsilon = learned_epsilon,
    learned_epsilon_constraint = learned_epsilon_constraint,
    name = name
  )

  create_layer(tfa$layers$FilterResponseNormalization, object, args)

}


#' @title Gaussian Error Linear Unit
#'
#' @details A smoother version of ReLU generally used in the BERT or BERT architecture based
#' models. Original paper: https://arxiv.org/abs/1606.08415
#'
#' @note Input shape: Arbitrary. Use the keyword argument `input_shape` (tuple of integers, d
#' oes not include the samples axis) when using this layer as the first layer in a model.
#' @note  Output shape: Same shape as the input.
#' @param approximate (bool) Whether to apply approximation
#' @param object Model or layer object
#' @param ... additional parameters to pass
#' @importFrom keras create_layer
#' @return A tensor
#' @export
layer_activation_gelu <- function(object, approximate = TRUE, ...) {
  args = list(
    approximate = approximate,
    ...
  )

  create_layer(tfa$layers$GELU, object, args)

}

#' @title Group normalization layer
#'
#' @details Group Normalization divides the channels into groups and computes within each group
#' the mean and variance for normalization. Empirically, its accuracy is more stable than batch
#' norm in a wide range of small batch sizes, if learning rate is adjusted linearly with batch
#' sizes. Relation to Layer Normalization: If the number of groups is set to 1, then this operation
#' becomes identical to Layer Normalization. Relation to Instance Normalization: If the number of
#' groups is set to the input dimension (number of groups is equal to number of channels), then this
#' operation becomes identical to Instance Normalization.
#' @param object Model or layer object
#' @param groups Integer, the number of groups for Group Normalization. Can be in the range [1, N]
#' where N is the input dimension. The input dimension must be divisible by the number of groups.
#' @param axis Integer, the axis that should be normalized.
#' @param epsilon Small float added to variance to avoid dividing by zero.
#' @param center If TRUE, add offset of beta to normalized tensor. If False, beta is ignored.
#' @param scale If TRUE, multiply by gamma. If False, gamma is not used.
#' @param beta_initializer Initializer for the beta weight.
#' @param gamma_initializer Initializer for the gamma weight.
#' @param beta_regularizer Optional regularizer for the beta weight.
#' @param gamma_regularizer Optional regularizer for the gamma weight.
#' @param beta_constraint Optional constraint for the beta weight.
#' @param gamma_constraint Optional constraint for the gamma weight.
#' @param ... additional parameters to pass
#'
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_group_normalization <- function(object,
                                      groups = 2,
                                      axis = -1,
                                      epsilon = 0.001,
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

  create_layer(tfa$layers$GroupNormalization, object, args)

}


#' @title Maxout layer
#'
#' @details "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza,
#' Aaron Courville, Yoshua Bengio. https://arxiv.org/abs/1302.4389 Usually the operation
#' is performed in the filter/channel dimension. This can also be used after Dense layers
#' to reduce number of features.
#' @param object Model or layer object
#'
#'
#'
#' @param num_units Specifies how many features will remain after maxout in the axis dimension
#' (usually channel). This must be a factor of number of features.
#' @param axis The dimension where max pooling will be performed. Default is the last dimension.
#'
#' @param ... additional parameters to pass
#'
#'
#'
#'
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_maxout <- function(object, num_units,
                         axis = -1, ...) {

  args = list(
    num_units = as.integer(num_units),
    axis = as.integer(axis),
    ...
  )

  create_layer(tfa$layers$Maxout, object, args)
}


#' @title Project into the Poincare ball with norm <= 1.0 - epsilon
#'
#' @details https://en.wikipedia.org/wiki/Poincare_ball_model Used in Poincare Embeddings
#' for Learning Hierarchical Representations Maximilian Nickel, Douwe Kiela
#' https://arxiv.org/pdf/1705.08039.pdf For a 1-D tensor with axis = 0, computes
#'
#' @param object Model or layer object
#' @param axis Axis along which to normalize.  A scalar or a vector of integers.
#' @param epsilon A small deviation from the edge of the unit sphere for numerical stability.
#' @param ... additional parameters to pass
#'
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_poincare_normalize <- function(object,
                                     axis = 1,
                                     epsilon = 1e-05,
                                     ...) {
  args = list(axis = as.integer(axis),
              epsilon = epsilon,
              ...)

  create_layer(tfa$layers$PoincareNormalize, object, args)

}


#' @title Sparsemax activation function
#'
#' @details The output shape is the same as the input shape. https://arxiv.org/abs/1602.02068
#'
#' @param object Model or layer object
#' @param axis Integer, axis along which the sparsemax normalization is applied.
#' @param ... additional parameters to pass
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_sparsemax <- function(object,
                            axis = -1,
                            ...) {
  args = list(axis = as.integer(axis), ...)

  create_layer(tfa$layers$PoincareNormalize, object, args)

}


#' @title Weight Normalization layer
#'
#' @details This wrapper reparameterizes a layer by decoupling the weight's magnitude and
#' direction.
#' This speeds up convergence by improving the conditioning of the optimization problem.
#' Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural
#' Networks: https://arxiv.org/abs/1602.07868 Tim Salimans, Diederik P. Kingma (2016)
#' WeightNormalization wrapper works for keras and tf layers.
#'
#' @param object Model or layer object
#' @param layer a layer instance.
#' @param data_init If `TRUE` use data dependent variable initialization
#' @param ... additional parameters to pass
#'
#' @examples
#'
#' \dontrun{
#'
#' model= keras_model_sequential() %>%
#' layer_weight_normalization(
#' layer_conv_2d(filters = 2, kernel_size = 2, activation = 'relu'),
#' input_shape = c(32L, 32L, 3L))
#' model
#'
#'
#' }
#'
#' @importFrom keras create_layer
#'
#' @return A tensor
#' @export
layer_weight_normalization <- function(object,
                                       layer,
                                       data_init = TRUE,
                                       ...) {

  args = list(layer = layer,
              data_init = data_init,
              ...)

  create_layer(tfa$layers$WeightNormalization, object, args)

}









