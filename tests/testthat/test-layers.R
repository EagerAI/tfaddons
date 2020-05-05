context("layers")

source("utils.R")

test_succeeds("layer_activation_gelu", {
  model = keras_model_sequential() %>%
    layer_activation_gelu(input_shape=c(5L,5L))
  model
})


test_succeeds("layer_filter_response_normalization", {
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                  activation = activation_gelu) %>%
    layer_filter_response_normalization()
  model
})

test_succeeds("layer_group_normalization", {
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                  activation = activation_gelu) %>%
    layer_group_normalization()
  model
})

test_succeeds("layer_instance_normalization", {
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                  activation = activation_gelu) %>%
    layer_instance_normalization()
  model
})

test_succeeds("layer_maxout", {
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                  activation = activation_gelu) %>%
    layer_maxout(1)
  model
})


test_succeeds("layer_poincare_normalize", {
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                  activation = activation_gelu) %>%
    layer_poincare_normalize(1)
  model
})


test_succeeds("layer_sparsemax", {
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = 10, kernel_size = c(3,3),input_shape = c(28,28,1),
                  activation = activation_gelu) %>%
    layer_sparsemax()
  model
})

test_succeeds("layer_weight_normalization", {
  model = keras_model_sequential() %>%
    layer_weight_normalization(input_shape = c(28L,28L,1L),
                               layer_conv_2d(filters = 10, kernel_size = c(3,3),
                                             activation = 'relu')) %>%
    layer_sparsemax()
  model
})


test_succeeds("layer_multi_head_attention", {
  mha = layer_multi_head_attention(head_size=128, num_heads=128)
  query = tf$random$uniform(list(32L, 20L, 200L)) # (batch_size, query_elements, query_depth)
  key = tf$random$uniform(list(32L, 15L, 300L)) # (batch_size, key_elements, key_depth)
  value = tf$random$uniform(list(32L, 15L, 400L)) # (batch_size, key_elements, value_depth)
  attention = mha(list(query, key, value)) # (batch_size, query_elements, value_depth)
})


