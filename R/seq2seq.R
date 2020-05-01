#' @title Attention Wrapper
#'
#'
#'
#' @param object Model or layer object
#' @param cell An instance of RNNCell.
#' @param attention_mechanism A list of AttentionMechanism instances or a single instance.
#' @param attention_layer_size A list of Python integers or a single Python integer, the
#' depth of the attention (output) layer(s). If `NULL` (default), use the context as attention
#' at each time step. Otherwise, feed the context and cell output into the attention layer
#' to generate attention at each time step. If attention_mechanism is a list,
#' attention_layer_size must be a list of the same length. If attention_layer is set, this
#' must be `NULL`. If attention_fn is set, it must guaranteed that the outputs of `attention_fn`
#' also meet the above requirements.
#' @param alignment_history Python boolean, whether to store alignment history from all time
#' steps in the final output state (currently stored as a time major TensorArray on which you
#' must call stack()).
#' @param cell_input_fn (optional) A callable.
#' The default is: lambda inputs, attention: tf$concat(list(inputs, attention), -1).
#' @param output_attention Python bool. If True (default), the output at each time step is the
#' attention value. This is the behavior of Luong-style attention mechanisms. If FALSE, the output
#' at each time step is the output of cell. This is the behavior of Bhadanau-style attention
#' mechanisms. In both cases, the attention tensor is propagated to the next time step via the
#' state and is used there. This flag only controls whether the attention mechanism is propagated
#' up to the next cell in an RNN stack or to the top RNN output.
#' @param initial_cell_state The initial state value to use for the cell when the user calls
#' get_initial_state(). Note that if this value is provided now, and the user uses a batch_size
#' argument of get_initial_state which does not match the batch size of initial_cell_state,
#' proper behavior is not guaranteed.
#' @param name Name to use when creating ops.
#' @param attention_layer A list of tf$keras$layers$Layer instances or a single tf$keras$layers$Layer
#' instance taking the context and cell output as inputs to generate attention at each time step.
#' If `NULL` (default), use the context as attention at each time step. If attention_mechanism is a list,
#' attention_layer must be a list of the same length. If attention_layers_size is set, this must be `NULL`.
#' @param attention_fn An optional callable function that allows users to provide their own customized
#' attention function, which takes input (attention_mechanism, cell_output, attention_state, attention_layer)
#' and outputs (attention, alignments, next_attention_state). If provided, the attention_layer_size should
#' be the size of the outputs of attention_fn.
#' @param ... Other keyword arguments to pass
#' @importFrom purrr map
#'
#' @note If you are using the `decoder_beam_search` with a cell wrapped in `AttentionWrapper`, then
#' you must ensure that:
#'  - The encoder output has been tiled to `beam_width` via `tile_batch` (NOT `tf$tile`).
#'  - The `batch_size` argument passed to the `get_initial_state` method of this wrapper
#' is equal to `true_batch_size * beam_width`.
#'  - The initial state created with `get_initial_state` above contains a `cell_state` value
#'  containing properly tiled final state from the encoder.
#' @importFrom keras create_layer
#' @return None
#'
#'
#' @export
attention_wrapper <- function(object,
                              cell,
                              attention_mechanism,
                              attention_layer_size = NULL,
                              alignment_history = FALSE,
                              cell_input_fn = NULL,
                              output_attention = TRUE,
                              initial_cell_state = NULL,
                              name = NULL,
                              attention_layer = NULL,
                              attention_fn = NULL,
                              ...) {

  args = list(
    cell = cell,
    attention_mechanism = attention_mechanism,
    attention_layer_size = attention_layer_size,
    alignment_history = alignment_history,
    cell_input_fn = cell_input_fn,
    output_attention = output_attention,
    initial_cell_state =  initial_cell_state,
    name = name,
    attention_layer = attention_layer,
    attention_fn = attention_fn,
    ...
  )

  if(is.list(attention_layer_size)) {
    args$attention_layer_size <- map(attention_layer_size, ~ as.character(.) %>% as.integer)
  } else if(is.vector(attention_layer_size)) {
    args$attention_layer_size <- as.integer(as.character(attention_layer_size))
  }

  create_layer(tfa$seq2seq$AttentionWrapper, object, args)

}


#' @title Attention Wrapper State
#'
#' @description `namedlist` storing the state of a `attention_wrapper`.
#'
#' @param object Model or layer object
#' @param cell_state The state of the wrapped RNNCell at the previous time step.
#' @param attention The attention emitted at the previous time step.
#' @param alignments A single or tuple of Tensor(s) containing the alignments
#' emitted at the previous time step for each attention mechanism.
#' @param alignment_history (if enabled) a single or tuple of TensorArray(s)
#' containing alignment matrices from all time steps for each attention mechanism.
#' Call stack() on each to convert to a Tensor.
#' @param attention_state A single or tuple of nested objects containing attention
#' mechanism state for each attention mechanism. The objects may contain Tensors or
#' TensorArrays.
#' @importFrom keras create_layer
#' @return None
#' @export
attention_wrapper_state <- function(object,
                                    cell_state, attention,
                                    alignments, alignment_history,
                                    attention_state) {

  args <- list(
    cell_state = cell_state,
    attention = attention,
    alignments = alignments,
    alignment_history = alignment_history,
    attention_state = attention_state
  )

  create_layer(tfa$seq2seq$AttentionWrapperState, object, args)

}



#' @title Bahdanau Attention
#'
#' @description Implements Bahdanau-style (additive) attention
#'
#' @details This attention has two forms. The first is Bahdanau attention, as described in:
#' Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. "Neural Machine Translation by Jointly
#' Learning to Align and Translate." ICLR 2015. https://arxiv.org/abs/1409.0473 The second
#' is the normalized form. This form is inspired by the weight normalization article:
#' Tim Salimans, Diederik P. Kingma. "Weight Normalization: A Simple Reparameterization
#' to Accelerate Training of Deep Neural Networks." https://arxiv.org/abs/1602.07868
#' To enable the second form, construct the object with parameter `normalize=TRUE`.
#'
#'
#' @param object Model or layer object
#' @param units The depth of the query mechanism.
#' @param memory The memory to query; usually the output of an RNN encoder. This tensor
#' should be shaped [batch_size, max_time, ...].
#' @param memory_sequence_length (optional): Sequence lengths for the batch entries in
#' memory. If provided, the memory tensor rows are masked with zeros for values past the
#' respective sequence lengths.
#' @param normalize boolean. Whether to normalize the energy term.
#' @param probability_fn (optional) string, the name of function to convert the attention
#' score to probabilities. The default is softmax which is tf.nn.softmax. Other options is hardmax, which is hardmax() within this module. Any other value will result into validation error. Default to use softmax.
#' @param kernel_initializer (optional), the name of the initializer for the attention kernel.
#' @param dtype The data type for the query and memory layers of the attention mechanism.
#' @param name Name to use when creating ops.
#' @param ... A list that contains other common arguments for layer creation.
#'
#' @importFrom keras create_layer
#'
#' @return None
#'
#'
#'
#' @export
attention_bahdanau <- function(object,
                               units,
                               memory = NULL,
                               memory_sequence_length = NULL,
                               normalize = FALSE,
                               probability_fn = 'softmax',
                               kernel_initializer = 'glorot_uniform',
                               dtype = NULL,
                               name = 'BahdanauAttention',
                               ...) {
  args = list(
    units = as.integer(units),
    memory = memory,
    memory_sequence_length = memory_sequence_length,
    normalize = normalize,
    probability_fn = probability_fn,
    kernel_initializer = kernel_initializer,
    dtype = dtype,
    name = name,
    ...
  )

  if(!is.null(memory_sequence_length))
    args$memory_sequence_length <- as.integer(memory_sequence_length)

  create_layer(tfa$seq2seq$BahdanauAttention, object, args)

}



#' @title Bahdanau Monotonic Attention
#'
#' @description Monotonic attention mechanism with Bahadanau-style energy function.
#' @details This type of attention enforces a monotonic constraint on the attention
#' distributions; that is once the model attends to a given point in the memory it
#' can't attend to any prior points at subsequence output timesteps. It achieves this
#' by using the _monotonic_probability_fn instead of softmax to construct its attention
#' distributions. Since the attention scores are passed through a sigmoid, a learnable
#' scalar bias parameter is applied after the score function and before the sigmoid.
#' Otherwise, it is equivalent to BahdanauAttention. This approach is proposed in
#'
#' Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
#' "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
#' ICML 2017. https://arxiv.org/abs/1704.00784
#'
#'
#'
#'
#' @param object Model or layer object
#' @param units The depth of the query mechanism.
#' @param memory The memory to query; usually the output of an RNN encoder. This tensor should be shaped [batch_size, max_time, ...].
#' @param memory_sequence_length (optional): Sequence lengths for the batch entries in memory. If provided, the memory tensor rows are masked with zeros for values past the respective sequence lengths.
#' @param normalize Python boolean. Whether to normalize the energy term.
#' @param sigmoid_noise Standard deviation of pre-sigmoid noise. See the docstring for _monotonic_probability_fn for more information.
#' @param sigmoid_noise_seed (optional) Random seed for pre-sigmoid noise.
#' @param score_bias_init Initial value for score bias scalar. It's recommended to initialize this to a negative value when the length of the memory is large.
#' @param mode How to compute the attention distribution. Must be one of 'recursive', 'parallel', or 'hard'. See the docstring for tfa.seq2seq.monotonic_attention for more information.
#' @param kernel_initializer (optional), the name of the initializer for the attention kernel.
#' @param dtype The data type for the query and memory layers of the attention mechanism.
#' @param name Name to use when creating ops.
#'
#'
#'
#'
#' @param ... A list that contains other common arguments for layer creation.
#'
#' @importFrom keras create_layer
#'
#' @return None
#'
#'
#'
#' @export
layer_bahdanau_monotonic_attention <- function(object,
                                               units,
                                               memory = NULL,
                                               memory_sequence_length = NULL,
                                               normalize = FALSE,
                                               sigmoid_noise = 0.0,
                                               sigmoid_noise_seed = NULL,
                                               score_bias_init = 0.0,
                                               mode = 'parallel',
                                               kernel_initializer = 'glorot_uniform',
                                               dtype = NULL,
                                               name = 'BahdanauMonotonicAttention',
                                               ...) {
  args =list(
    units = as.integer(units),
    memory = memory,
    memory_sequence_length = memory_sequence_length,
    normalize = normalize,
    sigmoid_noise = sigmoid_noise,
    sigmoid_noise_seed = sigmoid_noise_seed,
    score_bias_init = score_bias_init,
    mode = mode,
    kernel_initializer = kernel_initializer,
    dtype = dtype,
    name = name,
    ...
  )

  create_layer(tfa$seq2seq$BahdanauMonotonicAttention, object, args)
}



#' @title Base Decoder
#'
#' @description An RNN Decoder that is based on a Keras layer.
#'
#'
#'
#' @param object Model or layer object
#'
#' @param cell An RNNCell instance.
#' @param sampler A Sampler instance.
#' @param output_layer (Optional) An instance of tf$layers$Layer, i.e., tf$layers$Dense.
#' Optional layer to apply to the RNN output prior to storing the result or sampling.
#'
#'
#' @param ... Other keyword arguments for layer creation.
#' @importFrom keras create_layer
#'
#' @return None
#' @export
layer_base_decoder <- function(object,
                               cell,
                               sampler,
                               output_layer = NULL,
                               ...) {

  args = list(
    cell = cell,
    sampler = sampler,
    output_layer = output_layer,
    ...
  )

  create_layer(tfa$seq2seq$BaseDecoder, object, args)

}



#' @title Basic Decoder
#'
#' @inheritParams layer_base_decoder
#' @importFrom keras create_layer
#'
#' @return None
#' @export
layer_basic_decoder <- function(object,
                               cell,
                               sampler,
                               output_layer = NULL,
                               ...) {

  args = list(
    cell = cell,
    sampler = sampler,
    output_layer = output_layer,
    ...
  )

  create_layer(tfa$seq2seq$BaseDecoder, object, args)

}




#' @title Basic decoder output
#'
#'
#'
#' @param rnn_output the output of RNN cell
#' @param sample_id the `id` of the sample
#' @return None
#' @export
layer_basic_decoder_output <- function(rnn_output, sample_id) {

  args <- list(
    rnn_output = rnn_output,
    sample_id = sample_id
  )

  do.call(tfa$seq2seq$BasicDecoderOutput, args)

}


#' @title BeamSearch sampling decoder
#'
#' @note If you are using the `BeamSearchDecoder` with a cell wrapped in
#' `AttentionWrapper`, then you must ensure that:
#'  - The encoder output has been tiled to `beam_width` via
#'  `tfa.seq2seq.tile_batch` (NOT `tf.tile`).
#'  - The `batch_size` argument passed to the `get_initial_state` method of
#'  this wrapper is equal to `true_batch_size * beam_width`.
#'  - The initial state created with `get_initial_state` above contains a
#'  `cell_state` value containing properly tiled final state from the encoder.
#'
#'
#' @param object Model or layer object
#' @param cell An RNNCell instance.
#' @param beam_width integer, the number of beams.
#' @param embedding_fn A callable that takes a vector tensor of ids (argmax ids).
#' @param output_layer (Optional) An instance of tf.keras.layers.Layer,
#' i.e., tf$keras$layers$Dense. Optional layer to apply to the RNN output prior
#' to storing the result or sampling.
#' @param length_penalty_weight Float weight to penalize length. Disabled with 0.0.
#' @param coverage_penalty_weight Float weight to penalize the coverage of source
#' sentence. Disabled with 0.0.
#' @param reorder_tensor_arrays If `TRUE`, TensorArrays' elements within the cell
#' state will be reordered according to the beam search path. If the TensorArray
#' can be reordered, the stacked form will be returned. Otherwise, the TensorArray
#' will be returned as is. Set this flag to False if the cell state contains
#' TensorArrays that are not amenable to reordering.
#' @param ... A list, other keyword arguments for initialization.
#'
#'
#'
#' @importFrom keras create_layer
#'
#' @return None
#' @export
layer_beam_search_decoder <- function(object,
                                      cell,
                                      beam_width,
                                      embedding_fn = NULL,
                                      output_layer = NULL,
                                      length_penalty_weight = 0.0,
                                      coverage_penalty_weight = 0.0,
                                      reorder_tensor_arrays = TRUE,
                                      ...) {

  args = list(
    cell = cell,
    beam_width = as.integer(beam_width),
    embedding_fn = embedding_fn,
    output_layer = output_layer,
    length_penalty_weight = length_penalty_weight,
    coverage_penalty_weight = coverage_penalty_weight,
    reorder_tensor_arrays = reorder_tensor_arrays,
    ...
  )

  create_layer(tfa$seq2seq$BeamSearchDecoder, object, args)

}


#' @title Beam Search Decoder Output
#'
#'
#'
#' @param scores calculate the scores for each beam
#' @param predicted_ids The final prediction. A tensor of shape
#' `[batch_size, T, beam_width]` (or `[T, batch_size, beam_width]` if `output_time_major`
#' is `TRUE`). Beams are ordered from best to worst.
#' @param parent_ids The parent ids of shape `[max_time, batch_size, beam_width]`.
#'
#' @export
layer_beam_search_decoder_output <- function(scores, predicted_ids, parent_ids) {

  python_function_result <- tfa$seq2seq$BeamSearchDecoderOutput(
    scores = scores,
    predicted_ids = predicted_ids,
    parent_ids = parent_ids
  )

  do.call(tfa$seq2seq$BeamSearchDecoderOutput, args)

}




#' @title Beam Search Decoder State
#'
#'
#' @param _cls _cls
#' @param cell_state cell_state
#' @param log_probs log_probs
#' @param finished finished
#' @param lengths lengths
#' @param accumulated_attention_probs accumulated_attention_probs
#'
#' @export
layer_beam_search_decoder_state <- function(cell_state, log_probs,
                                            finished, lengths,
                                            accumulated_attention_probs) {

  args <- list(
    cell_state = cell_state,
    log_probs = log_probs,
    finished = finished,
    lengths = lengths,
    accumulated_attention_probs = accumulated_attention_probs
  )

  do.call(tfa$seq2seq$BeamSearchDecoderState, args)

}

















