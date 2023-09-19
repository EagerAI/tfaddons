#' @title Attention Wrapper
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
    args$attention_layer_size <- map(args$attention_layer_size, ~ as.character(.) %>% as.integer)
  } else if(is.vector(attention_layer_size)) {
    args$attention_layer_size <- as.integer(as.character(args$attention_layer_size))
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
#' score to probabilities. The default is softmax which is tf.nn.softmax. Other options is hardmax,
#' which is hardmax() within this module. Any other value will result into validation
#' error. Default to use softmax.
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
#' @param memory The memory to query; usually the output of an RNN encoder. This tensor
#' should be shaped [batch_size, max_time, ...].
#' @param memory_sequence_length (optional): Sequence lengths for the batch entries in memory.
#' If provided, the memory tensor rows are masked with zeros for values past the respective
#' sequence lengths.
#' @param normalize Python boolean. Whether to normalize the energy term.
#' @param sigmoid_noise Standard deviation of pre-sigmoid noise. See the docstring for
#' `_monotonic_probability_fn` for more information.
#' @param sigmoid_noise_seed (optional) Random seed for pre-sigmoid noise.
#' @param score_bias_init Initial value for score bias scalar. It's recommended to initialize
#' this to a negative value when the length of the memory is large.
#' @param mode How to compute the attention distribution. Must be one of 'recursive',
#' 'parallel', or 'hard'. See the docstring for tfa.seq2seq.monotonic_attention for more information.
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
attention_bahdanau_monotonic <- function(object,
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
decoder_base <- function(object,
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
#' @importFrom keras create_layer
#' @param object Model or layer object
#' @param cell An RNNCell instance.
#' @param sampler A Sampler instance.
#' @param output_layer (Optional) An instance of tf$layers$Layer,
#' i.e., tf$layers$Dense. Optional layer to apply to the RNN output
#' prior to storing the result or sampling.
#' @param ... Other keyword arguments for layer creation.
#' @return None
#' @export
decoder_basic <- function(object,
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

  create_layer(tfa$seq2seq$BasicDecoder, object, args)

}




#' @title Basic decoder output
#'
#' @param rnn_output the output of RNN cell
#' @param sample_id the `id` of the sample
#' @return None
#' @export
decoder_basic_output <- function(rnn_output, sample_id) {

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
#'  `tile_batch()` (NOT `tf$tile`).
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
decoder_beam_search <- function(object,
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
#' @param scores calculate the scores for each beam
#' @param predicted_ids The final prediction. A tensor of shape
#' `[batch_size, T, beam_width]` (or `[T, batch_size, beam_width]` if `output_time_major`
#' is `TRUE`). Beams are ordered from best to worst.
#' @param parent_ids The parent ids of shape `[max_time, batch_size, beam_width]`.
#' @return None
#' @export
decoder_beam_search_output <- function(scores, predicted_ids, parent_ids) {

  python_function_result <- tfa$seq2seq$BeamSearchDecoderOutput(
    scores = scores,
    predicted_ids = predicted_ids,
    parent_ids = parent_ids
  )

  do.call(tfa$seq2seq$BeamSearchDecoderOutput, args)

}




#' @title Beam Search Decoder State
#'
#' @param cell_state cell_state
#' @param log_probs log_probs
#' @param finished finished
#' @param lengths lengths
#' @param accumulated_attention_probs accumulated_attention_probs
#' @return None
#' @export
decoder_beam_search_state <- function(cell_state, log_probs,
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

#' @title Base abstract class that allows the user to customize sampling.
#'
#' @param initialize_fn callable that returns (finished, next_inputs) for the first iteration.
#' @param sample_fn callable that takes (time, outputs, state) and emits tensor sample_ids.
#' @param next_inputs_fn callable that takes (time, outputs, state, sample_ids) and emits
#' (finished, next_inputs, next_state).
#' @param sample_ids_shape Either a list of integers, or a 1-D Tensor of type int32, the
#' shape of each value in the sample_ids batch. Defaults to a scalar.
#' @param sample_ids_dtype The dtype of the sample_ids tensor. Defaults to int32.
#'
#'
#'
#' @return None
#'
#'
#' @export
sampler_custom <- function(initialize_fn,
                                 sample_fn,
                                 next_inputs_fn,
                                 sample_ids_shape = NULL,
                                 sample_ids_dtype = NULL) {

  args = list(
    initialize_fn,
    sample_fn,
    next_inputs_fn,
    sample_ids_shape = NULL,
    sample_ids_dtype = NULL
  )

  do.call(tfa$seq2seq$CustomSampler, args)
}


#' @title An RNN Decoder abstract interface object.
#'
#' @details
#' - inputs: (structure of) tensors and TensorArrays that is passed as input to the RNNCell
#' composing the decoder, at each time step.
#' - state: (structure of) tensors and TensorArrays that is passed to the RNNCell instance as the state.
#' - finished: boolean tensor telling whether each sequence in the batch is finished.
#' - training: boolean whether it should behave in training mode or in inference mode.
#' - outputs: Instance of BasicDecoderOutput. Result of the decoding, at each time step.
#'
#'
#' @param ... arguments to pass
#'
#' @return None
#' @export
decoder <- function(...) {
  args = list(...)

  do.call(tfa$seq2seq$Decoder, args)
}

#' @title Dynamic decode
#'
#' @description Perform dynamic decoding with `decoder`.
#'
#' @details Calls `initialize()` once and `step()` repeatedly on the Decoder object.
#'
#' @param decoder A `Decoder` instance.
#' @param output_time_major boolean. Default: `FALSE` (batch major). If `TRUE`, outputs
#' are returned as time major tensors (this mode is faster). Otherwise, outputs are returned
#' as batch major tensors (this adds extra time to the computation).
#' @param impute_finished boolean. If `TRUE`, then states for batch entries which are
#' marked as finished get copied through and the corresponding outputs get zeroed out. This
#' causes some slowdown at each time step, but ensures that the final state and outputs have
#' the correct values and that backprop ignores time steps that were marked as finished.
#' @param maximum_iterations `int32` scalar, maximum allowed number of decoding steps. Default
#' is `NULL` (decode until the decoder is fully done).
#' @param parallel_iterations Argument passed to `tf$while_loop`.
#' @param swap_memory Argument passed to `tf$while_loop`.
#' @param training boolean. Indicates whether the layer should behave in training mode or
#' in inference mode. Only relevant when `dropout` or `recurrent_dropout` is used.
#' @param scope Optional variable scope to use.
#' @param ... A list, other keyword arguments for
#' dynamic_decode. It might contain arguments for `BaseDecoder` to initialize, which takes
#' all tensor inputs during `call()`.
#'
#' @return `(final_outputs, final_state, final_sequence_lengths)`.
#'
#' @section Raises:
#' TypeError: if `decoder` is not an instance of `Decoder`. ValueError: if `maximum_iterations`
#' is provided but is not a scalar.
#'
#' @export
decode_dynamic <- function(decoder, output_time_major = FALSE,
                           impute_finished = FALSE, maximum_iterations = NULL,
                           parallel_iterations = 32L, swap_memory = FALSE,
                           training = NULL, scope = NULL, ...) {

  args <- list(
    decoder = decoder,
    output_time_major = output_time_major,
    impute_finished = impute_finished,
    maximum_iterations = maximum_iterations,
    parallel_iterations = parallel_iterations,
    swap_memory = swap_memory,
    training = training,
    scope = scope,
    ...
  )

  if(!is.null(maximum_iterations))
    args$maximum_iterations <- as.integer(args$maximum_iterations)

  do.call(tfa$seq2seq$dynamic_decode, args)

}



#' @title Final Beam Search Decoder Output
#'
#' @description Final outputs returned by the beam search after all decoding is finished.
#'
#'
#' @param predicted_ids The final prediction. A tensor of shape `[batch_size, T, beam_width]`
#' (or `[T, batch_size, beam_width]` if `output_time_major` is TRUE). Beams are ordered from
#' best to worst.
#' @param beam_search_decoder_output An instance of `BeamSearchDecoderOutput` that describes
#' the state of the beam search.
#' @return None
#' @export
decoder_final_beam_search_output <- function(predicted_ids, beam_search_decoder_output) {

  args <- list(
    predicted_ids = predicted_ids,
    beam_search_decoder_output = beam_search_decoder_output
  )

  do.call(tfa$seq2seq$FinalBeamSearchDecoderOutput, args)

}



#' @title Gather tree
#'
#' @param step_ids requires the step id
#' @param parent_ids The parent ids of shape `[max_time, batch_size, beam_width]`.
#' @param max_sequence_lengths get max_sequence_length across all beams for each batch.
#' @param end_token `int32` scalar, the token that marks end of decoding.
#'
#'
#' @return None
#' @export
gather_tree <- function(step_ids, parent_ids,
                              max_sequence_lengths, end_token) {

  args = list(
    step_ids = step_ids, parent_ids = parent_ids,
    max_sequence_lengths = max_sequence_lengths,
    end_token = end_token
  )

  if(!is.null(max_sequence_lengths))
    args$max_sequence_lengths <- as.integer(args$max_sequence_lengths)

  if(!is.null(end_token))
    args$end_token <- as.integer(args$end_token)

  do.call(tfa$seq2seq$gather_tree, args)

}


#' @title Gather tree from array
#'
#' @description Calculates the full beams for `TensorArray`s.
#'
#'
#' @param t A stacked `TensorArray` of size `max_time` that contains `Tensor`s of
#' shape `[batch_size, beam_width, s]` or `[batch_size * beam_width, s]` where `s`
#' is the depth shape.
#' @param parent_ids The parent ids of shape `[max_time, batch_size, beam_width]`.
#' @param sequence_length The sequence length of shape `[batch_size, beam_width]`.
#'
#' @return A `Tensor` which is a stacked `TensorArray` of the same size and type as
#' `t` and where beams are sorted in each `Tensor` according to `parent_ids`.
#'
#' @export
gather_tree_from_array <- function(t, parent_ids, sequence_length) {

  args <- list(
    t = t,
    parent_ids = parent_ids,
    sequence_length = as.integer(sequence_length)
  )

  do.call(tfa$seq2seq$gather_tree_from_array, args)

}


#' @title Greedy Embedding Sampler
#'
#' @description A sampler for use during inference.
#'
#' @details Uses the argmax of the output (treated as logits) and passes the result through
#' an embedding layer to get the next input.
#' @param embedding_fn A optional callable that takes a vector tensor of ids (argmax ids),
#' or the params argument for embedding_lookup. The returned tensor will be passed to the
#' decoder input. Default to use tf$nn$embedding_lookup.
#'
#'
#'
#'
#'
#' @return None
#' @export
sampler_greedy_embedding <- function(embedding_fn = NULL) {

  args = list(
    embedding_fn = embedding_fn
  )

  do.call(tfa$seq2seq$GreedyEmbeddingSampler, args)

}


#' @title Hardmax
#'
#' @description Returns batched one-hot vectors.
#'
#' @details The depth index containing the `1` is that of the maximum logit value.
#'
#' @param logits A batch tensor of logit values.
#' @param name Name to use when creating ops.
#'
#' @return A batched one-hot tensor.
#'
#' @export
hardmax <- function(logits, name = NULL) {

  args <- list(
    logits = logits,
    name = name
  )

  do.call(tfa$seq2seq$hardmax, args)

}


#' @title Inference Sampler
#'
#' @details A helper to use during inference with a custom sampling function.
#'
#' @param sample_fn A callable that takes outputs and emits tensor sample_ids.
#' @param sample_shape Either a list of integers, or a 1-D Tensor of type int32,
#' the shape of the each sample in the batch returned by sample_fn.
#' @param sample_dtype the dtype of the sample returned by sample_fn.
#' @param end_fn A callable that takes sample_ids and emits a bool vector shaped
#' [batch_size] indicating whether each sample is an end token.
#' @param next_inputs_fn (Optional) A callable that takes sample_ids and returns
#' the next batch of inputs. If not provided, sample_ids is used as the next batch of inputs.
#' @param ... A list that contains other common arguments for layer creation.
#'
#' @return None
#' @export
sampler_inference <- function(sample_fn,
                                   sample_shape,
                                   sample_dtype = tf$int32,
                                   end_fn,
                                   next_inputs_fn = NULL,
                                   ...) {
  args = list(
    sample_fn = sample_fn,
    sample_shape = sample_shape,
    sample_dtype = sample_dtype,
    end_fn = end_fn,
    next_inputs_fn = next_inputs_fn,
    ...
  )

  do.call(tfa$seq2seq$InferenceSampler, args)

}


#' @title Implements Luong-style (multiplicative) attention scoring.
#'
#' @details This attention has two forms. The first is standard Luong attention,
#' as described in:
#' Minh-Thang Luong, Hieu Pham, Christopher D. Manning. Effective Approaches to
#' Attention-based Neural Machine Translation. EMNLP 2015.
#' The second is the scaled form inspired partly by the normalized form of Bahdanau
#' attention.
#' To enable the second form, construct the object with parameter `scale=TRUE`.
#'
#' @param object Model or layer object
#' @param units The depth of the attention mechanism.
#' @param memory The memory to query; usually the output of an RNN encoder. This tensor should be shaped [batch_size, max_time, ...].
#' @param memory_sequence_length (optional): Sequence lengths for the batch entries in memory. If provided, the memory tensor rows are masked with zeros for values past the respective sequence lengths.
#' @param scale boolean. Whether to scale the energy term.
#' @param probability_fn (optional) string, the name of function to convert the attention score to probabilities. The default is softmax which is tf.nn.softmax. Other options is hardmax, which is hardmax() within this module. Any other value will result intovalidation error. Default to use softmax.
#' @param dtype The data type for the memory layer of the attention mechanism.
#' @param name Name to use when creating ops.
#' @param ... A list that contains other common arguments for layer creation.
#'
#'
#' @importFrom keras create_layer
#' @return None
#' @export
attention_luong <- function(object,
                            units,
                            memory = NULL,
                            memory_sequence_length = NULL,
                            scale = FALSE,
                            probability_fn = 'softmax',
                            dtype = NULL,
                            name = 'LuongAttention',
                            ...) {

  args = list(
    units = as.integer(units),
    memory = memory,
    memory_sequence_length = memory_sequence_length,
    scale = scale,
    probability_fn = probability_fn,
    dtype = dtype,
    name =  name,
    ...
  )

  if(!is.null(memory_sequence_length))
    args$memory_sequence_length <- as.integer(args$memory_sequence_length)

  create_layer(tfa$seq2seq$LuongAttention, object, args)
}


#' @title Monotonic attention mechanism with Luong-style energy function.
#'
#' @details This type of attention enforces a monotonic constraint on the attention
#' distributions; that is once the model attends to a given point in the memory it
#' can't attend to any prior points at subsequence output timesteps. It achieves
#' this by using the _monotonic_probability_fn instead of softmax to construct its
#' attention distributions. Otherwise, it is equivalent to LuongAttention.
#' This approach is proposed in
#' [Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck, "Online and
#' Linear-Time Attention by Enforcing Monotonic Alignments." ICML 2017.](https://arxiv.org/abs/1704.00784)
#'
#'
#'
#' @param object Model or layer object
#' @param units The depth of the query mechanism.
#' @param memory The memory to query; usually the output of an RNN encoder. This
#' tensor should be shaped [batch_size, max_time, ...].
#' @param memory_sequence_length (optional): Sequence lengths for the batch entries
#' in memory. If provided, the memory tensor rows are masked with zeros for values
#' past the respective sequence lengths.
#' @param scale boolean. Whether to scale the energy term.
#' @param sigmoid_noise Standard deviation of pre-sigmoid noise. See the docstring
#' for `_monotonic_probability_fn` for more information.
#' @param sigmoid_noise_seed (optional) Random seed for pre-sigmoid noise.
#' @param score_bias_init Initial value for score bias scalar. It's recommended to
#' initialize this to a negative value when the length of the memory is large.
#' @param mode How to compute the attention distribution. Must be one of 'recursive',
#' 'parallel', or 'hard'. See the docstring for tfa.seq2seq.monotonic_attention for
#' more information.
#' @param dtype The data type for the query and memory layers of the attention mechanism.
#' @param name Name to use when creating ops.
#' @param ... A list that contains other common arguments for layer creation.
#'
#'
#' @importFrom keras create_layer
#' @return None
#' @export
attention_luong_monotonic <- function(object,
                                      units,
                                      memory = NULL,
                                      memory_sequence_length = NULL,
                                      scale = FALSE,
                                      sigmoid_noise = 0.0,
                                      sigmoid_noise_seed = NULL,
                                      score_bias_init = 0.0,
                                      mode = 'parallel',
                                      dtype = NULL,
                                      name = 'LuongMonotonicAttention',
                                      ...) {

  args = list(
    units  = as.integer(units),
    memory = memory,
    memory_sequence_length = memory_sequence_length,
    scale = scale,
    sigmoid_noise = sigmoid_noise,
    sigmoid_noise_seed = sigmoid_noise_seed,
    score_bias_init = score_bias_init,
    mode = mode,
    dtype = dtype,
    name = name,
    ...
  )

  if(!is.null(memory_sequence_length))
    args$memory_sequence_length <- as.integer(args$memory_sequence_length)
  if(!is.null(sigmoid_noise_seed))
    args$sigmoid_noise_seed <- as.integer(args$sigmoid_noise_seed)

  create_layer(tfa$seq2seq$LuongMonotonicAttention, object, args)
}

#' @title Monotonic attention
#'
#' @description Compute monotonic attention distribution from choosing probabilities.
#'
#' @details Monotonic attention implies that the input sequence is processed in an
#' explicitly left-to-right manner when generating the output sequence. In
#' addition, once an input sequence element is attended to at a given output
#' timestep, elements occurring before it cannot be attended to at subsequent
#' output timesteps. This function generates attention distributions
#' according to these assumptions. For more information, see `Online and
#' Linear-Time Attention by Enforcing Monotonic Alignments`.
#'
#' @param p_choose_i Probability of choosing input sequence/memory element i.
#' Should be of shape (batch_size, input_sequence_length), and should all be
#' in the range [0, 1].
#' @param previous_attention The attention distribution from the previous output
#' timestep. Should be of shape (batch_size, input_sequence_length). For the first
#' output timestep, preevious_attention[n] should be [1, 0, 0, ..., 0] for all n
#' in [0, ... batch_size - 1].
#' @param mode How to compute the attention distribution. Must be one of 'recursive',
#' 'parallel', or 'hard'.  'recursive' uses tf$scan to recursively compute the
#' distribution. This is slowest but is exact, general, and does not suffer from
#' numerical instabilities.  'parallel' uses parallelized cumulative-sum and
#' cumulative-product operations to compute a closed-form solution to the recurrence
#' relation defining the attention distribution. This makes it more efficient than
#' 'recursive', but it requires numerical checks which make the distribution non-exact.
#' This can be a problem in particular when input_sequence_length is long and/or p_choose_i
#' has entries very close to 0 or 1. * 'hard' requires that the probabilities in p_choose_i
#' are all either 0 or 1, and subsequently uses a more efficient and exact solution.
#'
#' @return A tensor of shape (batch_size, input_sequence_length) representing
#' the attention distributions for each sequence in the batch.
#'
#' @section Raises:
#' ValueError: mode is not one of 'recursive', 'parallel', 'hard'.
#'
#' @export
attention_monotonic <- function(p_choose_i, previous_attention, mode) {

  args<- list(
    p_choose_i = p_choose_i,
    previous_attention = previous_attention,
    mode = mode
  )

  do.call(tfa$seq2seq$monotonic_attention, args)

}


#' @title Safe cumprod
#'
#' @description Computes cumprod of x in logspace using cumsum to avoid underflow.
#'
#' @details The cumprod function and its gradient can result in numerical instabilities
#' when its argument has very small and/or zero values. As long as the
#' argument is all positive, we can instead compute the cumulative product as
#' exp(cumsum(log(x))). This function can be called identically to
#' tf$cumprod.
#'
#' @param x Tensor to take the cumulative product of.
#' @param ... Passed on to cumsum; these are identical to those in cumprod
#' @return Cumulative product of x.
#'
#' @export
safe_cumprod <- function(x, ...) {

  args <- list(
    x = x,
    ...
  )

  do.call(tfa$seq2seq$safe_cumprod, args)

}


#' @title Sample Embedding Sampler
#'
#' @description  A sampler for use during inference.
#' @details Uses sampling (from a distribution) instead of argmax and passes
#' the result through an embedding layer to get the next input.
#'
#' @param embedding_fn (Optional) A callable that takes a vector tensor of ids (argmax ids),
#' or the params argument for embedding_lookup. The returned tensor will be passed to the
#' decoder input.
#' @param softmax_temperature (Optional) float32 scalar, value to divide the logits by
#' before computing the softmax. Larger values (above 1.0) result in more random samples,
#' while smaller values push the sampling distribution towards the argmax. Must be strictly
#' greater than 0. Defaults to 1.0.
#' @param seed (Optional) The sampling seed.
#'
#'
#'
#'
#'
#' @return None
#' @export
sampler_sample_embedding <- function(embedding_fn = NULL,
                                           softmax_temperature = NULL,
                                           seed = NULL) {

  args = list(embedding_fn = embedding_fn,
              softmax_temperature = softmax_temperature,
              seed = seed)

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  do.call(tfa$seq2seq$SampleEmbeddingSampler,args)

}


#' @title Sampler
#'
#' @description Interface for implementing sampling in seq2seq decoders.
#'
#' @param ... parametr to pass batch_size, initialize, next_inputs, sample, sample_ids_dtype, sample_ids_shape
#'
#' @return None
#' @export
sampler <- function(...) {
  args = list(...)

  do.call(tfa$seq2seq$Sampler, args)
}




#' @title A training sampler that adds scheduled sampling
#'
#' @param sampling_probability A float32 0-D or 1-D tensor: the probability of sampling
#' categorically from the output ids instead of reading directly from the inputs.
#' @param embedding_fn A callable that takes a vector tensor of ids (argmax ids), or the
#' params argument for embedding_lookup.
#' @param time_major bool. Whether the tensors in inputs are time major. If `FALSE`
#' (default), they are assumed to be batch major.
#' @param seed The sampling seed.
#' @param scheduling_seed The schedule decision rule sampling seed.
#'
#'
#'
#'
#'
#'
#'
#'
#'
#' @return Returns -1s for sample_ids where no sampling took place; valid sample id values elsewhere.
#' @export
sampler_scheduled_embedding_training <- function(sampling_probability,
                                                 embedding_fn = NULL,
                                                 time_major = FALSE,
                                                 seed  = NULL,
                                                 scheduling_seed  = NULL) {


  args = list(
    sampling_probability = sampling_probability,
    embedding_fn = embedding_fn,
    time_major = time_major,
    seed  = seed,
    scheduling_seed  = scheduling_seed
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)
  if(!is.null(scheduling_seed))
    args$scheduling_seed <- as.integer(args$scheduling_seed)

  do.call(tfa$seq2seq$ScheduledEmbeddingTrainingSampler, args)
}



#' @title Scheduled Output Training Sampler
#' @description A training sampler that adds scheduled sampling directly to outputs.
#'
#'
#' @param sampling_probability A float32 scalar tensor: the probability of sampling
#' from the outputs instead of reading directly from the inputs.
#' @param time_major bool. Whether the tensors in inputs are time major. If False (default),
#' they are assumed to be batch major.
#' @param seed The sampling seed.
#' @param next_inputs_fn (Optional) callable to apply to the RNN outputs to create the next
#' input when sampling. If None (default), the RNN outputs will be used as the next inputs.
#'
#'
#'
#'
#' @return FALSE for sample_ids where no sampling took place; TRUE elsewhere.
#' @export
sampler_scheduled_output_training <- function(sampling_probability,
                                              time_major = FALSE,
                                              seed = NULL,
                                              next_inputs_fn = NULL) {

  args = list(sampling_probability = sampling_probability,
              time_major = time_major,
              seed = seed,
              next_inputs_fn = next_inputs_fn)

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  do.call(tfa$seq2seq$ScheduledOutputTrainingSampler, args)
}

#' @title Weighted cross-entropy loss for a sequence of logits.
#'
#' @param ... A list of parameters
#'
#' @return None
#'
#' @export
loss_sequence <- function(...) {
  args = list(...)

  do.call(tfa$seq2seq$SequenceLoss, args)
}


#' @title Tile batch
#'
#' @description Tile the batch dimension of a (possibly nested structure of) tensor(s)
#'
#' @details t. For each tensor t in a (possibly nested structure) of tensors,
#' this function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
#' of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a
#' shape `[batch_size * multiplier, s0, s1, ...]` composed of minibatch
#' entries `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is
#' repeated `multiplier` times.
#'
#' @param t `Tensor` shaped `[batch_size, ...]`.
#' @param multiplier Python int.
#' @param name Name scope for any created operations.
#'
#' @return A (possibly nested structure of) `Tensor` shaped `[batch_size * multiplier, ...]`.
#'
#' @section Raises:
#' ValueError: if tensor(s) `t` do not have a statically known rank or the rank is < 1.
#'
#' @export
tile_batch <- function(t, multiplier, name = NULL) {

  args <- list(
    t = t,
    multiplier = as.integer(multiplier),
    name = name
  )

  do.call(tfa$seq2seq$tile_batch, args)

}


#' @title A Sampler for use during training.
#'
#' @description Only reads inputs.
#'
#' @param time_major bool. Whether the tensors in inputs are time major.
#' If `FALSE` (default), they are assumed to be batch major.
#'
#'
#' @return None
#' @export
sampler_training <- function(time_major = FALSE) {
  args = list(time_major = time_major)

  do.call(tfa$seq2seq$TrainingSampler, args)
}







