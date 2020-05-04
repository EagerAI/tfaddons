#' @title CRF binary score
#'
#' @description Computes the binary scores of tag sequences.
#'
#'
#' @param tag_indices A [batch_size, max_seq_len] matrix of tag indices.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#' @param transition_params A [num_tags, num_tags] matrix of binary potentials.
#'
#' @return binary_scores: A [batch_size] vector of binary scores.
#'
#' @export
crf_binary_score <- function(tag_indices,
                             sequence_lengths,
                             transition_params) {

  args <- list(
    tag_indices = tag_indices,
    sequence_lengths = as.integer(sequence_lengths),
    transition_params = transition_params
  )

  do.call(tfa$text$crf_binary_score, args)

}


#' @title CRF decode
#'
#' @description Decode the highest scoring sequence of tags.
#'
#'
#' @param potentials A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
#' @param transition_params A [num_tags, num_tags] matrix of binary potentials.
#' @param sequence_length A [batch_size] vector of true sequence lengths.
#'
#' @return decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
#' Contains the highest scoring tag indices. best_score: A [batch_size] vector,
#' containing the score of `decode_tags`.
#'
#' @export
crf_decode <- function(potentials, transition_params, sequence_length) {

  args <- list(
    potentials = potentials,
    transition_params = transition_params,
    sequence_length = as.integer(sequence_length)
  )

  do.call(tfa$text$crf_decode, args)

}


#' @title CRF decode backward
#'
#' @description Computes backward decoding in a linear-chain CRF.
#'
#'
#' @param inputs A [batch_size, num_tags] matrix of backpointer of next step (in time order).
#' @param state A [batch_size, 1] matrix of tag index of next step.
#'
#' @return new_tags: A [batch_size, num_tags] tensor containing the new tag indices.
#'
#' @export
crf_decode_backward <- function(inputs, state) {

  args <- list(
    inputs = inputs,
    state = state
  )

  do.call(tfa$text$crf_decode_backward, args)

}


#' @title CRF decode forward
#'
#' @description Computes forward decoding in a linear-chain CRF.
#'
#'
#' @param inputs A [batch_size, num_tags] matrix of unary potentials.
#' @param state A [batch_size, num_tags] matrix containing the previous step's score values.
#' @param transition_params A [num_tags, num_tags] matrix of binary potentials.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#'
#' @return backpointers: A [batch_size, num_tags] matrix of backpointers.
#' new_state: A [batch_size, num_tags] matrix of new score values.
#'
#' @export
crf_decode_forward <- function(inputs,
                               state,
                               transition_params,
                               sequence_lengths) {

  python_function_result <- tfa$text$crf_decode_forward(
    inputs = inputs,
    state = state,
    transition_params = transition_params,
    sequence_lengths = as.integer(sequence_lengths)
  )

}


#' @title CRF forward
#'
#' @description Computes the alpha values in a linear-chain CRF.
#'
#' @details See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
#'
#' @param inputs A [batch_size, num_tags] matrix of unary potentials.
#' @param state A [batch_size, num_tags] matrix containing the previous alpha values.
#' @param transition_params A [num_tags, num_tags] matrix of binary potentials. This
#' matrix is expanded into a [1, num_tags, num_tags] in preparation for the broadcast
#' summation occurring within the cell.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#'
#' @return new_alphas: A [batch_size, num_tags] matrix containing the new alpha values.
#'
#' @export
crf_forward <- function(inputs, state,
                        transition_params,
                        sequence_lengths) {

  args <- list(
    inputs = inputs,
    state = state,
    transition_params = transition_params,
    sequence_lengths = as.integer(sequence_lengths)
  )

  do.call(tfa$text$crf_forward, args)

}


#' @title CRF log likelihood
#'
#' @description Computes the log-likelihood of tag sequences in a CRF.
#'
#' @param inputs A [batch_size, max_seq_len, num_tags] tensor of unary potentials to use
#' as input to the CRF layer.
#' @param tag_indices A [batch_size, max_seq_len] matrix of tag indices for which we
#' compute the log-likelihood.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#' @param transition_params A [num_tags, num_tags] transition matrix, if available.
#'
#' @return log_likelihood: A [batch_size] Tensor containing the log-likelihood of each example,
#' given the sequence of tag indices. transition_params: A [num_tags, num_tags] transition matrix.
#' This is either provided by the caller or created in this function.
#'
#' @export
crf_log_likelihood <- function(inputs,
                               tag_indices,
                               sequence_lengths,
                               transition_params = NULL) {

  args <- list(
    inputs = inputs,
    tag_indices = tag_indices,
    sequence_lengths = as.integer(sequence_lengths),
    transition_params = transition_params
  )

  do.call(tfa$text$crf_log_likelihood, args)

}



#' @title CRF log norm
#'
#' @description Computes the normalization for a CRF.
#'
#'
#' @param inputs A [batch_size, max_seq_len, num_tags] tensor of unary potentials
#' to use as input to the CRF layer.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#' @param transition_params A [num_tags, num_tags] transition matrix.
#'
#' @return log_norm: A [batch_size] vector of normalizers for a CRF.
#'
#' @export
crf_log_norm <- function(inputs,
                         sequence_lengths,
                         transition_params) {

  args <- list(
    inputs = inputs,
    sequence_lengths = as.integer(sequence_lengths),
    transition_params = transition_params
  )

  do.call(tfa$text$crf_log_norm, args)

}



#' @title CRF multitag sequence score
#'
#' @description Computes the unnormalized score of all tag sequences matching
#'
#' @details tag_bitmap. tag_bitmap enables more than one tag to be considered
#' correct at each time
#' step. This is useful when an observed output at a given time step is
#' consistent with more than one tag, and thus the log likelihood of that
#' observation must take into account all possible consistent tags. Using
#' one-hot vectors in tag_bitmap gives results identical to
#' crf_sequence_score.
#'
#' @param inputs A [batch_size, max_seq_len, num_tags] tensor of unary potentials
#' to use as input to the CRF layer.
#' @param tag_bitmap A [batch_size, max_seq_len, num_tags] boolean tensor representing
#' all active tags at each index for which to calculate the unnormalized score.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#' @param transition_params A [num_tags, num_tags] transition matrix.
#'
#' @return sequence_scores: A [batch_size] vector of unnormalized sequence scores.
#'
#' @export
crf_multitag_sequence_score <- function(inputs, tag_bitmap, sequence_lengths, transition_params) {

  args <- list(
    inputs = inputs,
    tag_bitmap = tag_bitmap,
    sequence_lengths = as.integer(sequence_lengths),
    transition_params = transition_params
  )

  do.call(tfa$text$crf_multitag_sequence_score, args)

}


#' @title CRF sequence score
#'
#' @description Computes the unnormalized score for a tag sequence.
#'
#'
#' @param inputs A [batch_size, max_seq_len, num_tags] tensor of unary potentials
#' to use as input to the CRF layer.
#' @param tag_indices A [batch_size, max_seq_len] matrix of tag indices for which
#' we compute the unnormalized score.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#' @param transition_params A [num_tags, num_tags] transition matrix. Returns:
#'
#' @return sequence_scores: A [batch_size] vector of unnormalized sequence scores.
#'
#' @export
crf_sequence_score <- function(inputs,
                               tag_indices,
                               sequence_lengths,
                               transition_params) {

  args <- list(
    inputs = inputs,
    tag_indices = tag_indices,
    sequence_lengths = as.integer(sequence_lengths),
    transition_params = transition_params
  )

  do.call(tfa$text$crf_sequence_score, args)

}




#' @title CRF unary score
#'
#' @description Computes the unary scores of tag sequences.
#'
#'
#' @param tag_indices A [batch_size, max_seq_len] matrix of tag indices.
#' @param sequence_lengths A [batch_size] vector of true sequence lengths.
#' @param inputs A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
#'
#' @return unary_scores: A [batch_size] vector of unary scores.
#'
#' @export
crf_unary_score <- function(tag_indices,
                            sequence_lengths,
                            inputs) {

  args <- list(
    tag_indices = tag_indices,
    sequence_lengths = as.integer(sequence_lengths),
    inputs = inputs
  )

  do.call(tfa$text$crf_unary_score, args)

}




#' @title Parse time
#'
#' @description Parse an input string according to the provided format string into a
#'
#' @details Unix time. Parse an input string according to the provided format string
#' into a Unix
#' time, the number of seconds / milliseconds / microseconds / nanoseconds
#' elapsed since January 1, 1970 UTC. Uses strftime()-like formatting options, with
#' the same extensions as
#' FormatTime(), but with the exceptions that %E#S is interpreted as %E*S, and
#' %E#f as %E*f. %Ez and %E*z also accept the same inputs. %Y consumes as many numeric
#' characters as it can, so the matching
#' data should always be terminated with a non-numeric. %E4Y always
#' consumes exactly four characters, including any sign. Unspecified fields are taken
#' from the default date and time of ... "1970-01-01 00:00:00.0 +0000" For example,
#' parsing a string of "15:45" (%H:%M) will return an
#' Unix time that represents "1970-01-01 15:45:00.0 +0000". Note that ParseTime only
#' heeds the fields year, month, day, hour,
#' minute, (fractional) second, and UTC offset. Other fields, like
#' weekday (%a or %A), while parsed for syntactic validity, are
#' ignored in the conversion. Date and time fields that are out-of-range will be treated as
#' errors rather than normalizing them like `absl::CivilSecond` does.
#' For example, it is an error to parse the date "Oct 32, 2013"
#' because 32 is out of range. A leap second of ":60" is normalized to ":00" of the following
#' minute with fractional seconds discarded. The following table
#' shows how the given seconds and subseconds will be parsed: "59.x" -> 59.x // exact "60.x" -> 00.0 // normalized "00.x" -> 00.x // exact
#'
#' @param time_string The input time string to be parsed.
#' @param time_format The time format.
#' @param output_unit The output unit of the parsed unix time. Can only be SECOND, MILLISECOND, MICROSECOND, NANOSECOND.
#'
#' @return the number of seconds / milliseconds / microseconds / nanoseconds elapsed since January 1, 1970 UTC.
#'
#' @section Raises:
#' ValueError: If `output_unit` is not a valid value, if parsing `time_string` according to `time_format` failed.
#'
#' @export
parse_time <- function(time_string, time_format, output_unit) {

  args <- list(
    time_string = time_string,
    time_format = time_format,
    output_unit = output_unit
  )

  do.call(tfa$text$parse_time, args)

}




#' @title Skip gram sample
#'
#' @description Generates skip-gram token and label paired Tensors from the input
#'
#' @details tensor. Generates skip-gram `("token", "label")` pairs using each
#' element in the
#' rank-1 `input_tensor` as a token. The window size used for each token will
#' be randomly selected from the range specified by `[min_skips, max_skips]`,
#' inclusive. See https://arxiv.org/abs/1301.3781 for more details about
#' skip-gram. For example, given `input_tensor = ["the", "quick", "brown", "fox",
#' "jumps"]`, `min_skips = 1`, `max_skips = 2`, `emit_self_as_target = FALSE`,
#' the output `(tokens, labels)` pairs for the token "quick" will be randomly
#' selected from either `(tokens=["quick", "quick"], labels=["the", "brown"])`
#' for 1 skip, or `(tokens=["quick", "quick", "quick"],
#' labels=["the", "brown", "fox"])` for 2 skips. If `emit_self_as_target = TRUE`,
#' each token will also be emitted as a label
#' for itself. From the previous example, the output will be either
#' `(tokens=["quick", "quick", "quick"], labels=["the", "quick", "brown"])`
#' for 1 skip, or `(tokens=["quick", "quick", "quick", "quick"],
#' labels=["the", "quick", "brown", "fox"])` for 2 skips.
#' The same process is repeated for each element of `input_tensor` and
#' concatenated together into the two output rank-1 `Tensors` (one for all the
#' tokens, another for all the labels). If `vocab_freq_table` is specified,
#' tokens in `input_tensor` that are not
#' present in the vocabulary are discarded. Tokens whose frequency counts are
#' below `vocab_min_count` are also discarded. Tokens whose frequency
#' proportions in the corpus exceed `vocab_subsampling` may be randomly
#' down-sampled. See Eq. 5 in http://arxiv.org/abs/1310.4546 for more details
#' about subsampling. Due to the random window sizes used for each token, the lengths of the
#' outputs are non-deterministic, unless `batch_size` is specified to batch
#' the outputs to always return `Tensors` of length `batch_size`.
#'
#' @param input_tensor A rank-1 `Tensor` from which to generate skip-gram candidates.
#' @param min_skips `int` or scalar `Tensor` specifying the minimum window size to
#' randomly use for each token. Must be >= 0 and <= `max_skips`. If `min_skips` and
#' `max_skips` are both 0, the only label outputted will be the token itself when
#' `emit_self_as_target = TRUE` - or no output otherwise.
#' @param max_skips `int` or scalar `Tensor` specifying the maximum window size to
#' randomly use for each token. Must be >= 0.
#' @param start `int` or scalar `Tensor` specifying the position in `input_tensor`
#' from which to start generating skip-gram candidates.
#' @param limit `int` or scalar `Tensor` specifying the maximum number of elements
#' in `input_tensor` to use in generating skip-gram candidates. -1 means to use the
#' rest of the `Tensor` after `start`.
#' @param emit_self_as_target `bool` or scalar `Tensor` specifying whether to emit
#' each token as a label for itself.
#' @param vocab_freq_table (Optional) A lookup table (subclass of
#' `lookup.InitializableLookupTableBase`) that maps tokens to their raw frequency counts.
#' If specified, any token in `input_tensor` that is not found in `vocab_freq_table` will
#' be filtered out before generating skip-gram candidates. While this will typically map
#' to integer raw frequency counts, it could also map to float frequency proportions.
#' `vocab_min_count` and `corpus_size` should be in the same units as this.
#' @param vocab_min_count (Optional) `int`, `float`, or scalar `Tensor` specifying minimum
#' frequency threshold (from `vocab_freq_table`) for a token to be kept in `input_tensor`.
#' If this is specified, `vocab_freq_table` must also be specified - and they should both
#' be in the same units.
#' @param vocab_subsampling (Optional) `float` specifying frequency proportion threshold
#' for tokens from `input_tensor`. Tokens that occur more frequently (based on the ratio
#' of the token's `vocab_freq_table` value to the `corpus_size`) will be randomly down-sampled.
#' Reasonable starting values may be around 1e-3 or 1e-5. If this is specified, both
#' `vocab_freq_table` and `corpus_size` must also be specified.
#' See Eq. 5 in http://arxiv.org/abs/1310.4546 for more details.
#' @param corpus_size (Optional) `int`, `float`, or scalar `Tensor` specifying the total
#' number of tokens in the corpus (e.g., sum of all the frequency counts of `vocab_freq_table`).
#' Used with `vocab_subsampling` for down-sampling frequently occurring tokens. If this
#' is specified, `vocab_freq_table` and `vocab_subsampling` must also be specified.
#' @param batch_size (Optional) `int` specifying batch size of returned `Tensors`.
#' @param batch_capacity (Optional) `int` specifying batch capacity for the queue used for
#' batching returned `Tensors`. Only has an effect if `batch_size` > 0.
#' Defaults to 100 * `batch_size` if not specified.
#' @param seed (Optional) `int` used to create a random seed for window size and subsampling.
#' See `set_random_seed` docs for behavior.
#' @param name (Optional) A `string` name or a name scope for the operations.
#'
#' @return A `list` containing (token, label) `Tensors`. Each output `Tensor` is of rank-1 and
#' has the same type as `input_tensor`. The `Tensors` will be of length `batch_size`;
#' if `batch_size` is not specified, they will be of random length, though they will be
#' in sync with each other as long as they are evaluated together.
#'
#' @section Raises:
#' ValueError: If `vocab_freq_table` is not provided, but `vocab_min_count`,
#' `vocab_subsampling`, or `corpus_size` is specified. If `vocab_subsampling` and
#' `corpus_size` are not both present or both absent.
#'
#' @export
skip_gram_sample <- function(input_tensor, min_skips = 1, max_skips = 5,
                             start = 0, limit = -1, emit_self_as_target = FALSE,
                             vocab_freq_table = NULL, vocab_min_count = NULL,
                             vocab_subsampling = NULL, corpus_size = NULL,
                             batch_size = NULL, batch_capacity = NULL,
                             seed = NULL, name = NULL) {

  args <- list(
    input_tensor = input_tensor,
    min_skips = as.integer(min_skips),
    max_skips = as.integer(max_skips),
    start = as.integer(start),
    limit = as.integer(limit),
    emit_self_as_target = emit_self_as_target,
    vocab_freq_table = vocab_freq_table,
    vocab_min_count = vocab_min_count,
    vocab_subsampling = vocab_subsampling,
    corpus_size = corpus_size,
    batch_size = batch_size,
    batch_capacity = batch_capacity,
    seed = seed,
    name = name
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  if(!is.null(batch_size))
    args$batch_size <- as.integer(args$batch_size)

  if(!is.null(batch_capacity))
    args$batch_capacity <- as.integer(args$batch_capacity)

  do.call(tfa$text$skip_gram_sample, args)

}



#' @title Skip gram sample with text vocab
#'
#' @description Skip-gram sampling with a text vocabulary file.
#'
#' @details Wrapper around `skip_gram_sample()` for use with a text vocabulary file.
#' The vocabulary file is expected to be a plain-text file, with lines of
#' `vocab_delimiter`-separated columns. The `vocab_token_index` column should
#' contain the vocabulary term, while the `vocab_freq_index` column should
#' contain the number of times that term occurs in the corpus. For example,
#' with a text vocabulary file of:
#' ``` bonjour,fr,42 hello,en,777 hola,es,99 ```
#' You should set `vocab_delimiter=","`, `vocab_token_index=0`, and
#' `vocab_freq_index=2`. See `skip_gram_sample()` documentation for more details
#' about the skip-gram
#' sampling process.
#'
#' @param input_tensor A rank-1 `Tensor` from which to generate skip-gram candidates.
#' @param vocab_freq_file `string` specifying full file path to the text vocab file.
#' @param vocab_token_index `int` specifying which column in the text vocab file contains the
#' tokens.
#' @param vocab_token_dtype `DType` specifying the format of the tokens in the text vocab file.
#' @param vocab_freq_index `int` specifying which column in the text vocab file contains the
#' frequency counts of the tokens.
#' @param vocab_freq_dtype `DType` specifying the format of the frequency counts in the text
#' vocab file.
#' @param vocab_delimiter `string` specifying the delimiter used in the text vocab file.
#' @param vocab_min_count `int`, `float`, or scalar `Tensor` specifying minimum frequency
#' threshold (from `vocab_freq_file`) for a token to be kept in `input_tensor`. This should
#' correspond with `vocab_freq_dtype`.
#' @param vocab_subsampling (Optional) `float` specifying frequency proportion threshold for
#' tokens from `input_tensor`. Tokens that occur more frequently will be randomly down-sampled.
#' Reasonable starting values may be around 1e-3 or 1e-5. See Eq. 5
#' in http://arxiv.org/abs/1310.4546 for more details.
#' @param corpus_size (Optional) `int`, `float`, or scalar `Tensor` specifying the total number
#' of tokens in the corpus (e.g., sum of all the frequency counts of `vocab_freq_file`). Used with
#' `vocab_subsampling` for down-sampling frequently occurring tokens. If this is specified,
#' `vocab_freq_file` and `vocab_subsampling` must also be specified. If `corpus_size` is needed
#' but not supplied, then it will be calculated from `vocab_freq_file`. You might want to supply
#' your own value if you have already eliminated infrequent tokens from your vocabulary files
#' (where frequency < vocab_min_count) to save memory in the internal token lookup table. Otherwise,
#' the unused tokens' variables will waste memory. The user-supplied `corpus_size` value must be
#' greater than or equal to the sum of all the frequency counts of `vocab_freq_file`.
#' @param min_skips `int` or scalar `Tensor` specifying the minimum window size to randomly use for
#' each token. Must be >= 0 and <= `max_skips`. If `min_skips` and `max_skips` are both 0, the only
#' label outputted will be the token itself.
#' @param max_skips `int` or scalar `Tensor` specifying the maximum window size to randomly use for
#' each token. Must be >= 0.
#' @param start `int` or scalar `Tensor` specifying the position in `input_tensor` from which to start
#' generating skip-gram candidates.
#' @param limit `int` or scalar `Tensor` specifying the maximum number of elements in `input_tensor`
#' to use in generating skip-gram candidates. -1 means to use the rest of the `Tensor` after `start`.
#' @param emit_self_as_target `bool` or scalar `Tensor` specifying whether to emit each token as a
#' label for itself.
#' @param batch_size (Optional) `int` specifying batch size of returned `Tensors`.
#' @param batch_capacity (Optional) `int` specifying batch capacity for the queue used for
#' batching returned `Tensors`. Only has an effect if `batch_size` > 0.
#' Defaults to 100 * `batch_size` if not specified.
#' @param seed (Optional) `int` used to create a random seed for window size and subsampling.
#' See [`set_random_seed`](../../g3doc/python/constant_op.md#set_random_seed) for behavior.
#' @param name (Optional) A `string` name or a name scope for the operations.
#'
#' @return A `list` containing (token, label) `Tensors`. Each output `Tensor` is of rank-1 and
#' has the same type as `input_tensor`. The `Tensors` will be of length `batch_size`;
#' if `batch_size` is not specified, they will be of random length, though they will be
#' in sync with each other as long as they are evaluated together.
#'
#' @section Raises:
#' ValueError: If `vocab_token_index` or `vocab_freq_index` is less than 0 or exceeds the
#' number of columns in `vocab_freq_file`. If `vocab_token_index` and `vocab_freq_index`
#' are both set to the same column. If any token in `vocab_freq_file` has a negative frequency.
#'
#' @export
skip_gram_sample_with_text_vocab <- function(input_tensor, vocab_freq_file,
                                             vocab_token_index = 0, vocab_token_dtype = tf$string,
                                             vocab_freq_index = 1, vocab_freq_dtype = tf$float64,
                                             vocab_delimiter = ",", vocab_min_count = NULL,
                                             vocab_subsampling = NULL, corpus_size = NULL,
                                             min_skips = 1, max_skips = 5, start = 0,
                                             limit = -1, emit_self_as_target = FALSE,
                                             batch_size = NULL, batch_capacity = NULL,
                                             seed = NULL, name = NULL) {

  args <- list(
    input_tensor = input_tensor,
    vocab_freq_file = vocab_freq_file,
    vocab_token_index = as.integer(vocab_token_index),
    vocab_token_dtype = vocab_token_dtype,
    vocab_freq_index = as.integer(vocab_freq_index),
    vocab_freq_dtype = vocab_freq_dtype,
    vocab_delimiter = vocab_delimiter,
    vocab_min_count = vocab_min_count,
    vocab_subsampling = vocab_subsampling,
    corpus_size = corpus_size,
    min_skips = as.integer(min_skips),
    max_skips = as.integer(max_skips),
    start = as.integer(start),
    limit = as.integer(limit),
    emit_self_as_target = emit_self_as_target,
    batch_size = batch_size,
    batch_capacity = batch_capacity,
    seed = seed,
    name = name
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  if(!is.null(batch_size))
    args$batch_size <- as.integer(args$batch_size)

  if(!is.null(batch_capacity))
    args$batch_capacity <- as.integer(args$batch_capacity)


  do.call(tfa$text$skip_gram_sample_with_text_vocab, args)

}


#' @title Viterbi decode
#'
#' @description Decode the highest scoring sequence of tags outside of TensorFlow.
#'
#' @details This should only be used at test time.
#'
#' @param score A [seq_len, num_tags] matrix of unary potentials.
#' @param transition_params A [num_tags, num_tags] matrix of binary potentials.
#'
#' @return viterbi: A [seq_len] list of integers containing the highest scoring tag indices.
#' viterbi_score: A float containing the score for the Viterbi sequence.
#'
#' @export
viterbi_decode <- function(score, transition_params) {

  args <- list(
    score = score,
    transition_params = transition_params
  )

  do.call(tfa$text$viterbi_decode, args)

}










