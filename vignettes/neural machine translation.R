
library(tensorflow)
library(keras)
library(data.table)
library(tfdatasets)
library(tfaddons)

# Preprocessing -----------------------------------------------------------

# Assumes you've downloaded and unzipped one of the bilingual datasets offered at
# http://www.manythings.org/anki/ and put it into a directory "data"
# This example translates English to Dutch.
download_data = function(){
  if(!dir.exists('data')) {
    dir.create('data')
  }
  if(!file.exists('data/nld-eng.zip')) {
    download.file('http://www.manythings.org/anki/nld-eng.zip',
                  destfile = file.path("data", basename('nld-eng.zip')))
    unzip('data/nld-eng.zip', exdir = 'data')
  }
}

download_data()

filepath <- file.path("data", "nld.txt")

df = data.table::fread(filepath, header = FALSE, encoding = 'UTF-8',
                       select = c(1,2), nrows = -1)

text_cleaner <- function(text){
  text = text %>%
    # replace non ascii
    textclean::replace_non_ascii() %>%
    # remove all non relevant symbols (letters, spaces, and apostrophes are retained)
    textclean::strip(apostrophe.remove = TRUE) %>%
    paste('<start> ', ., ' <end>')
}

df = sapply(1:2, function(x) text_cleaner(df[[x]])) %>% as.data.table()

text_tok <- function(text) {
  tokenizer = text_tokenizer(filters='')
  tokenizer %>% fit_text_tokenizer(text)
  vocab_size = tokenizer$word_index
  data = tokenizer %>%
    texts_to_sequences(text) %>%
    pad_sequences(padding='post')
  list(vocab_size,data,tokenizer)
}

c(input_vocab_size, data_en, tokenizer_en) %<-% c(df[['V1']] %>% text_tok())

c(output_vocab_size, data_de, tokenizer_de) %<-% c(df[['V2']] %>% text_tok())


# Split the dataset
indices_to_take = sample.int(n = nrow(df), size = floor(0.8*nrow(df)), replace = FALSE)

split_data <- function(data) {
  c(train, test) %<-% list(data[indices_to_take, ], data[-indices_to_take, ] )
  list(train, test)
}


c(en_train, en_test, de_train, de_test) %<-% c(split_data(data_en), split_data(data_de))

rm(df, filepath, indices_to_take, download_data, split_data, text_cleaner, text_tok)

batch_size = 64L
buffer_size = nrow(en_train)
steps_per_epoch = buffer_size  %/% batch_size
embedding_dims = 256L
rnn_units = 1024L
dense_units = 1024L
dtype = tf$float32   #used to initialize DecoderCell Zero state


dataset = tensor_slices_dataset(list(en_train, de_train)) %>%
  dataset_shuffle(buffer_size) %>% dataset_batch(batch_size, drop_remainder = TRUE)


EncoderNetwork = reticulate::PyClass(
  'EncoderNetwork',
  inherit = tf$keras$Model,
  defs = list(

    `__init__` = function(self, input_vocab_size, embedding_dims, rnn_units) {

      super()$`__init__`()

      self$encoder_embedding = layer_embedding(input_dim = length(input_vocab_size),
                                               output_dim = embedding_dims)
      self$encoder_rnnlayer = layer_lstm(units = rnn_units, return_sequences = TRUE,
                                         return_state = TRUE)
      NULL
    }
  )
)



DecoderNetwork = reticulate::PyClass(
  'DecoderNetwork',
  inherit = tf$keras$Model,
  defs = list(

    `__init__` = function(self, output_vocab_size, embedding_dims, rnn_units) {

      super()$`__init__`()
      self$decoder_embedding = layer_embedding(input_dim = length(output_vocab_size),
                                               output_dim = embedding_dims)
      self$dense_layer = layer_dense(units = length(output_vocab_size))
      self$decoder_rnncell = tf$keras$layers$LSTMCell(rnn_units)
      # Sampler
      self$sampler = sampler_training()
      # Create attention mechanism with memory = NULL
      self$attention_mechanism = self$build_attention_mechanism(dense_units, NULL, c(rep(ncol(data_en), batch_size)))
      self$rnn_cell =  self$build_rnn_cell(batch_size)
      self$decoder = decoder_basic(cell=self$rnn_cell, sampler = self$sampler,
                                   output_layer = self$dense_layer)
      NULL
    },



    build_attention_mechanism = function(self, units, memory, memory_sequence_length) {
      attention_luong(units = units , memory = memory,
                      memory_sequence_length = memory_sequence_length)
    },

    build_rnn_cell = function(self, batch_size) {
      rnn_cell = attention_wrapper(cell = self$decoder_rnncell,
                                   attention_mechanism = self$attention_mechanism,
                                   attention_layer_size = dense_units)
      rnn_cell
    },

    build_decoder_initial_state = function(self, batch_size, encoder_state, dtype) {
      decoder_initial_state = self$rnn_cell$get_initial_state(batch_size = batch_size,
                                                              dtype = dtype)
      decoder_initial_state = decoder_initial_state$clone(cell_state = encoder_state)
      decoder_initial_state
    }
  )
)

encoderNetwork = EncoderNetwork(input_vocab_size, embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size, embedding_dims, rnn_units)
optimizer = tf$keras$optimizers$Adam()



loss_function <- function(y_pred, y) {
  #shape of y [batch_size, ty]
  #shape of y_pred [batch_size, Ty, output_vocab_size]
  loss = keras::loss_sparse_categorical_crossentropy(y, y_pred)
  mask = tf$logical_not(tf$math$equal(y,0L))   #output 0 for y=0 else output 1
  mask = tf$cast(mask, dtype=loss$dtype)
  loss = mask * loss
  loss = tf$reduce_mean(loss)
  loss
}

train_step <- function(input_batch, output_batch,encoder_initial_cell_state) {
  loss = 0L

  with(tf$GradientTape() %as% tape, {
    encoder_emb_inp = encoderNetwork$encoder_embedding(input_batch)
    c(a, a_tx, c_tx) %<-% encoderNetwork$encoder_rnnlayer(encoder_emb_inp,
                                                          initial_state = encoder_initial_cell_state)

    #[last step activations,last memory_state] of encoder passed as input to decoder Network
    # Prepare correct Decoder input & output sequence data
    decoder_input = tf$convert_to_tensor(output_batch %>% as.array() %>% .[,1:45]) # ignore <end>
    #compare logits with timestepped +1 version of decoder_input
    decoder_output = tf$convert_to_tensor(output_batch %>% as.array() %>% .[,2:46]) #ignore <start>

    # Decoder Embeddings
    decoder_emb_inp = decoderNetwork$decoder_embedding(decoder_input)

    #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
    decoderNetwork$attention_mechanism$setup_memory(a)
    decoder_initial_state = decoderNetwork$build_decoder_initial_state(batch_size,
                                                                       encoder_state = list(a_tx, c_tx),
                                                                       dtype = tf$float32)
    #BasicDecoderOutput
    c(outputs, res1, res2) %<-% decoderNetwork$decoder(decoder_emb_inp,initial_state = decoder_initial_state,
                                                       sequence_length = c(rep(ncol(data_en) - 1L, batch_size)))

    logits = outputs$rnn_output
    #Calculate loss

    loss = loss_function(logits, decoder_output)

  })
  #Returns the list of all layer variables / weights.
  #variables = encoderNetwork$trainable_variables + decoderNetwork$trainable_variables
  #variables = tf$math$add(encoderNetwork$trainable_variables, decoderNetwork$trainable_variables)
  variables = c(encoderNetwork$trainable_variables, decoderNetwork$trainable_variables)
  # differentiate loss wrt variables
  gradients = tape$gradient(loss, variables)
  #grads_and_vars – List of(gradient, variable) pairs.
  grads_and_vars = purrr::transpose(list(gradients,variables))
  optimizer$apply_gradients(grads_and_vars)
  loss
}

initialize_initial_state = function() {
  list(tf$zeros(c(batch_size, rnn_units)), tf$zeros(c(batch_size, rnn_units)))
}


epochs = 1


for (i in 1:sum(epochs + 1)) {
  encoder_initial_cell_state = initialize_initial_state()
  total_loss = 0.0
  res = dataset %>% dataset_take(steps_per_epoch) %>% iterate()
  for (batch in 1:length(res)) {
    c(input_batch, output_batch) %<-% res[[batch]]
    batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
    total_loss = total_loss + batch_loss
    if((batch+1) %% 5 == 0) {
      print(paste('total loss:', batch_loss$numpy(), 'epoch', i, 'batch',batch+1))
    }
  }

}
