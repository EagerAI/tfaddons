context("callbacks")

source("utils.R")


test_succeeds("data is generated", {
  x_data <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
  y_data <-  ifelse(runif(50,0,1) > 0.6, 1L,0L) %>% as.matrix()

  x_data2 <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
  y_data2 <-  ifelse(runif(50,0,1) > 0.6, 1L,0L) %>% as.matrix()
})



test_succeeds("callback_time_stopping", {
  keras_model_sequential() %>%
    layer_dense(1,input_shape = 5) %>%
    compile(loss='mse',optimizer='adam',metrics='accuracy') %>%
    fit(x_data,y_data,
        epochs = 1,
        verbose=0,
        validation_data = list(x_data2,y_data2),
        callbacks = list(tfaddons::callback_time_stopping(seconds = 1)
                         ))
})


test_succeeds("callback_tqdm_progress_bar", {
  keras_model_sequential() %>%
    layer_dense(1,input_shape = 5) %>%
    compile(loss='mse',optimizer='adam',metrics='accuracy') %>%
    fit(x_data,y_data,
        epochs = 1,
        verbose=0,
        validation_data = list(x_data2,y_data2),
        callbacks = list(tfaddons::callback_tqdm_progress_bar()
        ))
})




