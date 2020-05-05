context("image_ops")

source("utils.R")

test_succeeds('img_read', {
  dir.create('data')
  download.file('https://tensorflow.org/images/tf_logo.png',file.path("data", basename('tf_logo.png')))

  img_path = paste( 'data' ,'tf_logo.png', sep = '/')

  img_path = gsub(img_path, replacement = '/',pattern = '\\',fixed=TRUE)

  img_raw = tf$io$read_file(img_path)
  img = tf$io$decode_png(img_raw)
  img = tf$image$convert_image_dtype(img, tf$float32)
  img = tf$image$resize(img, c(500L,500L))
  bw_img = 1.0 - tf$image$rgb_to_grayscale(img)
})


test_succeeds('img_mean_filter2d', {
  mean = img_mean_filter2d(img,filter_shape = 11)
})

test_succeeds('img_median_filter2d', {
  median = img_median_filter2d(img,filter_shape = 11)
})

test_succeeds('img_rotate', {
  rotate = img_rotate(img, tf$constant(pi/8))
})

test_succeeds('img_transform', {
  transform = img_transform(img, c(1.0, 1.0, -250, 0.0, 1.0, 0.0, 0.0, 0.0))
})


test_succeeds('img_random_hsv_in_yiq', {
  delta = 0.5
  lower_saturation = 0.1
  upper_saturation = 0.9
  lower_value = 0.2
  upper_value = 0.8
  rand_hsvinyiq = img_random_hsv_in_yiq(img, delta, lower_saturation, upper_saturation, lower_value, upper_value)
})


test_succeeds('img_adjust_hsv_in_yiq', {
  delta = 0.5
  saturation = 0.3
  value = 0.6
  adj_hsvinyiq = img_adjust_hsv_in_yiq(img, delta, saturation, value)
})



test_succeeds('img_dense_image_warp', {
  input_img = tf$expand_dims(img, 0L)

  if(as.integer(input_img$shape[[2]]) == as.integer(input_img$shape[[3]])) {
    flow_shape = list(1L, as.integer(input_img$shape[[2]]), as.integer(input_img$shape[[3]]), 2L)
    init_flows = tf$random$normal(flow_shape) * 2.0
    dense_img_warp = img_dense_image_warp(input_img, init_flows)
    dense_img_warp = tf$squeeze(dense_img_warp, 0)
  }

})



test_succeeds('img_euclidean_dist_transform', {
  gray = tf$image$convert_image_dtype(bw_img,tf$uint8)
  gray = tf$expand_dims(gray, 0L)
  eucid = img_euclidean_dist_transform(gray)
  eucid = tf$squeeze(eucid, c(0,-1))
})





