#' @title Adjust hsv in yiq
#'
#' @description Adjust hue, saturation, value of an RGB image in YIQ color space.
#'
#' @details This is a convenience method that converts an RGB image to float
#' representation, converts it to YIQ, rotates the color around the
#' Y channel by delta_hue in radians, scales the chrominance channels
#' (I, Q) by scale_saturation, scales all channels (Y, I, Q) by scale_value,
#' converts back to RGB, and then back to the original data type. `image` is an RGB image.
#' The image hue is adjusted by converting the
#' image to YIQ, rotating around the luminance channel (Y) by
#' `delta_hue` in radians, multiplying the chrominance channels (I, Q) by
#' `scale_saturation`, and multiplying all channels (Y, I, Q) by
#' `scale_value`. The image is then converted back to RGB.
#'
#' @param image RGB image or images. Size of the last dimension must be 3.
#' @param delta_hue float, the hue rotation amount, in radians.
#' @param scale_saturation float, factor to multiply the saturation by.
#' @param scale_value float, factor to multiply the value by.
#' @param name A name for this operation (optional).
#'
#' @return Adjusted image(s), same shape and dtype as `image`.
#'
#' @export
img_adjust_hsv_in_yiq <- function(image, delta_hue = 0, scale_saturation = 1, scale_value = 1, name = NULL) {

  args <- list(
    image = image,
    delta_hue = as.integer(delta_hue),
    scale_saturation = as.integer(scale_saturation),
    scale_value = as.integer(scale_value),
    name = name
  )

  do.call(tfa$image$adjust_hsv_in_yiq, args)

}



#' @title Blend
#'
#' @description Blend image1 and image2 using 'factor'.
#'
#' @details Factor can be above 0.0. A value of 0.0 means only image1 is used.
#' A value of 1.0 means only image2 is used. A value between 0.0 and
#' 1.0 means we linearly interpolate the pixel values between the two
#' images. A value greater than 1.0 "extrapolates" the difference
#' between the two pixel values, and we clip the results to values
#' between 0 and 255.
#'
#' @param image1 An image Tensor of shape (num_rows, num_columns, num_channels) (HWC),
#' or (num_rows, num_columns) (HW), or (num_channels, num_rows, num_columns).
#' @param image2 An image Tensor of shape (num_rows, num_columns, num_channels) (HWC),
#' or (num_rows, num_columns) (HW), or (num_channels, num_rows, num_columns).
#' @param factor A floating point value or Tensor of type tf.float32 above 0.0.
#'
#' @return A blended image Tensor of tf$float32.
#'
#' @export
img_blend <- function(image1, image2, factor) {

  args <- list(
    image1 = image1,
    image2 = image2,
    factor = factor
  )

  do.call(tfa$image$blend, args)

}



#' @title Connected components
#'
#' @description Labels the connected components in a batch of images.
#'
#' @details A component is a set of pixels in a single input image, which are
#' all adjacent and all have the same non-zero value. The components
#' using a squared connectivity of one (all equal entries are joined with
#' their neighbors above,below, left, and right). Components across all
#' images have consecutive ids 1 through n.
#' Components are labeled according to the first pixel of the
#' component appearing in row-major order (lexicographic order by
#' image_index_in_batch, row, col).
#' Zero entries all have an output id of 0.
#' This op is equivalent with `scipy.ndimage.measurements.label`
#' on a 2D array with the default structuring element
#' (which is the connectivity used here).
#' @param images A 2D (H, W) or 3D (N, H, W) Tensor of image (integer,
#' floating point and boolean types are supported).
#' @param name The name of the op.
#'
#' @return Components with the same shape as `images`. entries that evaluate to
#' FALSE (e.g. 0/0.0f, FALSE) in `images` have value 0, and all other entries
#' map to a component id > 0.
#'
#' @section Raises:
#' TypeError: if `images` is not 2D or 3D.
#'
#' @export
img_connected_components <- function(images,
                                     name = NULL) {

  args <- list(
    images = images,
    name = name
  )

  do.call(tfa$image$connected_components, args)

}



#' @title Cutout
#'
#' @description Apply cutout (https://arxiv.org/abs/1708.04552) to images.
#'
#' @details This operation applies a (mask_height x mask_width) mask of zeros to
#' a location within `img` specified by the offset. The pixel values filled in will be of the
#' value `replace`. The located where the mask will be applied is randomly
#' chosen uniformly over the whole images.
#'
#' @param images A tensor of shape (batch_size, height, width, channels) (NHWC),
#' (batch_size, channels, height, width)(NCHW).
#' @param mask_size Specifies how big the zero mask that will be generated is that
#' is applied to the images. The mask will be of size (mask_height x mask_width).
#' Note: mask_size should be divisible by 2.
#' @param offset A list of (height, width) or (batch_size, 2)
#' @param constant_values What pixel value to fill in the images in the area that
#' has the cutout mask applied to it.
#' @param data_format A string, one of `channels_last` (default) or `channels_first`.
#' The ordering of the dimensions in the inputs. `channels_last` corresponds to
#' inputs with shape `(batch_size, ..., channels)` while `channels_first` corresponds
#' to inputs with shape `(batch_size, channels, ...)`.
#'
#' @return An image Tensor.
#' @importFrom purrr map
#' @section Raises:
#' InvalidArgumentError: if mask_size can't be divisible by 2.
#'
#' @export
img_cutout <- function(images, mask_size,
                       offset = list(0, 0),
                       constant_values = 0,
                       data_format = "channels_last") {

  args <- list(
    images = images,
    mask_size = mask_size,
    offset = offset,
    constant_values = as.integer(constant_values),
    data_format = data_format
  )

  if(is.list(offset)) {
    args$offset <- map(args$offset, ~ as.character(.) %>% as.integer)
  } else if(is.vector(offset)) {
    args$offset <- as.integer(as.character(args$offset))
  }

  if(is.list(mask_size)) {
    args$mask_size <- map(args$mask_size, ~ as.character(.) %>% as.integer)
  } else if(is.vector(mask_size)) {
    args$mask_size <- as.integer(as.character(args$mask_size))
  }

  do.call(tfa$image$cutout, args)

}



#' @title Dense image warp
#'
#' @description Image warping using per-pixel flow vectors.
#'
#' @details Apply a non-linear warp to the image, where the warp is specified by a
#' dense flow field of offset vectors that define the correspondences of
#' pixel values in the output image back to locations in the source image.
#' Specifically, the pixel value at output[b, j, i, c] is
#' images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c]. The locations specified by
#' this formula do not necessarily map to an int
#' index. Therefore, the pixel value is obtained by bilinear
#' interpolation of the 4 nearest pixels around
#' (b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
#' of the image, we use the nearest pixel values at the image boundary.
#'
#' @param image 4-D float Tensor with shape [batch, height, width, channels].
#' @param flow A 4-D float Tensor with shape [batch, height, width, 2].
#' @param name A name for the operation (optional).
#' @note Note that image and flow can be of type tf$half, tf$float32, or tf$float64, and
#' do not necessarily have to be the same type.
#'
#' @return A 4-D float `Tensor` with shape`[batch, height, width, channels]` and same type as input image.
#'
#' @section Raises:
#' ValueError: if height < 2 or width < 2 or the inputs have the wrong number of dimensions.
#'
#' @export
img_dense_image_warp <- function(image,
                             flow,
                             name = NULL) {

  args <- list(
    image = image,
    flow = flow,
    name = NULL
  )

  do.call(tfa$image$dense_image_warp, args)

}




#' @title Equalize
#'
#' @description Equalize image(s)
#'
#'
#' @param image A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
#' or (num_images, num_channels, num_rows, num_columns) (NCHW), or
#' (num_rows, num_columns, num_channels) (HWC), or (num_channels, num_rows, num_columns) (CHW),
#' or (num_rows, num_columns) (HW). The rank must be statically known (the shape is
#' not TensorShape(None)).
#' @param data_format Either 'channels_first' or 'channels_last'
#' @param name The name of the op. Returns: Image(s) with the same type and
#' shape as `images`, equalized.
#'
#' @return Image(s) with the same type and shape as `images`, equalized.
#'
#' @export
img_equalize <- function(image,
                     data_format = "channels_last",
                     name = NULL) {

  args <- list(
    image = image,
    data_format = data_format,
    name = name
  )

  do.call(tfa$image$equalize, args)

}



#' @title Euclidean dist transform
#'
#' @description Applies euclidean distance transform(s) to the image(s).
#'
#'
#' @param images A tensor of shape (num_images, num_rows, num_columns, 1) (NHWC),
#' or (num_rows, num_columns, 1) (HWC) or (num_rows, num_columns) (HW).
#' @param dtype DType of the output tensor.
#' @param name The name of the op.
#'
#' @return Image(s) with the type `dtype` and same shape as `images`, with the
#' transform applied. If a tensor of all ones is given as input, the output tensor
#' will be filled with the max value of the `dtype`.
#'
#' @section Raises:
#' TypeError: If `image` is not tf.uint8, or `dtype` is not floating point.
#' ValueError: If `image` more than one channel, or `image` is not of rank between 2 and 4.
#'
#' @export
img_euclidean_dist_transform <- function(images, dtype = tf$float32, name = NULL) {

  args <- list(
    images = images,
    dtype = dtype,
    name = name
  )

  do.call(tfa$image$euclidean_dist_transform, args)

}


#' @title Interpolate bilinear
#'
#' @description Similar to Matlab's interp2 function.
#'
#' @details Finds values for query points on a grid using bilinear interpolation.
#' @param grid a 4-D float Tensor of shape [batch, height, width, channels].
#' @param query_points a 3-D float Tensor of N points with shape [batch, N, 2].
#' @param indexing whether the query points are specified as row and column (ij),
#' or Cartesian coordinates (xy).
#' @param name a name for the operation (optional).
#'
#' @return values: a 3-D `Tensor` with shape `[batch, N, channels]`
#'
#' @section Raises:
#' ValueError: if the indexing mode is invalid, or if the shape of the inputs invalid.
#'
#' @export
img_interpolate_bilinear <- function(grid,
                                     query_points,
                                     indexing = 'ij',
                                     name = NULL) {

  args <- list(
    grid = grid,
    query_points = query_points,
    indexing = indexing,
    name = name
  )

  do.call(tfa$image$interpolate_bilinear, args)

}



#' @title Interpolate spline
#'
#' @description Interpolate signal using polyharmonic interpolation.
#'
#' @details The interpolant has the form
#' f(x) = `\\sum_{i = 1}^n w_i \\phi(||x - c_i||) + v^T x + b`. This is a sum of two terms:
#' (1) a weighted sum of radial basis function
#' (RBF) terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term
#' with a bias. The \\(c_i\\) vectors are 'training' points.
#' In the code, b is absorbed into v
#' by appending 1 as a final dimension to x. The coefficients w and v are
#' estimated such that the interpolant exactly fits the value of the function
#' at the \\(c_i\\) points, the vector w is orthogonal to each \\(c_i\\),
#' and the vector w sums to 0. With these constraints, the coefficients
#' can be obtained by solving a linear system. `\\(\\phi\\)` is an RBF, parametrized by
#' an interpolation
#' order. Using order=2 produces the well-known thin-plate spline. We also provide the
#' option to perform regularized interpolation. Here, the
#' interpolant is selected to trade off between the squared loss on the
#' training data and a certain measure of its curvature
#' ([details](https://en.wikipedia.org/wiki/Polyharmonic_spline)).
#' Using a regularization weight greater than zero has the effect that the
#' interpolant will no longer exactly fit the training data. However, it may
#' be less vulnerable to overfitting, particularly for high-order
#' interpolation. Note the interpolation procedure is differentiable with respect to all
#' inputs besides the order parameter. We support dynamically-shaped inputs,
#' where batch_size, n, and m are NULL
#' at graph construction time. However, d and k must be known.
#'
#' @param train_points `[batch_size, n, d]` float `Tensor` of n d-dimensional
#' locations. These do not need to be regularly-spaced.
#' @param train_values `[batch_size, n, k]` float `Tensor` of n c-dimensional
#' values evaluated at train_points.
#' @param query_points `[batch_size, m, d]` `Tensor` of m d-dimensional locations
#' where we will output the interpolant's values.
#' @param order order of the interpolation. Common values are 1
#' for `\\(\\phi(r) = r\\), 2 for \\(\\phi(r) = r^2 * log(r)\\) (thin-plate spline), or 3 for \\(\\phi(r) = r^3\\)`.
#' @param regularization_weight weight placed on the regularization term. This will
#' depend substantially on the problem, and it should always be tuned. For many problems,
#' it is reasonable to use no regularization. If using a non-zero value, we recommend
#' a small value like 0.001.
#' @param name name prefix for ops created by this function
#'
#' @return `[b, m, k]` float `Tensor` of query values. We use train_points and train_values
#' to perform polyharmonic interpolation. The query values are the values of the interpolant
#' evaluated at the locations specified in query_points.
#'
#' @section This is a sum of two terms: (1) a weighted sum of radial basis function:
#' (RBF) terms, with the centers \\(c_1, ... c_n\\), and (2) a linear term with a bias.
#' The \\(c_i\\) vectors are 'training' points. In the code, b is absorbed into v by
#' appending 1 as a final dimension to x. The coefficients w and v are estimated such
#' that the interpolant exactly fits the value of the function at the \\(c_i\\) points,
#' the vector w is orthogonal to each \\(c_i\\), and the vector w sums to 0. With these
#' constraints, the coefficients can be obtained by solving a linear system.
#'
#' @export
img_interpolate_spline <- function(train_points, train_values,
                                   query_points,
                                   order, regularization_weight = 0.0,
                                   name = "interpolate_spline") {

  args <- list(
    train_points = train_points,
    train_values = train_values,
    query_points = query_points,
    order = order,
    regularization_weight = regularization_weight,
    name = name
  )

  do.call(tfa$image$interpolate_spline, args)

}


#' @title Mean filter2d
#'
#' @description Perform mean filtering on image(s).
#'
#' @param image Either a 2-D Tensor of shape [height, width], a 3-D Tensor of
#' shape [height, width, channels], or a 4-D Tensor of
#' shape [batch_size, height, width, channels].
#' @param filter_shape An integer or tuple/list of 2 integers, specifying the height
#' and width of the 2-D mean filter. Can be a single integer to specify the same
#' value for all spatial dimensions.
#' @param padding A string, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type
#' of padding algorithm to use, which is compatible with mode argument in tf.pad.
#' For more details, please refer to https://www.tensorflow.org/api_docs/python/tf/pad.
#' @param constant_values A scalar, the pad value to use in "CONSTANT" padding mode.
#' @param name A name for this operation (optional).
#'
#' @return 3-D or 4-D `Tensor` of the same dtype as input.
#'
#' @section Raises:
#' ValueError: If `image` is not 2, 3 or 4-dimensional, if `padding` is other
#' than "REFLECT", "CONSTANT" or "SYMMETRIC", or if `filter_shape` is invalid.
#' @importFrom purrr map
#' @export
img_mean_filter2d <- function(image,
                          filter_shape = list(3, 3),
                          padding = 'REFLECT',
                          constant_values = 0,
                          name = NULL) {

  args <- list(
    image = image,
    filter_shape = filter_shape,
    padding = padding,
    constant_values = as.integer(constant_values),
    name = name
  )

  if(is.list(filter_shape)) {
    args$filter_shape <- map(args$filter_shape, ~ as.character(.) %>% as.integer)
  } else if(is.vector(filter_shape)) {
    args$filter_shape <- as.integer(as.character(args$filter_shape))
  }

  do.call(tfa$image$mean_filter2d, args)

}


#' @title Median filter2d
#'
#' @description Perform median filtering on image(s).
#'
#' @param image Either a 2-D Tensor of shape [height, width], a 3-D Tensor of
#' shape [height, width, channels], or a 4-D Tensor of
#' shape [batch_size, height, width, channels].
#' @param filter_shape An integer or tuple/list of 2 integers, specifying the height
#' and width of the 2-D median filter. Can be a single integer to specify the same
#' value for all spatial dimensions.
#' @param padding A string, one of "REFLECT", "CONSTANT", or "SYMMETRIC". The type
#' of padding algorithm to use, which is compatible with mode argument in tf.pad. For
#' more details, please refer to https://www.tensorflow.org/api_docs/python/tf/pad.
#' @param constant_values A scalar, the pad value to use in "CONSTANT" padding mode.
#' @param name A name for this operation (optional)
#'
#' @importFrom purrr map
#' @return 3-D or 4-D `Tensor` of the same dtype as input.
#'
#' @section Raises:
#' ValueError: If `image` is not 2, 3 or 4-dimensional, if `padding` is other
#' than "REFLECT", "CONSTANT" or "SYMMETRIC", or if `filter_shape` is invalid.
#'
#' @export
img_median_filter2d <- function(image,
                                filter_shape = list(3, 3),
                                padding = 'REFLECT',
                                constant_values = 0,
                                name = NULL) {

  args <- list(
    image = image,
    filter_shape = filter_shape,
    padding = padding,
    constant_values = as.integer(constant_values),
    name = name
  )

  if(is.list(filter_shape)) {
    args$filter_shape <- map(args$filter_shape, ~ as.character(.) %>% as.integer)
  } else if(is.vector(filter_shape)) {
    args$filter_shape <- as.integer(as.character(args$filter_shape))
  }

  do.call(tfa$image$median_filter2d, args)

}


#' @title Random cutout
#'
#' @description Apply cutout (https://arxiv.org/abs/1708.04552) to images.
#'
#' @details This operation applies a (mask_height x mask_width) mask of zeros to
#' a random location within `img`. The pixel values filled in will be of the
#' value `replace`. The located where the mask will be applied is randomly
#' chosen uniformly over the whole images.
#'
#' @param images A tensor of shape (batch_size, height, width, channels) (NHWC),
#' (batch_size, channels, height, width)(NCHW).
#' @param mask_size Specifies how big the zero mask that will be generated is that
#' is applied to the images. The mask will be of size (mask_height x mask_width).
#' Note: mask_size should be divisible by 2.
#' @param constant_values What pixel value to fill in the images in the area that
#' has the cutout mask applied to it.
#' @param seed An integer. Used in combination with `tf$random$set_seed` to
#' create a reproducible sequence of tensors across multiple calls.
#' @param data_format A string, one of `channels_last` (default) or `channels_first`.
#' The ordering of the dimensions in the inputs. `channels_last` corresponds to inputs
#' with shape `(batch_size, ..., channels)` while `channels_first` corresponds to inputs
#' with shape `(batch_size, channels, ...)`.
#'
#' @return An image Tensor.
#' @importFrom purrr map
#' @section Raises:
#' InvalidArgumentError: if mask_size can't be divisible by 2.
#'
#' @export
img_random_cutout <- function(images, mask_size,
                              constant_values = 0,
                              seed = NULL,
                              data_format = "channels_last") {

  args <- list(
    images = images,
    mask_size = mask_size,
    constant_values = as.integer(constant_values),
    seed = seed,
    data_format = data_format
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)

  if(is.list(mask_size)) {
    args$mask_size <- map(args$mask_size, ~ as.character(.) %>% as.integer)
  } else if(is.vector(mask_size)) {
    args$mask_size <- as.integer(as.character(args$mask_size))
  }

  do.call(tfa$image$random_cutout, args)

}





#' @title Random hsv in yiq
#'
#' @description Adjust hue, saturation, value of an RGB image randomly in YIQ color
#'
#' @details space. Equivalent to `adjust_yiq_hsv()` but uses a `delta_h` randomly
#' picked in the interval `[-max_delta_hue, max_delta_hue]`, a
#' `scale_saturation` randomly picked in the interval
#' `[lower_saturation, upper_saturation]`, and a `scale_value`
#' randomly picked in the interval `[lower_saturation, upper_saturation]`.
#'
#' @param image RGB image or images. Size of the last dimension must be 3.
#' @param max_delta_hue float. Maximum value for the random delta_hue.
#' Passing 0 disables adjusting hue.
#' @param lower_saturation float. Lower bound for the random scale_saturation.
#' @param upper_saturation float. Upper bound for the random scale_saturation.
#' @param lower_value float. Lower bound for the random scale_value.
#' @param upper_value float. Upper bound for the random scale_value.
#' @param seed An operation-specific seed. It will be used in conjunction with
#' the graph-level seed to determine the real seeds that will be used in this
#' operation. Please see the documentation of set_random_seed for its interaction
#' with the graph-level random seed.
#' @param name A name for this operation (optional).
#'
#' @return 3-D float tensor of shape `[height, width, channels]`.
#'
#' @section Raises:
#' ValueError: if `max_delta`, `lower_saturation`, `upper_saturation`,
#' `lower_value`, or `upper_value` is invalid.
#'
#' @export
img_random_hsv_in_yiq <- function(image, max_delta_hue = 0,
                                  lower_saturation = 1,
                                  upper_saturation = 1,
                                  lower_value = 1,
                                  upper_value = 1,
                                  seed = NULL, name = NULL) {

  args <- list(
    image = image,
    max_delta_hue = max_delta_hue,
    lower_saturation = lower_saturation,
    upper_saturation = upper_saturation,
    lower_value = lower_value,
    upper_value = upper_value,
    seed = seed,
    name = name
  )

  if(!is.null(seed))
    args$seed <- as.integer(args$seed)




  do.call(tfa$image$random_hsv_in_yiq, args)

}




#' @title Resampler
#'
#' @description Resamples input data at user defined coordinates.
#'
#' @details The resampler currently only supports bilinear interpolation of 2D data.
#'
#'
#' @param data Tensor of shape [batch_size, data_height, data_width, data_num_channels]
#' containing 2D data that will be resampled.
#' @param warp Tensor of minimum rank 2 containing the coordinates at which resampling
#' will be performed. Since only bilinear interpolation is currently supported, the last
#' dimension of the warp tensor must be 2, representing the (x, y) coordinate where x is
#' the index for width and y is the index for height.
#' @param name Optional name of the op.
#'
#' @return Tensor of resampled values from `data`. The output tensor shape is determined
#' by the shape of the warp tensor. For example, if `data` is of shape
#' `[batch_size, data_height, data_width, data_num_channels]` and warp of
#' shape `[batch_size, dim_0, ... , dim_n, 2]` the output will be of
#' shape `[batch_size, dim_0, ... , dim_n, data_num_channels]`.
#'
#' @section Raises:
#' ImportError: if the wrapper generated during compilation is not present when the function is called.
#'
#' @export
img_resampler <- function(data,
                      warp,
                      name = NULL) {

  args <- list(
    data = data,
    warp = warp,
    name = name
  )


  do.call(tfa$image$resampler, args)

}


#' @title Rotate
#'
#' @description Rotate image(s) counterclockwise by the passed angle(s) in radians.
#'
#'
#' @param images A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
#' (num_rows, num_columns, num_channels) (HWC), or (num_rows, num_columns) (HW).
#' @param angles A scalar angle to rotate all images by, or (if images has rank 4) a vector
#' of length num_images, with an angle for each image in the batch.
#' @param interpolation Interpolation mode. Supported values: "NEAREST", "BILINEAR".
#' @param name The name of the op.
#'
#' @return Image(s) with the same type and shape as `images`, rotated by the given angle(s).
#' Empty space due to the rotation will be filled with zeros.
#' @importFrom purrr map
#' @section Raises:
#' TypeError: If `image` is an invalid type.
#'
#' @export
img_rotate <- function(images, angles,
                   interpolation = "NEAREST",
                   name = NULL) {

  args <- list(
    images = images,
    angles = angles,
    interpolation = interpolation,
    name = name
  )

  if(is.list(angles)) {
    args$angles <- map(args$angles, ~ as.character(.) %>% as.integer)
  } else if(is.vector(angles)) {
    args$angles <- as.integer(as.character(args$angles))
  }

  do.call(tfa$image$rotate, args)

}


#' @title Shear x-axis
#'
#' @description Perform shear operation on an image (x-axis)
#'
#'
#' @param image A 3D image Tensor.
#' @param level A float denoting shear element along y-axis
#' @param replace A one or three value 1D tensor to fill empty pixels.
#'
#' @return Transformed image along X or Y axis, with space outside image filled with replace.
#'
#' @export
img_shear_x <- function(image, level, replace) {

  args <- list(
    image = image,
    level = level,
    replace = replace
  )

  do.call(tfa$image$shear_x, args)

}


#' @title Shear y-axis
#'
#' @description Perform shear operation on an image (y-axis)
#'
#'
#' @param image A 3D image Tensor.
#' @param level A float denoting shear element along x-axis
#' @param replace A one or three value 1D tensor to fill empty pixels.
#'
#' @return Transformed image along X or Y axis, with space outside image filled with replace.
#'
#' @export
img_shear_y <- function(image, level, replace) {

  args <- list(
    image = image,
    level = level,
    replace = replace
  )

  do.call(tfa$image$shear_y, args)

}


#' @title Sparse image warp
#'
#' @description Image warping using correspondences between sparse control points.
#'
#' @details Apply a non-linear warp to the image, where the warp is specified by
#' the source and destination locations of a (potentially small) number of
#' control points. First, we use a polyharmonic spline
#' (`tf$contrib$image$interpolate_spline`) to interpolate the displacements
#' between the corresponding control points to a dense flow field.
#' Then, we warp the image using this dense flow field
#' (`tf$contrib$image$dense_image_warp`). Let t index our control points.
#' For regularization_weight=0, we have:
#' warped_image[b, dest_control_point_locations[b, t, 0],
#' dest_control_point_locations[b, t, 1], :] =
#' image[b, source_control_point_locations[b, t, 0],
#' source_control_point_locations[b, t, 1], :]. For regularization_weight > 0,
#' this condition is met approximately, since
#' regularized interpolation trades off smoothness of the interpolant vs.
#' reconstruction of the interpolant at the control points.
#' See `tf$contrib$image$interpolate_spline` for further documentation of the
#' interpolation_order and regularization_weight arguments.
#'
#' @param image `[batch, height, width, channels]` float `Tensor`
#' @param source_control_point_locations `[batch, num_control_points, 2]` float `Tensor`
#' @param dest_control_point_locations `[batch, num_control_points, 2]` float `Tensor`
#' @param interpolation_order polynomial order used by the spline interpolation
#' @param regularization_weight weight on smoothness regularizer in interpolation
#' @param num_boundary_points How many zero-flow boundary points to include at each image edge.
#' Usage:
#' num_boundary_points=0: don't add zero-flow points
#' num_boundary_points=1: 4 corners of the image
#' num_boundary_points=2: 4 corners and one in the middle of each edge (8 points total)
#' num_boundary_points=n: 4 corners and n-1 along each edge
#' @param name A name for the operation (optional).
#'
#' @return warped_image: `[batch, height, width, channels]` float `Tensor` with
#' same type as input image. flow_field: `[batch, height, width, 2]`
#' float `Tensor` containing the dense flow field produced by the interpolation.
#'
#' @export
img_sparse_image_warp <- function(image, source_control_point_locations,
                                  dest_control_point_locations,
                                  interpolation_order = 2,
                                  regularization_weight = 0.0,
                                  num_boundary_points = 0,
                                  name = "sparse_image_warp") {

  args <- list(
    image = image,
    source_control_point_locations = source_control_point_locations,
    dest_control_point_locations = dest_control_point_locations,
    interpolation_order = as.integer(interpolation_order),
    regularization_weight = regularization_weight,
    num_boundary_points = as.integer(num_boundary_points),
    name = name
  )

  do.call(tfa$image$sparse_image_warp, args)

}



#' @title Transform
#'
#' @description Applies the given transform(s) to the image(s).
#'
#' @param images A tensor of shape (num_images, num_rows, num_columns, num_channels)
#' (NHWC), (num_rows, num_columns, num_channels) (HWC), or (num_rows, num_columns) (HW).
#' @param transforms Projective transform matrix/matrices. A vector of length 8 or tensor
#' of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it
#' maps the output point (x, y) to a transformed input point
#' (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), where k = c0 x + c1 y + 1.
#' The transforms are inverted compared to the transform mapping input points to output points.
#' Note that gradients are not backpropagated into transformation parameters.
#' @param interpolation Interpolation mode. Supported values: "NEAREST", "BILINEAR".
#' @param output_shape Output dimesion after the transform, [height, width]. If NULL,
#' output is the same size as input image.
#' @param name The name of the op.
#' @importFrom purrr map
#' @return Image(s) with the same type and shape as `images`, with
#' the given transform(s) applied. Transformed coordinates outside of the
#' input image will be filled with zeros.
#'
#' @section Raises:
#' TypeError: If `image` is an invalid type. ValueError: If output shape is not 1-D int32 Tensor.
#'
#' @export
img_transform <- function(images,
                          transforms,
                          interpolation = 'NEAREST',
                          output_shape = NULL,
                          name = NULL) {

  args <- list(
    images = images,
    transforms = transforms,
    interpolation = interpolation,
    output_shape = output_shape,
    name = name
  )

  if (!is.null(output_shape))
    if(is.list(output_shape)) {
      args$output_shape <- map(args$output_shape, ~ as.character(.) %>% as.integer)
    } else if(is.vector(output_shape)) {
      args$output_shape <- as.integer(as.character(args$output_shape))
    }

  do.call(tfa$image$transform, args)

}


#' @title Translate
#'
#' @description Translate image(s) by the passed vectors(s).
#'
#' @param images A tensor of shape (num_images, num_rows, num_columns, num_channels) (NHWC),
#' (num_rows, num_columns, num_channels) (HWC), or (num_rows, num_columns) (HW). The rank must
#' be statically known (the shape is not TensorShape(None)).
#' @param translations A vector representing [dx, dy] or (if images has rank 4) a matrix of
#' length num_images, with a [dx, dy] vector for each image in the batch.
#' @param interpolation Interpolation mode. Supported values: "NEAREST", "BILINEAR".
#' @param name The name of the op.
#'
#'
#' @return Image(s) with the same type and shape as `images`, translated by the
#' given vector(s). Empty space due to the translation will be filled with zeros.
#' @importFrom purrr map
#' @section Raises:
#' TypeError: If `images` is an invalid type.
#'
#' @export
img_translate <- function(images,
                          translations,
                          interpolation = 'NEAREST',
                          name = NULL) {

  args <- list(
    images = images,
    translations = translations,
    interpolation = interpolation,
    name = name
  )

  if(is.list(translations)) {
    args$translations <- map(args$translations, ~ as.character(.) %>% as.integer)
  } else if(is.vector(translations)) {
    args$translations <- as.integer(as.character(args$translations))
  }

  do.call(tfa$image$translate, args)

}


#' @title Translate xy dims
#'
#' @description Translates image in X or Y dimension.
#'
#'
#' @param image A 3D image Tensor.
#' @param translate_to A 1D tensor to translate [x, y]
#' @param replace A one or three value 1D tensor to fill empty pixels.
#'
#' @return Translated image along X or Y axis, with space outside image
#' filled with replace. Raises: ValueError: if axis is neither 0 nor 1.
#'
#' @section Raises:
#' ValueError: if axis is neither 0 nor 1.
#'
#' @export
img_translate_xy <- function(image, translate_to, replace) {

  args <- list(
    image = image,
    translate_to = translate_to,
    replace = replace
  )

  do.call(tfa$image$translate_xy, args)

}














