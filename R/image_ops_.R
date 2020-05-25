#' @title From 4D image
#'
#' @description Convert back to an image with `ndims` rank.
#'
#'
#' @param image 4D tensor.
#' @param ndims The original rank of the image.
#'
#' @return `ndims`-D tensor with the same type.
#'
#' @export
img_from_4D <- function(image, ndims) {

  args <- list(
    image = image,
    ndims = as.integer(ndims)
  )

  do.call(tfa$image$color_ops$from_4D_image, args)

}


#' @title To 4D image
#'
#' @description Convert 2/3/4D image to 4D image.
#'
#'
#' @param image 2/3/4D tensor.
#'
#' @return 4D tensor with the same type.
#'
#' @export
img_to_4D <- function(image) {

  args <- list(
    image = image
  )

  do.call(tfa$image$color_ops$to_4D_image, args)

}

#' @title Sharpness
#'
#' @description Change sharpness of image(s)
#'
#'
#' @param image an image
#' @param factor A floating point value or Tensor above 0.0.
#'
#' @return Image(s) with the same type and shape as `images`, sharper.
#'
#' @export
sharpness <- function(image, factor) {

  args <- list(
    image = image,
    factor = factor
  )

  do.call(tfa$image$color_ops$sharpness, args)

}


#' @title Angles to projective transforms
#'
#' @description Returns projective transform(s) for the given angle(s).
#'
#'
#' @param angles A scalar angle to rotate all images by, or (for batches of images)
#' a vector with an angle to rotate each image in the batch. The rank must be statically
#' known (the shape is not `TensorShape(NULL)`.
#' @param image_height Height of the image(s) to be transformed.
#' @param image_width Width of the image(s) to be transformed.
#' @param name name of the op.
#'
#' @return A tensor of shape (num_images, 8). Projective transforms which can be given to `transform` op.
#'
#' @export
img_angles_to_projective_transforms <- function(angles, image_height,
                                                image_width, name = NULL) {

  args <- list(
    angles = angles,
    image_height = image_height,
    image_width = image_width,
    name = name
  )

  do.call(tfa$image$transform_ops$angles_to_projective_transforms, args)

}


#' @title Compose transforms
#'
#' @description Composes the transforms tensors.
#'
#'
#' @param transforms List of image projective transforms to be composed.
#' Each transform is length 8 (single transform) or shape (N, 8) (batched transforms).
#' The shapes of all inputs must be equal, and at least one input must be given.
#' @param name The name for the op.
#'
#' @return A composed transform tensor. When passed to `transform` op, equivalent to
#' applying each of the given transforms to the image in order.
#'
#' @export
img_compose_transforms <- function(transforms, name = NULL) {

  args <- list(
    transforms = transforms,
    name = name
  )

  do.call(tfa$image$transform_ops$compose_transforms, args)

}



#' @title Flat transforms to matrices
#'
#' @description Converts projective transforms to affine matrices.
#'
#' @details Note that the output matrices map output coordinates to input coordinates.
#' For the forward transformation matrix, call `tf$linalg$inv` on the result.
#'
#' @param transforms Vector of length 8, or batches of transforms with shape `(N, 8)`.
#' @param name The name for the op.
#'
#' @return 3D tensor of matrices with shape `(N, 3, 3)`. The output matrices
#' map the *output coordinates* (in homogeneous coordinates) of each transform
#' to the corresponding *input coordinates*.
#'
#' @section Raises:
#' ValueError: If `transforms` have an invalid shape.
#'
#' @export
img_flat_transforms_to_matrices <- function(transforms, name = NULL) {

  args <- list(
    transforms = transforms,
    name = name
  )

  do.call(tfa$image$transform_ops$flat_transforms_to_matrices, args)

}



#' @title Matrices to flat transforms
#'
#' @description Converts affine matrices to projective transforms.
#'
#' @details Note that we expect matrices that map output coordinates to input
#' coordinates. To convert forward transformation matrices,
#' call `tf$linalg$inv` on the matrices and use the result here.
#'
#' @param transform_matrices One or more affine transformation matrices, for the
#' reverse transformation in homogeneous coordinates. Shape `c(3, 3)` or `c(N, 3, 3)`.
#' @param name The name for the op.
#'
#' @return 2D tensor of flat transforms with shape `(N, 8)`, which may be passed into `transform` op.
#'
#' @section Raises:
#' ValueError: If `transform_matrices` have an invalid shape.
#'
#' @export
img_matrices_to_flat_transforms <- function(transform_matrices, name = NULL) {

  args <- list(
    transform_matrices = transform_matrices,
    name = name
  )

  do.call(tfa$image$transform_ops$matrices_to_flat_transforms, args)

}


#' @title Translations to projective transforms
#'
#' @description Returns projective transform(s) for the given translation(s).
#'
#'
#' @param translations A 2-element list representing [dx, dy] or a matrix of 2-element
#' lists representing [dx, dy] to translate for each image (for a batch of images). The
#' rank must be statically known (the shape is not `TensorShape(NULL)`).
#' @param name The name of the op.
#'
#' @return A tensor of shape c(num_images, 8) projective transforms which can be given to `img_transform`.
#'
#' @export
img_translations_to_projective_transforms <- function(translations, name = NULL) {

  python_function_result <- tfa$image$translate_ops$translations_to_projective_transforms(
    translations = translations,
    name = name
  )

}


#' @title Get ndims
#' @description Print dimensions
#'
#' @param image image
#' @return dimensions of the image
#' @export
img_get_ndims <- function(image) {

  args <- list(
    image = image
  )

  do.call(tfa$image$utils$get_ndims,args)

}



#' @title Wrap
#'
#' @description wrap an image array
#'
#' @param image a 3D Image Tensor with 4 channels.
#' @return 'image' with an extra channel set to all 1s.
#' @export
img_wrap <- function(image) {

  args <- list(
    image = image
  )

  do.call(tfa$image$utils$wrap,args)

}


#' @title Uwrap
#'
#' @description Unwraps an image produced by wrap.
#'
#' @details Where there is a 0 in the last channel for every spatial position,
#' the rest of the three channels in that spatial dimension are grayed (set to 128).
#' Operations like translate and shear on a wrapped Tensor will leave 0s in empty
#' locations. Some transformations look at the intensity of values to do preprocessing,
#' and we want these empty pixels to assume the 'average' value, rather than pure black.
#'
#' @param image image
#' @param replace a one or three value 1D tensor to fill empty pixels.
#' @return a 3D image Tensor with 3 channels.
#' @export
img_unwrap <- function(image,replace) {

  args <- list(
    image = image,
    replace = replace
  )

  do.call(tfa$image$utils$wrap,args)

}












