import cv2
import numpy as np
class ImagePyramid:
  def __init__(self, image, filter_dimension, sigma, pyramid_length, left_or_right):
      self.image = image
      self.filter_dimension = filter_dimension
      self.sigma = sigma
      self.pyramid_length = pyramid_length
      self.subsampled_images = []
      self.upsampled_images = []
      self.laplacian_images = []
      self.left_or_right = left_or_right
      self.build_image_pyramid()

  def apply_gauss(self, image):
    image_height, image_width, image_channel = image.shape
    filter = self.get_gaussian_filter()
    padding_value = self.filter_dimension // 2
    for row in range(padding_value, image_height - padding_value):
      for column in range(padding_value, image_width - padding_value):
        for channel in range(image_channel):
          image_part = image[row - padding_value: row + padding_value + 1, column - padding_value: column + padding_value + 1, channel]
          image[row][column][channel] = np.sum(np.multiply(image_part, filter))
    return image

  def apply_subsampling(self, image):
    image_height, image_width, image_channel = image.shape
    subsampled_image = np.zeros((image_height // 2, image_width // 2, image_channel), np.uint8)
    for row in range(image_height // 2):
      for column in range(image_width // 2):
        for channel in range(image_channel):
          subsampled_image[row][column][channel] = image[row * 2, column * 2, channel]
    return subsampled_image

  def apply_upsampling(self, image):
    image_height, image_width, image_channel = image.shape
    upsampled_image = np.zeros((image_height * 2, image_width * 2, image_channel), np.uint8)
    for row in range(image_height * 2):
      for column in range(image_width * 2):
        for channel in range(image_channel):
          upsampled_image[row][column][channel] = image[row // 2, column // 2, channel]
    return upsampled_image

  def apply_half_image_mask(self, image):
    image_height, image_width, image_channel = image.shape
    for row in range(image_height):
      for column in range(image_width):
        for channel in range(image_channel):
          if self.left_or_right == "right" and column < image_width // 2:
            image[row][column][channel] = 0
          elif self.left_or_right == "left" and column >= image_width // 2:
            image[row][column][channel] = 0

  def build_image_pyramid(self):
    #apply subsampling
    self.subsampled_images.append(np.copy(self.image))
    for _ in range(self.pyramid_length):
      gauss_applied_image = self.apply_gauss(self.subsampled_images[-1])
      self.subsampled_images.append(self.apply_subsampling(np.copy(gauss_applied_image)))

    #apply upsampling
    for img in range(1, len(self.subsampled_images)):
      self.upsampled_images.append(self.apply_upsampling(self.subsampled_images[img]))

    #subtract subsampled images from upsampled images
    for img in range(len(self.upsampled_images)):
      self.laplacian_images.append(cv2.subtract(self.subsampled_images[img], self.upsampled_images[img]))
    
    #apply mask
    for img in range(len(self.laplacian_images)):
      self.apply_half_image_mask(self.laplacian_images[img])

    #apply mask
    for img in range(len(self.upsampled_images)):
      self.apply_half_image_mask(self.upsampled_images[img])

  def get_laplacian_pyramid(self):
    return self.laplacian_images

  def get_upsampled_images(self):
    return self.subsampled_images

  def get_gaussian_filter(self):
    x_axis, y_axis = np.mgrid[-self.filter_dimension // 2 + 1: self.filter_dimension // 2 + 1, -self.filter_dimension // 2 + 1: self.filter_dimension // 2 + 1]
    gaussian_filter = np.exp(-((x_axis ** 2 + y_axis ** 2 ) / (2.0 * self.sigma ** 2)))
    return gaussian_filter/gaussian_filter.sum()
