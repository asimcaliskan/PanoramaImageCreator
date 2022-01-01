import cv2
import numpy as np
class ImageBlender:
  def __init__(self, left_image, right_image, pyramid_length, sigma, filter_dimension):
      self.left_image = left_image
      self.right_image = right_image
      self.filter_dimension = filter_dimension
      self.sigma = sigma
      self.pyramid_length = pyramid_length
      self.left_gauss_pyramid = []
      self.right_gauss_pyramid = []
      self.left_upsampled_images = []
      self.right_upsampled_images = []
      self.left_laplacian_pyramid = []
      self.right_lapcalican_pyramid = []
      self.combined_laplacian_pyramid = []
      self.blended_images = []

  def apply_gauss(self, image):
    # image_height, image_width, image_channel = image.shape
    # filter = self.get_gaussian_filter()
    # padding_value = self.filter_dimension // 2
    # for row in range(padding_value, image_height - padding_value):
    #   for column in range(padding_value, image_width - padding_value):
    #     for channel in range(image_channel):
    #       image_part = image[row - padding_value: row + padding_value + 1, column - padding_value: column + padding_value + 1, channel]
    #       image[row][column][channel] = np.sum(np.multiply(image_part, filter))
    return cv2.GaussianBlur(image, (self.filter_dimension, self.filter_dimension), self.sigma)

  def apply_subsampling(self, image):
    image = self.apply_gauss(image)
    image_height, image_width, image_channel = image.shape
    subsampled_image = np.zeros((image_height // 2, image_width // 2, image_channel), np.uint8)
    for row in range(image_height // 2):
      for column in range(image_width // 2):
        for channel in range(image_channel):
          subsampled_image[row][column][channel] = image[row * 2, column * 2, channel]
    return subsampled_image

  def apply_upsampling(self, image, shape):
    image_height, image_width, image_channel = image.shape
    upsampled_image = np.zeros(shape, np.uint8)
    for row in range(image_height * 2):
      for column in range(image_width * 2):
        for channel in range(image_channel):
          upsampled_image[row][column][channel] = image[row // 2, column // 2, channel]
    return self.apply_gauss(upsampled_image)

  def apply_half_image_mask(self, image):
    image_height, image_width, image_channel = image.shape
    for row in range(image_height):
      for column in range(image_width):
        for channel in range(image_channel):
          if self.left_or_right == "right" and column < image_width // 2:
            image[row][column][channel] = 0
          elif self.left_or_right == "left" and column >= image_width // 2:
            image[row][column][channel] = 0
  
  def blend(self):
    #crate left and right gauss pyramid
    #left
    self.left_gauss_pyramid.append(self.left_image)
    for _ in range(self.pyramid_length - 1):
      self.left_gauss_pyramid.append(self.apply_subsampling(np.copy(self.left_gauss_pyramid[-1])))
    
    #right
    self.right_gauss_pyramid.append(self.right_image)
    for _ in range(self.pyramid_length - 1):
      self.right_gauss_pyramid.append(self.apply_subsampling(np.copy(self.right_gauss_pyramid[-1])))
    #
    
    #create left and right upsampled images list
    #left
    for img in range(1, len(self.left_gauss_pyramid)):
      upsampled_image = self.apply_upsampling(np.copy(self.left_gauss_pyramid[img]), self.left_gauss_pyramid[img - 1].shape)
      self.left_upsampled_images.append(upsampled_image)
    
    #right
    for img in range(1, len(self.right_gauss_pyramid)):
      upsampled_image = self.apply_upsampling(np.copy(self.right_gauss_pyramid[img]), self.right_gauss_pyramid[img - 1].shape)
      self.right_upsampled_images.append(upsampled_image)
    #

    #create left and right laplacian pyramid list
    #left
    for img in range(len(self.left_upsampled_images)):
      self.left_laplacian_pyramid.append(np.subtract(self.left_gauss_pyramid[img], self.left_upsampled_images[img]))
    
    #right
    for img in range(len(self.right_upsampled_images)):
      self.right_lapcalican_pyramid.append(np.subtract(self.right_gauss_pyramid[img], self.right_upsampled_images[img]))
    #

    #create combined laplacian pyramid
    for img in range(len(self.right_lapcalican_pyramid)):
      self.combined_laplacian_pyramid.append(np.concatenate((self.left_laplacian_pyramid[img], self.right_lapcalican_pyramid[img]), axis=1))  

    #blend
    for img in range(len(self.left_upsampled_images) - 1, -1, -1):
      if len(self.blended_images) == 0:
        concatenated_upsamples = np.concatenate((self.left_upsampled_images[img], self.right_upsampled_images[img]), axis=1)
        self.blended_images.append(self.apply_upsampling(np.add(concatenated_upsamples, self.combined_laplacian_pyramid[img]), self.combined_laplacian_pyramid[img - 1].shape))
      elif img > 0:
        self.blended_images.append(self.apply_upsampling(np.add(self.blended_images[-1], self.combined_laplacian_pyramid[img]), self.combined_laplacian_pyramid[img - 1].shape))
      else:
        self.blended_images.append(np.add(self.blended_images[-1], self.combined_laplacian_pyramid[img]))
    #self.write_list(self.blended_images)
    result = self.blended_images[-1].astype(np.uint8)
    return result

  
  def write_list(self, image_list):
    image_counter = 0
    for img in image_list:
      cv2.imwrite(str(image_counter) + ".jpg", img)
      image_counter += 1

  def get_laplacian_pyramid(self):
    return self.laplacian_images

  def get_upsampled_images(self):
    return self.subsampled_images

  def get_gaussian_filter(self):
    x_axis, y_axis = np.mgrid[-self.filter_dimension // 2 + 1: self.filter_dimension // 2 + 1, -self.filter_dimension // 2 + 1: self.filter_dimension // 2 + 1]
    gaussian_filter = np.exp(-((x_axis ** 2 + y_axis ** 2 ) / (2.0 * self.sigma ** 2)))
    return gaussian_filter/gaussian_filter.sum()
