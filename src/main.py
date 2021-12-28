from ImageBlender import ImageBlender
import numpy as np
import cv2
image_left = cv2.imread(r"C:\Users\MONSTER\GitHub\PanoromaImageCreator\src\apple.png")
image_right = cv2.imread(r"C:\Users\MONSTER\GitHub\PanoromaImageCreator\src\orange.png")

image_blender = ImageBlender(image_left, image_right, 3, 1.5, 5)

# left_upsamples = image_pyramid_left.get_upsampled_images()
# left_laplacian = image_pyramid_left.get_laplacian_pyramid()
# right_upsamples = image_pyramid_right.get_upsampled_images()
# right_laplacian = image_pyramid_right.get_laplacian_pyramid()

# def apply_upsampling(image):
#   image_height, image_width, image_channel = image.shape
#   upsampled_image = np.zeros((image_height * 2, image_width * 2, image_channel), np.uint8)
#   for row in range(image_height * 2):
#     for column in range(image_width * 2):
#       for channel in range(image_channel):
#         upsampled_image[row][column][channel] = image[row // 2, column // 2, channel]
#   return upsampled_image


# print(len(left_upsamples))
# print(len(left_laplacian))
# print(len(right_upsamples))
# print(len(right_laplacian))


# temp = []
# for img in range(len(left_upsamples) - 1, -1, - 1):
#   if img == len(left_upsamples) - 1:
#     image = cv2.add(left_upsamples[img], right_upsamples[img])
#     image = cv2.add(image, cv2.add(left_laplacian[img], right_laplacian[img]))
#     temp.append(image)
  
