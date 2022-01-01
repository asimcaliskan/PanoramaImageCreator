from ImageBlender import ImageBlender
import numpy as np
import cv2

"""
!!!ATTENTION!!!
TO RUN CODE
PLEASE GIVE THE PATH OF THE IMAGES IN THE "input_images" LIST
0 INDEX(LEFT) ---> N INDEX(RIGHT)
"""
input_images = [r"C:\Users\MONSTER\GitHub\PanoramaImageCreator\src\1.jpg", r"C:\Users\MONSTER\GitHub\PanoramaImageCreator\src\2.jpg", r"C:\Users\MONSTER\GitHub\PanoramaImageCreator\src\3.jpg"]

class ImageStitching():
    def __init__(self) :
        self.ratio = 0.85
        self.min_match = 15
        self.sift = cv2.SIFT_create()

    def registration(self, left_image, right_image):
        key_points_left_image, left_image_destination = self.sift.detectAndCompute(left_image, None)
        key_points_right_image, right_image_destination = self.sift.detectAndCompute(right_image, None)
        brute_force_matcher = cv2.BFMatcher()
        raw_matches = brute_force_matcher.knnMatch(left_image_destination, right_image_destination, k=2)
        good_matches=[]
        good_points = []
        for mL, mR in raw_matches:
            if mL.distance < self.ratio * mR.distance:
                good_points.append((mL.trainIdx, mR.queryIdx))
                good_matches.append([mL])
        #img3 = cv2.drawMatchesKnn(left_image, key_points_left_image, right_image, key_points_right_image, good_matches, None, flags=2)
        #cv2.imwrite('---.jpg', img3)
        if len(good_points) > self.min_match:
            left_image_key_points = np.float32(
                [key_points_left_image[i].pt for (_, i) in good_points])
            right_image_key_points = np.float32(
                [key_points_right_image[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(right_image_key_points, left_image_key_points, cv2.RANSAC, 5.0)
        return H

    def blending(self, left_image, right_image):
        H = self.registration(left_image, right_image)
        """
        +----------+----------+
        |          |          |
        |   left   |  right   |
        |  image   |  image   |
        |          |          |
        +----------+----------+
        blended image height = left or right image height
        blended image width  = left image width + right image width
        """
        left_image_height = left_image.shape[0]
        left_image_width = left_image.shape[1]
        right_image_width = right_image.shape[1]
        blended_image_height= left_image_height
        blended_image_width = left_image_width + right_image_width
        blended_image_with_left_image = np.zeros((blended_image_height, blended_image_width, 3))
        blended_image_with_left_image[0 : left_image.shape[0], 0 : left_image.shape[1], : ] = left_image
        H_applied_right_image = cv2.warpPerspective(right_image, H, (blended_image_width, blended_image_height))
        result = np.concatenate((blended_image_with_left_image[: , 0: blended_image_width // 2 ], H_applied_right_image[:, blended_image_width // 2 : blended_image_width]), axis=1)
        image_blender = ImageBlender(result[: , 0: blended_image_width // 2 ], result[:, blended_image_width // 2 : blended_image_width], 4, 1.5, 7)
        blended_image = image_blender.blend()
        return blended_image

def main():
    number_of_images = len(input_images)
    left_index = number_of_images // 2 - 1
    right_index = number_of_images // 2 + 1
    stitching_direction = "left"
    blended_image = cv2.imread(input_images[number_of_images // 2])
    while True:
        if stitching_direction == "right" and right_index < number_of_images:
            right_image = cv2.imread(input_images[right_index])
            blended_image = ImageStitching().blending(blended_image, right_image)
            right_index += 1
            stitching_direction = "left"

        elif stitching_direction == "left" and -1 < left_index:
            left_image = cv2.imread(input_images[left_index])
            blended_image = ImageStitching().blending(left_image, blended_image)
            left_index -= 1
            stitching_direction = "right"
        else:
            break
    cv2.imwrite("panorama.jpg", blended_image)
    
    
if __name__ == '__main__':
    main()
    
