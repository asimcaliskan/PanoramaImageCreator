import cv2
import numpy as np
import sys
from ImageBlender import ImageBlender

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.SIFT_create()

    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        return H

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        barrier = img1.shape[1]
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier : barrier ] = np.tile(np.linspace(1, 0, 0).T, (height_panorama, 1))
            mask[:, :barrier] = 1
        else:
            mask[:, barrier :barrier] = np.tile(np.linspace(0, 1, 0 ).T, (height_panorama, 1))
            mask[:, barrier :] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        H = self.registration(img1,img2)
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))
        result = np.concatenate((panorama1[: , 0: panorama1.shape[1] // 2 ], panorama2[:, panorama2.shape[1] // 2 : panorama2.shape[1]]), axis=1)
        # asd = panorama2[:, panorama2.shape[1] // 2 : panorama2.shape[1]]
        # cv2.imwrite("x1.jpg", asd)
        ImageBlender(result[: , 0: panorama1.shape[1] // 2 ], result[:, panorama2.shape[1] // 2 : panorama2.shape[1]], 3, 1.5, 9)

def main(argv1,argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    Image_Stitching().blending(img1,img2)
    
if __name__ == '__main__':
  main(r"C:\Users\MONSTER\GitHub\PanoromaImageCreator\src\im1.jpg", r"C:\Users\MONSTER\GitHub\PanoromaImageCreator\src\im2.jpg")
    
