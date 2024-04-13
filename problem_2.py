#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from numpy import linalg
import random

def compute_homography(pts):
    # Construct the A matrix
    A = np.zeros((2 * pts.shape[0], 9))
    for i in range(pts.shape[0]):
        x, y, w, u = pts[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, w * x, w * y, w]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, u * x, u * y, u]
    
    # Solve for the homography matrix
    _, _, vh = np.linalg.svd(A)
    H = np.reshape(vh[-1], (3, 3))
    
    # Normalize the homography matrix
    H /= H[2, 2]
    
    return H


def get_points(img_1, img_2, t, check_resize):
    if check_resize:
        resize_shape = (img_1.shape[1] // 5, img_1.shape[0] // 5)
        img_1 = cv2.resize(img_1, resize_shape)
        img_2 = cv2.resize(img_2, resize_shape)

    sift = cv2.SIFT_create()
    
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = sift.detectAndCompute(gray_1, None)
    kp2, des2 = sift.detectAndCompute(gray_2, None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    pts = []
    for match in matches:
        m, n = match
        if m.distance < t * n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            pts.append([*pt1, *pt2])
    
    pts = np.array(pts)
    
    return pts, img_1, img_2



def distance(pts, H):
    p_1 = np.array([pts[0], pts[1], 1])
    p_2 = np.array([pts[2], pts[3], 1])
    p_1_proj = np.dot(H, p_1)
    p_2_proj = np.dot(H, p_2)
    p_1_proj = p_1_proj / p_1_proj[2]
    p_2_proj = p_2_proj / p_2_proj[2]
    error = np.array([pts[2] - p_1_proj[0], pts[3] - p_1_proj[1], pts[0] - p_2_proj[0], pts[1] - p_2_proj[1]])
    return np.linalg.norm(error)


def ransac(pts, t):
    maxinl = []
    H_fin = None
    for i in range(500):
        four_pts = pts[random.sample(range(len(pts)), 4)]
        H = compute_homography(four_pts)
        inl = []
        for j in range(len(pts)):
            d = distance(pts[j], H)
            if d < t:
                inl.append(pts[j])
        if len(inl) > len(maxinl):
            maxinl = inl
            H_fin = H
    return H_fin


def crop(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)[1]
    white_pixels = np.where(thresh == 255)
    y_min, x_min = np.min(white_pixels, axis=1)
    y_max, x_max = np.max(white_pixels, axis=1)
    img = img[y_min:y_max+1, x_min:x_max+1]
    return img



#Load images
img_1 = cv2.imread('image_1.jpg')
img_2 = cv2.imread('image_2.jpg')
img_3 = cv2.imread('image_3.jpg')
img_4 = cv2.imread('image_4.jpg')

#Get matching points and compute homography for first image pair
point_1, img_1_resized, img_2_resized = get_points(img_1, img_2, 0.5, True)
H_1 = compute_homography(point_1)

#Stitch images 1 and 2 together
stitch_1 = cv2.warpPerspective(img_2_resized, linalg.inv(H_1), (img_1_resized.shape[1] + img_2_resized.shape[1], img_1_resized.shape[0] + img_2_resized.shape[0]))
stitch_1[0:img_1_resized.shape[0], 0:img_1_resized.shape[1]] = img_1_resized

#Crop the stitched image
stitch_1 = crop(stitch_1)

#Compute homography for second image pair
point_2, img_3_resized, img_4_resized = get_points(img_3, img_4, 0.52, True)
H_2 = compute_homography(point_2)

#Stitch images 3 and 4 together
stitch_2 = cv2.warpPerspective(img_4_resized, linalg.inv(H_2), (img_3_resized.shape[1] + img_4_resized.shape[1], img_3_resized.shape[0] + img_4_resized.shape[0]))
stitch_2[0:img_3_resized.shape[0], 0:img_3_resized.shape[1]] = img_3_resized

#Get matching points and compute homography for final image pair
point_3, img_5_resized, img_6_resized = get_points(stitch_1, stitch_2, 0.52, False)
H_3 = compute_homography(point_3)

#Stitch images 1-4 together
stitch_3 = cv2.warpPerspective(img_6_resized, linalg.inv(H_3), (img_5_resized.shape[1] + img_6_resized.shape[1], img_5_resized.shape[0] + img_6_resized.shape[0]))
stitch_3[0:img_5_resized.shape[0], 0:img_5_resized.shape[1]] = img_5_resized

#Resize and display the final stitched image
final = cv2.resize(stitch_3, (int(stitch_3.shape[1]*60/100), int(stitch_3.shape[0]*60/100)))
cv2.imshow("panoroma", final)
cv2.waitKey(0)


# In[ ]:




