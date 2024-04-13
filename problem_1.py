#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
video = cv2.VideoCapture("project2.avi")
world_coordinates = [[0, 0], [21.6, 0], [0, 27.9], [21.6, 27.9]]
k = np.array([[1.38e3, 0, 9.46e2], [0, 1.38e3, 5.27e2], [0, 0, 1]])
YAW = []
ROLL = []
PITCH = []
X = []
Y = []
Z = []
frame_count = 0

def HoughTransform(image):
    x, y = np.where(image > 0)
    width, height = image.shape
    thetas = np.deg2rad(np.arange(-90, 90))
    max_dist = int(np.ceil(np.sqrt(width * 2 + height * 2)))

    accumulator = {}
    for cx, cy in zip(x, y):
        for theta in thetas:
            rho = np.round(cx * np.sin(theta) + cy * np.cos(theta)) + max_dist
            if (rho, theta) in accumulator:
                accumulator[(rho, theta)] += 1
            else:
                accumulator[(rho, theta)] = 1

    return accumulator, max_dist

def detect_corners(accumulator, max_dist):
    corner_points = []
    
    # sort accumulator values in descending order
    sorted_accumulator = sorted(accumulator.items(), key=lambda x: x[1])
    sorted_accumulator.reverse()
    
    # get rho and theta values for top 4 accumulators
    rho = [sorted_accumulator[i][0][0] - max_dist for i in range(4)]
    theta = [sorted_accumulator[i][0][1] for i in range(4)]
    
    # get slope and intercept values from rho and theta
    m = [-math.cos(theta[i]) / math.sin(theta[i]) for i in range(len(theta))]
    b = [rho[i] / math.sin(theta[i]) for i in range(len(theta))]
    
    # calculate corners
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            try:
                m1, b1 = m[i], b[i]
                m2, b2 = m[j], b[j]
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                if x > 0 and y > 0:
                    corner_points.append((int(x), int(y)))
            except:
                continue
                
    return corner_points

def compute_homography(corners, world_coordinates, intrinsic_matrix):
    """
    Computes the homography matrix using the given corner points and world coordinates, and the intrinsic matrix.

    Args:
        corners: A list of (x, y) pixel coordinates of the corners of the paper.
        world_coordinates: A list of (u, v) real-world coordinates of the corners of the paper.
        intrinsic_matrix: A 3x3 numpy array representing the intrinsic matrix of the camera.

    Returns:
        A normalized homography matrix that maps pixel coordinates to real-world coordinates.
    """

    # Construct the matrix A for Ax=0
    A = []
    for i in range(len(corners)):
        x, y = corners[i]
        u, v = world_coordinates[i]
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    A = np.array(A)

    # Perform SVD on A
    _, _, V = np.linalg.svd(A)

    # Last column of V is the nullspace vector corresponding to the smallest singular value
    h = V[-1, :] / V[-1, -1]

    # Reshape the vector h into a 3x3 matrix H
    H = h.reshape((3, 3))

    # Normalize the homography matrix using the intrinsic matrix
    H_norm = np.dot(np.dot(np.linalg.inv(intrinsic_matrix), H), intrinsic_matrix)

    return H_norm

def get_r_t(H, intrinsic_matrix):
    # Get the columns of the homography matrix H
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

    # Compute the scale factor lambda
    invintrinsic_matrix = np.linalg.inv(intrinsic_matrix)
    lambd = 1 / np.linalg.norm(invintrinsic_matrix @ h1)

    # Compute the rotation matrix columns
    r1 = lambd * invintrinsic_matrix @ h1
    r2 = lambd * invintrinsic_matrix @ h2
    r3 = np.cross(r1, r2)

    # Compute the translation vector
    t = lambd * invintrinsic_matrix @ h3

    # Combine the rotation and translation into a 4x4 matrix
    Rt = np.column_stack((r1, r2, r3, t))
    P = np.vstack((Rt, np.array([0, 0, 0, 1])))

    return P

def plot(roll, pitch, yaw, X, Y, Z, frame_count):
    fig, Ax = plt.subplots(6,1,sharex=True, figsize=(5,6))
    for i, (title, data) in enumerate(zip(["yaw ", "pitch ", "roll", "X ", "Y ", "Z "], [yaw, pitch, roll, X, Y, Z])):
        Ax[i].plot(range(frame_count), data)
        Ax[i].set_ylabel(title)
    Ax[-1].set_xlabel("frames")
    fig.suptitle(' Plot')
    fig.tight_layout()
    fig.savefig("P1_output.png")
    
while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = video.read()

   # blur the frame to extract the paper
    blur = cv2.GaussianBlur(frame, (7, 7), cv2.BORDER_DEFAULT)
        
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 0, 255)

    x, l = HoughTransform(edges)

    y = detect_corners(x, l)

    for i in range(len(y)):
        cv2.circle(frame, (y[i][0], y[i][1]), 10, (0, 0, 255), -1)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)

# wait 20 milliseconds between frames and break the loop if the `q` key is pressed
    if cv2.waitKey(20) == ord('q'):
        break

# we also need to close the video and destroy all Windows
cv2.destroyAllWindows()
video_.release()
plot(ROLL, PITCH, YAW, X, Y, Z, frame_count)
print("\nX:")
print(X)
print("\nY:")
print(Y)
print("\nZ:")
print(Z)


# In[ ]:




