import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_lut(x, y):
    lut = np.interp(np.arange(256), x, y).astype('uint8')
    return lut

# Define the curve you want to apply (you can customize this)
# For example, let's create a simple S-curve
x = np.array([0, 4, 128, 192, 255])
y = np.array([0, 70, 128, 185, 255])

# Create LUT for each color channel
lut = create_lut(x, y)

# Load an image
image = cv2.imread('test2.jpg')

# Apply the LUT to each channel of the image
image_b = cv2.LUT(image[:,:,0], lut)
image_g = cv2.LUT(image[:,:,1], lut)
image_r = cv2.LUT(image[:,:,2], lut)

# Merge the channels back
image_adjusted = cv2.merge((image_b, image_g, image_r))

# Show the original and adjusted images
cv2.imshow('Original Image', image)
cv2.imshow('Adjusted Image', image_adjusted)
cv2.waitKey(0)
cv2.destroyAllWindows()
