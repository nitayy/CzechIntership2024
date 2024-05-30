import cv2
import numpy as np

# Read the image
image = cv2.imread('./tests/test7.jpg', cv2.IMREAD_GRAYSCALE)

# Denoise the image with a bilateral filter
denoised = cv2.bilateralFilter(image, 9, 75, 75)

# Increase contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(denoised)

# Use background subtraction
bg = cv2.medianBlur(contrast_enhanced, 21)
subtract = cv2.absdiff(contrast_enhanced, bg)

# Apply Adaptive Thresholding
thresh = cv2.adaptiveThreshold(subtract, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Apply Morphological Operations (Opening to remove noise, then closing to enhance features)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=3)

# Apply Laplacian Edge Detection for edge enhancement
laplacian = cv2.Laplacian(closed, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# Apply Canny Edge Detection
edges = cv2.Canny(laplacian, 50, 150)

# Apply Dilation
dilated = cv2.dilate(edges, kernel, iterations=3)

# Find Contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw Contours
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Denoised Image', denoised)
cv2.imshow('Contrast Enhanced', contrast_enhanced)
cv2.imshow('Background Subtracted', subtract)
cv2.imshow('Adaptive Thresholding', thresh)
cv2.imshow('Morphological Operations', closed)
cv2.imshow('Laplacian Edge Detection', laplacian)
cv2.imshow('Canny Edges', edges)
cv2.imshow('Dilated Edges', dilated)
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




# image = cv2.imread("test45.jpg")
# for i, contour in enumerate(filtered_contours):
#     # Get the bounding rectangle
#     x, y, w, h = cv2.boundingRect(contour)
#     print(x,y,w,h)
#     # Crop the image using the bounding rectangle
#     cropped_image = image[y:y+h, x:x+w]
#     # Save the cropped image
#     cv2.imwrite(f'test45/test45-output{i+1}.jpg', cropped_image)