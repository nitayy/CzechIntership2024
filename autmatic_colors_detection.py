import cv2
import numpy as np
import os


def create_lut_1d(x, y):
    """Creates a 1D lookup table for given x and y curve points."""
    return np.interp(np.arange(256), x, y).astype(np.uint8)


def apply_curve(image, curve):
    """Applies the given curve (LUT) to each channel of the image."""
    return cv2.LUT(image, curve)


def save_contour_images(image, contours, base_filename):
    """Saves each contour as a separate image."""
    for i, contour in enumerate(contours):
        contour_img = np.zeros_like(image)
        cv2.drawContours(contour_img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imwrite(f"{base_filename}-output{i + 1}.jpg", contour_img)


def process_image(image, luts, edge_bounds):
    """Processes the image with different LUTs and edge bounds, finds contours, and saves them."""
    height, width = image.shape[:2]
    count = 0
    i = 1
    edges = None
    image_with_contours = None
    print(len(luts))
    for lut in luts:
        output_image = image.copy()
        grid_positions = [(0, 0), (0, width // 2), (height // 2, 0), (height // 2, width // 2)]

        for idx, (start_y, start_x) in enumerate(grid_positions):
            end_y = start_y + height // 2
            end_x = start_x + width // 2
            region = output_image[start_y:end_y, start_x:end_x]
            if len(region.shape) == 3:  # Color image
                adjusted_region = cv2.merge([apply_curve(region[:, :, i], lut) for i in range(3)])
            else:  # Grayscale image
                adjusted_region = apply_curve(region, lut)
            output_image[start_y:end_y, start_x:end_x] = adjusted_region

        blurred_image = cv2.GaussianBlur(output_image, (5, 5), 0)
        print("AA")
        cv2.imwrite(f"test2-{i}.jpg",blurred_image,)
        for low, high in edge_bounds:
            edges = cv2.Canny(blurred_image, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1200]

            # if filtered_contours:
            #     count += 1
            #     save_contour_images(image, filtered_contours, f"test2-{count}")
            # image_with_contours = cv2.drawContours(output_image.copy(), contours, -1, (0, 255, 0), 1)



# Parameters
dots = 10
positions = 1000
edge_array = 6

# Define the points for LUTs
points = [
    (np.array([0, 128, 255]), np.array([0, 180, 255])),  # Curve 1
    (np.array([0, 64, 192, 255]), np.array([0, 80, 200, 255])),  # Curve 2
    (np.array([0, 255]), np.array([0, 255]))  # Identity Curve
]

# Create LUTs
luts = [create_lut_1d(x, y) for x, y in points[:dots]]

# Generate edge detection bounds
edge_bounds = [(low, high) for low in np.linspace(0, 255, edge_array) for high in np.linspace(0, 255, edge_array) if
               low < high]

# Load an image
img = "test2.jpg"
image = cv2.imread(img)

# Process the image
process_image(image, luts, edge_bounds)

# Inform the user about the process completion
print("Contour images saved successfully.")