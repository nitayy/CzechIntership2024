import cv2
import numpy as np
import os


def create_lut_1d(x, y):
    """Creates a 1D lookup table for given x and y curve points."""
    return np.interp(np.arange(256), x, y).astype(np.uint8)


def apply_curve(image, curve):
    """Applies the given curve (LUT) to each channel of the image."""
    return cv2.LUT(image, curve)


def save_processed_images(image, base_filename, count):
    """Saves the processed image."""
    cv2.imwrite(f"{base_filename}-output{count}.jpg", image)


def save_contour_images(image, contours, base_filename):
    """Saves each contour as a separate image on the processed image."""
    processed_image = image.copy()
    for i, contour in enumerate(contours):
        cv2.drawContours(processed_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imwrite(f"{base_filename}-output{i + 1}.jpg", processed_image)


def process_image(image, luts, edge_bounds):
    """Processes the image with different LUTs and edge bounds, saves the processed images."""
    height, width = image.shape[:2]
    count = 0

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

        for low, high in edge_bounds:
            edges = cv2.Canny(blurred_image, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1200]

            if filtered_contours:
                count += 1
                save_processed_images(output_image, f"image2-{count}-processed", count)
                save_contour_images(output_image, filtered_contours, f"image2-{count}-contours")


# Parameters
dots = 3
positions = 4
edge_array = 6

# Define the points for LUTs
points = [
    (np.linspace(0, 255, dots), np.linspace(0, 255, dots)),  # Curve 1
    (np.linspace(0, 255, dots), np.linspace(0, 255, dots)),  # Curve 2
    (np.linspace(0, 255, dots), np.linspace(0, 255, dots)),  # Curve 3
]

# Create LUTs
luts = [create_lut_1d(x, y) for x, y in points[:dots]]

# Generate edge detection bounds
edge_bounds = [(low, high) for low in np.linspace(0, 255, edge_array) for high in np.linspace(0, 255, edge_array) if
               low < high]

# Load an image
image = cv2.imread('test2.jpg')

# Process the image
process_image(image, luts, edge_bounds)

# Inform the user about the process completion
print("Processed images saved successfully.")
