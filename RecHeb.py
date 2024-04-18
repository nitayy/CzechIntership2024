import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Load the image
image = cv2.imread('gs1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Display the result
cv2.imshow('Black and White Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()