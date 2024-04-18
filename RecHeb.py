import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load image
image = cv2.imread('gs2.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Text detection (example using contours)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    text_region = gray[y:y + h, x:x + w]

    # Text recognition using Tesseract
    text = pytesseract.image_to_string(text_region, lang='heb')  # Hebrew language
    print("Detected text:", text)

    # Draw bounding box (optional)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display results
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
