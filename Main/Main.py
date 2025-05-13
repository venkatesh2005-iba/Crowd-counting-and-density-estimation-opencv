import cv2
import numpy as np

# Load image
image_path = "1.jpg"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.resize(image, (800, 600))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Threshold the image to binarythresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# Morphological operations to remove noise and fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours (each likely represents a person blob)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Count people-like blobs
count = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # Filter small blobs
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1

# Estimate density as ratio of non-zero pixels
non_zero = cv2.countNonZero(morph)
total_pixels = morph.shape[0] * morph.shape[1]
density = non_zero / total_pixels

# Display results
cv2.putText(image, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
cv2.putText(image, f'Density: {density:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow("Crowd Detection", image)
cv2.imshow("Processed Mask", morph)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, 