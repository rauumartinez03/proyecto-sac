import cv2
import easyocr

# Load the pre-trained EasyOCR model for English
reader = easyocr.Reader(['en'])

# Load the image
half = cv2.imread('matricula2.jpeg')
image = cv2.resize(half, (0, 0), fx = 0.5, fy = 0.5)
# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Otsu's thresholding
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform OCR on the thresholded image
result = reader.readtext(thresh)

# Extract the recognized text from the result
recognized_text = [entry[1] for entry in result]

# Filter out non-alphanumeric characters (assumed to be license plate characters)
license_plate_text = ''.join(filter(str.isalnum, recognized_text))

# Print the recognized license plate text
print("Recognized License Plate:", license_plate_text)

# Display the image with bounding boxes around the detected characters
for detection in result:
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2
    cv2.rectangle(image, top_left, bottom_right, color, thickness)
    cv2.putText(image, text, top_left, font, font_scale, color, thickness, cv2.LINE_AA)

# Display the output image
cv2.imshow('License Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
