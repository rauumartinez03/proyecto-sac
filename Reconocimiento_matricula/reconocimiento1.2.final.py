import cv2
import easyocr

# Load the pre-trained EasyOCR model for English
reader = easyocr.Reader(['en'])

# Load the image
image = cv2.imread('marin.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform OCR on the grayscale image
result = reader.readtext(gray)

# Extract the recognized text from the result
recognized_text = ''

for detection in result:
    text = detection[1]
    recognized_text += text + ' '

# Print the recognized license plate text
print("Recognized License Plate:", recognized_text)

# Display the image with bounding boxes around the detected characters
for detection in result:
    # Extract coordinates of the bounding box
    top_left = tuple(map(int, detection[0][0]))
    bottom_right = tuple(map(int, detection[0][2]))
    text = detection[1]
    
    # Draw bounding box and text
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Display the output image
cv2.imshow('License Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
