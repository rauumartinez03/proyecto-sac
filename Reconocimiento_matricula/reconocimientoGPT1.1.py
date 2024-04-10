import cv2
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Load the image
image = cv2.imread('cochecitolere.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize OpenCV's CascadeClassifier for license plate detection
license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Perform license plate detection
license_plates = license_plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

# Loop through detected license plates and perform OCR
for (x, y, w, h) in license_plates:
    # Extract the region of interest (ROI) containing the license plate
    plate_roi = gray[y:y+h, x:x+w]
    
    # Perform OCR on the license plate ROI
    result = reader.readtext(plate_roi)
    
    # Extract the recognized text from the result
    plate_text = result[0][1] if result else "No text detected"
    
    # Print the recognized text
    print("License Plate Text:", plate_text)

    # Draw rectangles around the detected license plates
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output image with detected license plates
cv2.imshow('License Plate Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
