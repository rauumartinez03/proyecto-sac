import cv2
import numpy as np
import imutils
import easyocr

def detect_license_plate(image_path):
    # Cargar imagen
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral y Canny edge detection
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Encontrar contornos y seleccionar el más grande
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:
        return None, None

    # Crear una máscara y recortar la región de interés
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Extraer la región de interés (ROI) de la imagen original
    (x, y, w, h) = cv2.boundingRect(location)
    roi = gray[y:y+h, x:x+w]

    return roi, location

def recognize_license_plate(roi):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(roi)
    if result:
        return result[0][-2]
    return None

def draw_license_plate(image, location, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(image, text=text, org=(location[0][0][0], location[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(image, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3)
    return res

def main(image_path):
    roi, location = detect_license_plate(image_path)
    if roi is not None:
        license_plate_text = recognize_license_plate(roi)
        if license_plate_text is not None:
            img = cv2.imread(image_path)
            result_image = draw_license_plate(img, location, license_plate_text)
            cv2.imshow("Result", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("License Plate:", license_plate_text)
        else:
            print("No se pudo reconocer la matrícula.")
    else:
        print("No se encontraron matrículas en la imagen.")

if __name__ == "__main__":
    image_path = "matricula.jpeg"
    main(image_path)
