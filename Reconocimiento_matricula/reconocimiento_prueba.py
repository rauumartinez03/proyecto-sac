import cv2
import easyocr

def reconocer_matricula(imagen):
    # Cargar la imagen con OpenCV
    img = cv2.imread(imagen)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir el ruido
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes en la imagen
    edges = cv2.Canny(gray, 50, 150)
    
    # Encontrar los contornos en la imagen
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordenar los contornos por área y seleccionar el más grande (supuesto matrícula)
    contorno = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Obtener las coordenadas del rectángulo que rodea el contorno
    x, y, w, h = cv2.boundingRect(contorno)
    
    # Recortar la región de interés (ROI) de la imagen
    roi = img[y:y+h, x:x+w]
    
    # Inicializar EasyOCR
    reader = easyocr.Reader(['en'])
    
    # Reconocer el texto en la ROI
    resultado = reader.readtext(roi)
    
    # Extraer el texto reconocido
    matricula = resultado[0][-2]
    
    return matricula

# Ruta de la imagen con la matrícula
imagen = 'matricula_raul.jpg'

# Reconocer matrícula
matricula = reconocer_matricula(imagen)
print("Matrícula reconocida:", matricula)
