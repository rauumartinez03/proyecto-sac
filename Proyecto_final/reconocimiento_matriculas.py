import cv2
import easyocr

# Cargar el modelo preentrenado de EasyOCR para inglés
reader = easyocr.Reader(['en'])

# Cargar la imagen
image = cv2.imread("matricula.jpeg")

# Convertir la imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Realizar OCR en la imagen en escala de grises
resultado = reader.readtext(gray)

# Extraer el texto reconocido del resultado
recognized_text = ''

for detection in resultado:
    texto = detection[1]
    recognized_text += texto + ' '

# Imprimir el texto reconocido de la matrícula
print("Matricula reconocida:", recognized_text)

# Mostrar la imagen con cuadros delimitadores alrededor de los caracteres detectados
for detection in resultado:
    # Extraer las coordenadas del cuadro delimitador
    top_izq = tuple(map(int, detection[0][0]))
    bottom_derecha = tuple(map(int, detection[0][2]))
    texto = detection[1]
    
    # Dibujar el cuadro delimitador y el texto
    cv2.rectangle(image, top_izq, bottom_derecha, (255, 0, 0), 2)
    cv2.putText(image, texto, (top_izq[0], top_izq[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Mostrar la imagen de salida
cv2.imshow('Matricula reconocida', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
