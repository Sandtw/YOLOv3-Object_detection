from cv2 import cv2
import numpy as np

# Lectura del modelo DNN

#* Configuración del modelo
config = "model/yolov3.cfg"
#* Pesos
weights = "model/yolov3.weights"
#* Labels
LABELS = open("model/coco.names").read().split("\n")[:-1]

#* Asignación de colores de cajas delimitadores por clase
colors = np.random.randint(0,255,size=(len(LABELS), 3), dtype='uint8')

#* Load model
net = cv2.dnn.readNetFromDarknet(config, weights)

# Lectura de imagenes a preprocesar
image = cv2.imread("images/img1.jpg")
height, width, _ = image.shape

#* Creamos un blob
    #?La entrada a la red es mediante un objeto blob
    #?Opencv lee una imagen por defecto en BGR (RGB: swapRB=True)
    #? crop = False, para cuando la imagen se reforme, no se genere recortes
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB = True, crop = False)
    #? blob es un objeto de matriz numpy 4D (imágenes, canales, ancho, alto)
#! print("blob.shape: ", blob.shape) 

# Detecciones y predicciones
    #?Yolov3 trabaja con 3 escalas para la detección, y también usa por cada cuadrícula 3 anchor

#* Obteniendo los nombres de todas las capas
ln = net.getLayerNames()
#! print("ln: ", ln)

#* Obteniendo las 3 capas de salidas
ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]
print("ln (outs): ", ln)

#* Establecemos el blob como input de la red
net.setInput(blob)
outputs = net.forward(ln)

boxes = []
confidences = []
classIDs = []

for i, output in enumerate(outputs):
    print(f"En la salida {i + 1} se tiene {output.shape[0]} cajas, con un vector de tamaño {output.shape[1]}")
    print(f"BOX de prueba: {output[0]}")
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        
        #? Filtra a todas las cajas con un unmbral de confianza
        if confidence > 0.5:
            box = detection[:4] * np.array([width, height, width, height])
            (x_center, y_center, w, h) = box.astype('int')
            x = int(x_center - (w/2))
            y = int(y_center - (h/2))

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classIDs.append(classID)

idx = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.5)
print(f'Indices de cajas elegidas: {idx}')
if len(idx) > 0:
    for i in idx:
        (x,y) = (boxes[i][0], boxes[i][1])
        (w,h) = (boxes[i][2], boxes[i][3])
        
        color = colors[classIDs[i]].tolist()
        text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
     


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


