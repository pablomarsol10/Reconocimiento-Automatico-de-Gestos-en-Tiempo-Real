import cv2
from ultralytics import YOLO  
import numpy as np
from fastai.vision.all import *
import torch
import pathlib
import os
import time

# Parche para cargar modelos de Colab (Linux) en Windows
pathlib.PosixPath = pathlib.WindowsPath

MODEL_RESNET_PATH = 'modelo_gestos_resnet18_final.pkl'
YOLO_PATH = 'best.pt'
CONFIDENCE_THRESHOLD = 0.85 # Umbral para confirmar el gesto (ResNet)
CONF_YOLO = 0.5             # Umbral para detectar que es una mano (YOLO)
MARGIN = 0.2
INPUT_SIZE = (224, 224)
WINDOW_NAME = 'Reconocimiento de Gestos (YOLO + ResNet)'

# 1. CARGA DEL MODELO CLASIFICADOR (RESNET)

learn = load_learner(MODEL_RESNET_PATH)
model_resnet = learn.model.eval().cpu()
vocab = learn.dls.vocab

print(f"Clasificador cargado")
print(f"Clases: {vocab}")

# 2. CARGA DEL MODELO DETECTOR (YOLO) 
detector = YOLO(YOLO_PATH)
print(f"Detector cargado")


# 3. PREPROCESAMIENTO
def preprocess_image(img_crop):
    img = cv2.resize(img_crop, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img).permute(2, 0, 1).float()
    tensor = tensor.div(255.0).unsqueeze(0)
    return tensor

# 4. BUCLE PRINCIPAL
cap = cv2.VideoCapture(0)
prev_time = time.time()
frame_counter = 0
fps_display = 0 

print(f"Cierra la ventana con la 'X' o pulsa 'q' para salir.\n")

cv2.namedWindow(WINDOW_NAME)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Espejo
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    h, w, _ = frame.shape
    
    # DETECCIÓN CON YOLO 
    results = detector(frame, verbose=False, conf=CONF_YOLO)

    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # YOLO nos da las coordenadas directas (x1, y1, x2, y2)
            x1_raw, y1_raw, x2_raw, y2_raw = map(int, box.xyxy[0])
            
            # Calculamos ancho y alto de la caja original
            box_w = x2_raw - x1_raw
            box_h = y2_raw - y1_raw
            
            # Ampliamos la caja un 20% (MARGIN) para dar contexto a la ResNet
            x1 = max(0, x1_raw - int(box_w * MARGIN))
            y1 = max(0, y1_raw - int(box_h * MARGIN))
            x2 = min(w, x2_raw + int(box_w * MARGIN))
            y2 = min(h, y2_raw + int(box_h * MARGIN))

            # INFERENCIA 
            if x2 > x1 and y2 > y1:
                hand_crop = frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    try:
                        input_tensor = preprocess_image(hand_crop)
                        
                        with torch.no_grad():
                            output = model_resnet(input_tensor)
                            probs = torch.softmax(output, dim=1)
                            prob_max, idx_max = probs.max(dim=1)
                            
                            gesture = vocab[idx_max.item()]
                            confidence = prob_max.item()

                        # UI RESULTADO 
                        color = (0, 255, 0) 
                        label = f"{gesture.upper()} {confidence:.0%}"
                        
                        # Si la confianza es baja, mostramos advertencia
                        if confidence < CONFIDENCE_THRESHOLD:
                            color = (0, 165, 255) # Naranja
                            label = "..." # 

                        # Dibujamos rectángulo y texto
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Etiqueta con fondo para que se lea bien
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + text_w, y1), color, -1)
                        cv2.putText(display_frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                   
                    except Exception as e:
                        pass # Evitamos crash si el recorte falla

    # CÁLCULO DE FPS 
    curr_time = time.time()
    frame_counter += 1
    if (curr_time - prev_time) > 0.5: 
        fps_display = frame_counter / (curr_time - prev_time)
        prev_time = curr_time
        frame_counter = 0 
    
    cv2.putText(display_frame, f'FPS: {int(fps_display)}', (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, display_frame)

    # CONTROL DE SALIDA 
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print("Ventana cerrada por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()