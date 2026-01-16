import cv2
import mediapipe as mp
import numpy as np
from fastai.vision.all import *
import torch
import pathlib
import os
import time

pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = 'modelo_gestos_resnet18_final.pkl'
CONFIDENCE_THRESHOLD = 0.85
MARGIN = 0.2
INPUT_SIZE = (224, 224)
WINDOW_NAME = 'Proyecto Final - Reconocimiento de Gestos' 

# 1. CARGA DEL MODELO
learn = load_learner(MODEL_PATH)
model = learn.model.eval().cpu()
vocab = learn.dls.vocab

print(f"Cerebro cargado: ResNet18")
print(f"Clases: {vocab}")

# 2. INICIALIZACIÓN DE MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
fps_display = 0  # El valor que se mostrará en pantalla

print(f"\nSISTEMA EN LÍNEA.")
print(f"   Cierra la ventana con la 'X' para salir.\n")

cv2.namedWindow(WINDOW_NAME)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Espejo
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    h, w, _ = frame.shape
    
    # Detección
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Coordenadas
            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            box_w = int((x_max - x_min) * w)
            box_h = int((y_max - y_min) * h)
            
            x1 = max(0, int(x_min * w) - int(box_w * MARGIN))
            y1 = max(0, int(y_min * h) - int(box_h * MARGIN))
            x2 = min(w, int(x_max * w) + int(box_w * MARGIN))
            y2 = min(h, int(y_max * h) + int(box_h * MARGIN))

            # Inferencia
            if x2 > x1 and y2 > y1:
                hand_crop = frame[y1:y2, x1:x2]
                
                if hand_crop.size > 0:
                    input_tensor = preprocess_image(hand_crop)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        probs = torch.softmax(output, dim=1)
                        prob_max, idx_max = probs.max(dim=1)
                        
                        gesture = vocab[idx_max.item()]
                        confidence = prob_max.item()

                    # UI Resultado
                    color = (0, 255, 0) 
                    label = f"{gesture.upper()} {confidence:.0%}"
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        color = (0, 165, 255) 
                        label = f"..."

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(display_frame, (x1, y1 - 30), (x2, y1), color, -1)
                    cv2.putText(display_frame, label, (x1 + 5, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # CÁLCULO DE FPS (Actualiza cada 0.5 seg) 
    curr_time = time.time()
    frame_counter += 1
    if (curr_time - prev_time) > 0.5: # Si ha pasado medio segundo
        fps_display = frame_counter / (curr_time - prev_time)
        prev_time = curr_time
        frame_counter = 0 # Reseteamos contador
    
    # Dibujar FPS estables
    cv2.putText(display_frame, f'FPS: {int(fps_display)}', (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar imagen
    cv2.imshow(WINDOW_NAME, display_frame)

    # CONTROL DE SALIDA 
    # 1. Necesitamos waitKey(1) para que la ventana refresque
    cv2.waitKey(1)
    
    # 2. Comprobamos si la ventana se ha cerrado
    # WND_PROP_VISIBLE devuelve 0 si la ventana se cerró
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print("Ventana cerrada por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()