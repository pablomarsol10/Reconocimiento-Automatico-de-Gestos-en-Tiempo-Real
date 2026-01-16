import cv2
import numpy as np
from fastai.vision.all import *
import torch
import pathlib
import os
import time



pathlib.PosixPath = pathlib.WindowsPath
# Usamos el modelo entrenado con imágenes completas (SIN RECORTE)
MODEL_PATH = 'modelo_gestos_nocrop.pkl' 
CONFIDENCE_THRESHOLD = 0.85
INPUT_SIZE = (224, 224)
WINDOW_NAME = 'Inferencia sin detector'

# 1. CARGA DEL MODELO

learn = load_learner(MODEL_PATH)
model = learn.model.eval().cpu()
vocab = learn.dls.vocab
print(f"Cerebro cargado: {MODEL_PATH}")
print(f"Clases detectables: {vocab}")


# 2. PREPROCESAMIENTO
def preprocess_image(img_full):
    # Redimensionamos la imagen completa a 224x224
    img = cv2.resize(img_full, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img).permute(2, 0, 1).float()
    tensor = tensor.div(255.0).unsqueeze(0)
    return tensor

# 3. BUCLE PRINCIPAL
cap = cv2.VideoCapture(0)

# Variables para FPS
prev_time = time.time()
frame_counter = 0
fps_display = 0 

print(f"\nCÁMARA ACTIVA.")

cv2.namedWindow(WINDOW_NAME)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Espejo
    frame = cv2.flip(frame, 1)
    
    # Copia para dibujar la interfaz
    display_frame = frame.copy()
    h, w, _ = frame.shape

    # INFERENCIA DIRECTA (Toda la imagen)
    # Aquí pasamos el frame entero. El modelo "mira" todo:
    input_tensor = preprocess_image(frame)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prob_max, idx_max = probs.max(dim=1)
        
        gesture = vocab[idx_max.item()]
        confidence = prob_max.item()

    # INTERFAZ GRÁFICA
    # Definimos colores según confianza
    if confidence > CONFIDENCE_THRESHOLD:
        color_status = (0, 255, 0) # Verde (Seguro)
        text_status = "DETECTADO"
    else:
        color_status = (0, 165, 255) # Naranja (Dudoso)
        text_status = "ANALIZANDO..."

    # 1. Panel de fondo (Caja negra semitransparente arriba a la izquierda)
    # Coordenadas: (x1, y1) -> (x2, y2)
    cv2.rectangle(display_frame, (10, 10), (350, 110), (0, 0, 0), -1)
    
    # 2. Borde de color según estado
    cv2.rectangle(display_frame, (10, 10), (350, 110), color_status, 2)

    # 3. Textos
    # Gesto detectado 
    cv2.putText(display_frame, f"GESTO: {gesture.upper()}", (30, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Confianza (Barra de progreso visual)
    bar_width = int(confidence * 300) # Largo de la barra basado en %
    cv2.rectangle(display_frame, (25, 65), (25 + bar_width, 75), color_status, -1)
    
    cv2.putText(display_frame, f"Confianza: {confidence:.1%}", (30, 95), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # FPS
    curr_time = time.time()
    frame_counter += 1
    if (curr_time - prev_time) > 0.5: 
        fps_display = frame_counter / (curr_time - prev_time)
        prev_time = curr_time
        frame_counter = 0 
    
    # Mostramos FPS en la esquina opuesta (Derecha)
    cv2.putText(display_frame, f'FPS: {int(fps_display)}', (w - 150, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar resultado
    cv2.imshow(WINDOW_NAME, display_frame)

    cv2.waitKey(1)
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()