# 🖐️ Real-Time Hand Gesture Recognition for 3D Interfaces (HCI)

Este proyecto desarrolla un sistema avanzado de **Interacción Persona-Ordenador (HCI)** capaz de reconocer gestos manuales en tiempo real utilizando una cámara RGB estándar y técnicas de **Deep Learning**. El sistema ha sido optimizado para ejecutarse en hardware de consumo (CPU), alcanzando un equilibrio crítico entre precisión y latencia.

---

## 📊 Dataset: HaGRID (HAnd Gesture Recognition Image Dataset)
Para garantizar la robustez en entornos reales, hemos utilizado el dataset **HaGRID**, capturado en condiciones no controladas.
* **Gestos Seleccionados**: *call, fist, like, ok, palm, peace, rock* y *stop*.
* **Volumen de Datos**: Procesamiento de **28.000 imágenes** (3.500 por clase) segregadas en conjuntos de entrenamiento y validación.
* **Preprocesamiento**: Redimensionado a 224x224 píxeles y normalización basada en las estadísticas de ImageNet.

---

## 🔬 Ciclo de Experimentación y Comparativas Técnicas

### 1️⃣ Visión Clásica vs. Deep Learning
Evaluamos la viabilidad de un enfoque tradicional frente a redes neuronales.
* **Template Matching**: Implementación basada en el coeficiente de **Correlación Cruzada Normalizada (NCC)** con una base de 24 plantillas manuales.
* **Resultados**: El método clásico presentó fallos críticos debido a la **ambigüedad morfológica** (confundiendo 'Rock' con 'Call' con un 66% de confianza errónea) y la falta de invarianza a la escala.
* **Conclusión**: Se validó la necesidad de utilizar **Redes Neuronales Convolucionales (CNN)** capaces de aprender características jerárquicas robustas que ignoran el ruido de los píxeles brutos.

### 2️⃣ Impacto del Data Augmentation (Regularización)
Para dotar al sistema de invarianza frente a la variabilidad del mundo real, aplicamos técnicas de aumento de datos sintéticos.
* **Transformaciones**: Rotaciones de hasta 15°, zoom aleatorio, ajustes de brillo/contraste y deformaciones de perspectiva.
* **Resultados**: En el modelo **ResNet18**, la precisión aumentó del **98.57%** al **99.14%**. 
* **Conclusión**: El Data Augmentation actúa como un regularizador efectivo, evitando el sobreajuste y permitiendo que la red generalice correctamente ante diferentes usuarios y fondos.

### 3️⃣ Recorte de ROI vs. Imágenes Completas
Analizamos si el modelo podía localizar el gesto de forma implícita o si requería una segmentación previa de la mano.
* **Resultados**: Al entrenar con imágenes completas (sin recortar), la precisión de ResNet18 cayó drásticamente al **83.21%**. El ruido visual del fondo confunde a los modelos ligeros.
* **Conclusión**: Para garantizar alta precisión y fluidez en CPU, es **obligatorio** un diseño modular: primero detectar la mano para "limpiar" la imagen y luego clasificar el recorte.

### 4️⃣ Detector Propio (YOLOv8) vs. MediaPipe
Desarrollamos un detector específico para sustituir dependencias externas y comparar rendimiento.
* **YOLOv8 Nano (Propio)**: Alcanzó un **mAP50 de 0.995** con una latencia de **21.69 ms** en CPU.
* **MediaPipe**: Solución basada en landmarks con una latencia de **26.90 ms**.
* **Decisión Técnica**: Aunque YOLOv8 fue ligeramente más rápido, seleccionamos **MediaPipe** para la interfaz final por su estabilidad superior ante rotaciones bruscas de la mano, asegurando un tracking mucho más natural.

---

## 🏆 Arquitectura Ganadora
El sistema final implementa un pipeline modular optimizado para **tiempo real**:
1.  **Localización**: MediaPipe (Tracking de alta estabilidad).
2.  **Segmentación**: Extracción automática de la Región de Interés (ROI) con margen de seguridad.
3.  **Clasificación**: **ResNet18** (Fine-tuned con Data Augmentation).

**Rendimiento Final**: Precisión del **99.14%** a una velocidad de **72.6 FPS** en CPU estándar.

---

## 🛠️ Stack Tecnológico
* **Lenguaje**: Python
* **Deep Learning**: PyTorch y FastAI
* **Visión por Computador**: OpenCV, MediaPipe y YOLOv8 (Ultralytics)
* **Infraestructura**: Google Colab para entrenamiento acelerado por GPU

---

## 📂 Contenido del Repositorio
* `ProyectoFinal.ipynb`: Cuaderno con el pipeline de ingeniería de datos, entrenamiento y validación.
* `Memoria_Tecnica.pdf`: Documentación detallada con el marco teórico y análisis científico.
* `video_demo.mp4`: Demostración del sistema en funcionamiento real.

---

## 👥 Autores
* **Juan Carlos Mora**
* **Alejandro López Domínguez**
* **Pablo Martín Soler**
