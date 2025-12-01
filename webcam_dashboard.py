#!/usr/bin/env python3
"""
Dashboard Profesional de Reconocimiento de Emociones en Tiempo Real.
Combina OpenCV y DeepFace con visualización estilo corporativo.

Basado en el diseño visual de dashboard con panel inferior.

Uso:
    python webcam_dashboard.py              # Ambos modelos
    python webcam_dashboard.py --opencv     # Solo OpenCV (más rápido)
    python webcam_dashboard.py --deepface   # Solo DeepFace (más preciso)

Controles:
    q - Salir
    s - Capturar screenshot
    m - Cambiar modelo activo
"""
import os
import sys
import cv2
import numpy as np
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suprimir logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.config import RESULTS_DIR, EMOTIONS

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CAM_INDEX = 0
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
PANEL_HEIGHT = 300

# Rendimiento
ANALYZE_EVERY_N_FRAMES = 3
SMOOTHING_ALPHA = 0.3

# ============================================================================
# COLORES PARA EMOCIONES (BGR)
# ============================================================================

COLORS = {
    "angry": (30, 30, 200),      # Rojo
    "disgust": (40, 180, 40),    # Verde
    "fear": (200, 60, 40),       # Azul oscuro
    "happy": (0, 200, 220),      # Amarillo
    "sad": (200, 160, 40),       # Azul claro
    "surprise": (55, 150, 255),  # Naranja
    "neutral": (180, 180, 180)   # Gris
}

# ============================================================================
# FUNCIONES DE DIBUJO
# ============================================================================

def rounded_rectangle(img, top_left, bottom_right, color, radius=10, thickness=-1):
    """Dibuja un rectángulo con esquinas redondeadas."""
    x1, y1 = top_left
    x2, y2 = bottom_right

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)


class EmotionDashboard:
    """Dashboard profesional de reconocimiento de emociones."""

    def __init__(self, use_opencv=True, use_deepface=True):
        self.use_opencv = use_opencv
        self.use_deepface = use_deepface

        # Modelos
        self.opencv_model = None
        self.deepface_model = None

        # Haar cascade para detección
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Estado de emociones suavizadas
        self.smoothed_opencv = {emo: 0.0 for emo in EMOTIONS}
        self.smoothed_deepface = {emo: 0.0 for emo in EMOTIONS}

        # Emociones dominantes
        self.dominant_opencv = "Detectando..."
        self.dominant_deepface = "Detectando..."

        # Región facial
        self.face_region = None

        # Contadores
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()

        # Cargar modelos
        self._load_models()

    def _load_models(self):
        """Carga los modelos de reconocimiento."""
        if self.use_opencv:
            print("Cargando modelo OpenCV...")
            try:
                from src.opencv_model import OpenCVEmotionRecognizer
                self.opencv_model = OpenCVEmotionRecognizer()
                if not self.opencv_model.load_model():
                    print("  Entrenando modelo OpenCV (espera unos minutos)...")
                    self.opencv_model.train(max_samples_per_class=300)
                print("  ✓ OpenCV listo")
            except Exception as e:
                print(f"  ✗ Error OpenCV: {e}")
                self.use_opencv = False

        if self.use_deepface:
            print("Cargando modelo DeepFace...")
            try:
                from deepface import DeepFace
                # Test rápido para cargar el modelo
                print("  ✓ DeepFace listo")
            except Exception as e:
                print(f"  ✗ Error DeepFace: {e}")
                self.use_deepface = False

    def detect_faces(self, frame):
        """Detecta rostros en el frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces

    def predict_opencv(self, face_img):
        """Predice emoción con OpenCV."""
        if self.opencv_model is None:
            return None, {}

        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img

            gray = cv2.resize(gray, (48, 48))
            gray = cv2.equalizeHist(gray)

            features = self.opencv_model.extract_features(gray)
            features_scaled = self.opencv_model.scaler.transform(features.reshape(1, -1))
            prediction = self.opencv_model.classifier.predict(features_scaled)[0]
            probabilities = self.opencv_model.classifier.predict_proba(features_scaled)[0]

            emotion = EMOTIONS[prediction]
            probs = {emo: prob * 100 for emo, prob in zip(EMOTIONS, probabilities)}

            return emotion, probs
        except:
            return None, {}

    def predict_deepface(self, frame):
        """Predice emoción con DeepFace."""
        try:
            from deepface import DeepFace

            # Reducir tamaño para análisis más rápido
            analysis_frame = cv2.resize(frame, (640, 480))

            result = DeepFace.analyze(
                analysis_frame,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            dominant = result.get('dominant_emotion', 'neutral').lower()
            emotions = result.get('emotion', {})

            # Convertir a diccionario con nombres en minúsculas
            probs = {k.lower(): float(v) for k, v in emotions.items()}

            # Obtener región facial
            region = result.get('region', None)
            if region and region.get('w', 0) > 30:
                self.face_region = region

            return dominant, probs
        except:
            return None, {}

    def update_smoothed(self, new_probs, smoothed_dict):
        """Actualiza probabilidades con suavizado."""
        for emo in EMOTIONS:
            if emo in new_probs:
                value = float(new_probs[emo])
                smoothed_dict[emo] = SMOOTHING_ALPHA * value + (1 - SMOOTHING_ALPHA) * smoothed_dict[emo]

    def draw_face_rectangle(self, frame):
        """Dibuja rectángulo alrededor del rostro."""
        if self.face_region and self.face_region.get('w', 0) > 0:
            # Escalar coordenadas al tamaño del frame
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480

            x = int(self.face_region.get('x', 0) * scale_x) - 8
            y = int(self.face_region.get('y', 0) * scale_y) - 8
            w = int(self.face_region.get('w', 0) * scale_x) + 16
            h = int(self.face_region.get('h', 0) * scale_y) + 16

            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            rounded_rectangle(frame, (x, y), (x2, y2), (60, 180, 255), radius=12, thickness=3)

    def draw_video_overlay(self, frame):
        """Dibuja información sobre el video."""
        # Emoción dominante
        if self.use_deepface:
            text = f"DeepFace: {self.dominant_deepface.capitalize()}"
            color = COLORS.get(self.dominant_deepface, (255, 255, 255))
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        if self.use_opencv:
            text = f"OpenCV: {self.dominant_opencv.capitalize()}"
            color = COLORS.get(self.dominant_opencv, (255, 255, 255))
            y_pos = 80 if self.use_deepface else 40
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (frame.shape[1] - 140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    def create_panel(self):
        """Crea el panel inferior con barras de emociones."""
        panel = np.zeros((PANEL_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)

        # Degradado de fondo
        for i in range(PANEL_HEIGHT):
            alpha = 0.85 + 0.15 * (i / PANEL_HEIGHT)
            panel[i] = (int(20 * alpha), int(30 * alpha), int(40 * alpha))

        # Configuración de barras
        bar_max_width = 400
        bar_height = 22
        y_start = 50
        y_spacing = 32

        # === PANEL OPENCV (izquierda) ===
        if self.use_opencv:
            # Header OpenCV
            rounded_rectangle(panel, (20, 10), (350, 45), (20, 60, 100), radius=8)
            cv2.putText(panel, "OpenCV (Machine Learning)",
                        (35, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

            for idx, emo in enumerate(EMOTIONS):
                y_pos = y_start + idx * y_spacing
                value = self.smoothed_opencv[emo]
                value_clamped = max(0.0, min(100.0, value))
                bar_width = int((value_clamped / 100.0) * bar_max_width)

                # Etiqueta
                cv2.putText(panel, emo.capitalize(), (30, y_pos + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                # Barra de fondo
                bg_x = 140
                cv2.rectangle(panel, (bg_x, y_pos), (bg_x + bar_max_width, y_pos + bar_height),
                             (40, 50, 70), -1)

                # Barra de valor
                if bar_width > 0:
                    cv2.rectangle(panel, (bg_x, y_pos), (bg_x + bar_width, y_pos + bar_height),
                                 COLORS[emo], -1)

                # Porcentaje
                cv2.putText(panel, f"{value_clamped:5.1f}%", (bg_x + bar_max_width + 10, y_pos + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # === PANEL DEEPFACE (derecha) ===
        if self.use_deepface:
            offset_x = 640 if self.use_opencv else 20

            # Header DeepFace
            rounded_rectangle(panel, (offset_x, 10), (offset_x + 380, 45), (60, 20, 100), radius=8)
            cv2.putText(panel, "DeepFace (Deep Learning)",
                        (offset_x + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 200), 2)

            for idx, emo in enumerate(EMOTIONS):
                y_pos = y_start + idx * y_spacing
                value = self.smoothed_deepface[emo]
                value_clamped = max(0.0, min(100.0, value))
                bar_width = int((value_clamped / 100.0) * bar_max_width)

                # Etiqueta
                cv2.putText(panel, emo.capitalize(), (offset_x + 10, y_pos + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

                # Barra de fondo
                bg_x = offset_x + 120
                cv2.rectangle(panel, (bg_x, y_pos), (bg_x + bar_max_width, y_pos + bar_height),
                             (40, 50, 70), -1)

                # Barra de valor
                if bar_width > 0:
                    cv2.rectangle(panel, (bg_x, y_pos), (bg_x + bar_width, y_pos + bar_height),
                                 COLORS[emo], -1)

                # Porcentaje
                cv2.putText(panel, f"{value_clamped:5.1f}%", (bg_x + bar_max_width + 10, y_pos + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # Footer
        footer_text = "Q=Salir | S=Screenshot | OpenCV vs DeepFace - Comparacion en Tiempo Real"
        cv2.putText(panel, footer_text, (20, PANEL_HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return panel

    def save_screenshot(self, combined_frame):
        """Guarda una captura de pantalla."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"dashboard_capture_{timestamp}.png")
        cv2.imwrite(filename, combined_frame)
        print(f"Screenshot guardado: {filename}")

    def run(self):
        """Ejecuta el dashboard."""
        print("\n" + "="*70)
        print("  DASHBOARD DE EMOCIONES - OpenCV vs DeepFace")
        print("="*70)
        print("  Controles:")
        print("    Q - Salir")
        print("    S - Guardar screenshot")
        print("="*70 + "\n")

        # Abrir cámara
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT - PANEL_HEIGHT)

        if not cap.isOpened():
            print("Error: No se pudo abrir la webcam")
            return

        print("Webcam iniciada. Presiona 'Q' para salir.\n")

        while True:
            t0 = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Redimensionar y voltear
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT - PANEL_HEIGHT))
            frame = cv2.flip(frame, 1)

            # Analizar cada N frames
            if self.frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                # OpenCV
                if self.use_opencv:
                    faces = self.detect_faces(frame)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_img = frame[y:y+h, x:x+w]
                        emotion, probs = self.predict_opencv(face_img)
                        if emotion:
                            self.dominant_opencv = emotion
                            self.update_smoothed(probs, self.smoothed_opencv)

                # DeepFace
                if self.use_deepface:
                    emotion, probs = self.predict_deepface(frame)
                    if emotion:
                        self.dominant_deepface = emotion
                        self.update_smoothed(probs, self.smoothed_deepface)

            # Dibujar
            self.draw_face_rectangle(frame)
            self.draw_video_overlay(frame)

            # Crear panel
            panel = self.create_panel()

            # Combinar video + panel
            combined = np.vstack([frame, panel])

            # Mostrar
            cv2.imshow("Dashboard de Emociones - OpenCV vs DeepFace", combined)

            # Controles
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(combined)

            # Calcular FPS
            self.frame_count += 1
            now = time.time()
            if (now - t0) > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - t0))

        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Dashboard cerrado correctamente")


def main():
    parser = argparse.ArgumentParser(description='Dashboard de emociones en tiempo real')
    parser.add_argument('--opencv', '-o', action='store_true', help='Solo usar OpenCV')
    parser.add_argument('--deepface', '-d', action='store_true', help='Solo usar DeepFace')

    args = parser.parse_args()

    if args.opencv and not args.deepface:
        use_opencv, use_deepface = True, False
    elif args.deepface and not args.opencv:
        use_opencv, use_deepface = False, True
    else:
        use_opencv, use_deepface = True, True

    dashboard = EmotionDashboard(use_opencv=use_opencv, use_deepface=use_deepface)
    dashboard.run()


if __name__ == "__main__":
    main()
