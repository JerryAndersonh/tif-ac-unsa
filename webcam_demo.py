#!/usr/bin/env python3
"""
Demo de reconocimiento de emociones en tiempo real usando webcam.
Compara los resultados de OpenCV y DeepFace en vivo.

Uso:
    python webcam_demo.py              # Usar ambos modelos
    python webcam_demo.py --opencv     # Solo OpenCV
    python webcam_demo.py --deepface   # Solo DeepFace

Controles:
    q - Salir
    s - Capturar screenshot
    m - Cambiar modo de visualización
"""
import os
import sys
import cv2
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RESULTS_DIR, EMOTIONS


class WebcamEmotionDemo:
    """Demo de reconocimiento de emociones en tiempo real."""

    def __init__(self, use_opencv=True, use_deepface=True):
        """
        Inicializa el demo.

        Args:
            use_opencv: Usar modelo OpenCV
            use_deepface: Usar modelo DeepFace
        """
        self.use_opencv = use_opencv
        self.use_deepface = use_deepface

        # Modelos
        self.opencv_model = None
        self.deepface_model = None

        # Haar cascade para detección de rostros
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Colores para emociones (BGR)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Rojo
            'disgust': (0, 128, 0),    # Verde oscuro
            'fear': (128, 0, 128),     # Púrpura
            'happy': (0, 255, 255),    # Amarillo
            'neutral': (128, 128, 128), # Gris
            'sad': (255, 0, 0),        # Azul
            'surprise': (0, 165, 255)  # Naranja
        }

        # Estado
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Resultados de predicción (para evitar calcular cada frame)
        self.opencv_result = None
        self.deepface_result = None
        self.prediction_interval = 5  # Predecir cada N frames

        self._load_models()

    def _load_models(self):
        """Carga los modelos necesarios."""
        if self.use_opencv:
            print("Cargando modelo OpenCV...")
            try:
                from src.opencv_model import OpenCVEmotionRecognizer
                self.opencv_model = OpenCVEmotionRecognizer()
                if not self.opencv_model.load_model():
                    print("Entrenando modelo OpenCV (esto tomará unos minutos)...")
                    self.opencv_model.train(max_samples_per_class=300)
                print("Modelo OpenCV listo.")
            except Exception as e:
                print(f"Error cargando OpenCV: {e}")
                self.use_opencv = False

        if self.use_deepface:
            print("Cargando modelo DeepFace...")
            try:
                from src.deepface_model import DeepFaceEmotionRecognizer
                self.deepface_model = DeepFaceEmotionRecognizer()
                print("Modelo DeepFace listo.")
            except Exception as e:
                print(f"Error cargando DeepFace: {e}")
                self.use_deepface = False

    def detect_faces(self, frame):
        """
        Detecta rostros en el frame.

        Args:
            frame: Frame BGR de la webcam

        Returns:
            Lista de regiones (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        return faces

    def predict_opencv(self, face_img):
        """Predice emoción usando OpenCV."""
        if self.opencv_model is None:
            return None, None

        try:
            # Convertir a escala de grises si es necesario
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_img

            # Redimensionar a 48x48
            gray = cv2.resize(gray, (48, 48))
            gray = cv2.equalizeHist(gray)

            # Extraer características y predecir
            features = self.opencv_model.extract_features(gray)
            features_scaled = self.opencv_model.scaler.transform(features.reshape(1, -1))
            prediction = self.opencv_model.classifier.predict(features_scaled)[0]
            probabilities = self.opencv_model.classifier.predict_proba(features_scaled)[0]

            emotion = EMOTIONS[prediction]
            probs = dict(zip(EMOTIONS, probabilities))

            return emotion, probs

        except Exception as e:
            return None, None

    def predict_deepface(self, face_img):
        """Predice emoción usando DeepFace."""
        if self.deepface_model is None:
            return None, None

        try:
            from deepface import DeepFace

            # DeepFace necesita imagen en formato BGR o path
            result = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', '').lower()

            # Normalizar probabilidades
            probs = {k.lower(): v/100 for k, v in emotions.items()}

            return dominant, probs

        except Exception as e:
            return None, None

    def draw_results(self, frame, faces):
        """
        Dibuja los resultados en el frame.

        Args:
            frame: Frame a modificar
            faces: Lista de rostros detectados
        """
        for i, (x, y, w, h) in enumerate(faces):
            # Extraer región del rostro
            face_img = frame[y:y+h, x:x+w]

            # Predecir emociones (solo cada N frames)
            if self.frame_count % self.prediction_interval == 0:
                if self.use_opencv:
                    self.opencv_result = self.predict_opencv(face_img)

                if self.use_deepface:
                    self.deepface_result = self.predict_deepface(face_img)

            # Dibujar rectángulo del rostro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Mostrar resultados
            y_offset = y - 10

            if self.use_opencv and self.opencv_result[0]:
                emotion_cv, probs_cv = self.opencv_result
                color = self.emotion_colors.get(emotion_cv, (255, 255, 255))
                text = f"OpenCV: {emotion_cv}"
                cv2.putText(frame, text, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset -= 25

            if self.use_deepface and self.deepface_result[0]:
                emotion_df, probs_df = self.deepface_result
                color = self.emotion_colors.get(emotion_df, (255, 255, 255))
                text = f"DeepFace: {emotion_df}"
                cv2.putText(frame, text, (x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_sidebar(self, frame):
        """
        Dibuja una barra lateral con información.

        Args:
            frame: Frame a modificar
        """
        h, w = frame.shape[:2]

        # Fondo semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-220, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Título
        cv2.putText(frame, "EMOCIONES", (w-200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Mostrar probabilidades si hay resultados
        y_pos = 70

        if self.use_opencv and self.opencv_result and self.opencv_result[1]:
            cv2.putText(frame, "OpenCV:", (w-210, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 20

            for emotion, prob in sorted(self.opencv_result[1].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
                bar_width = int(prob * 150)
                color = self.emotion_colors.get(emotion, (255, 255, 255))

                cv2.rectangle(frame, (w-210, y_pos-10), (w-210+bar_width, y_pos), color, -1)
                text = f"{emotion[:3]}: {prob:.2f}"
                cv2.putText(frame, text, (w-210, y_pos-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                y_pos += 18

        y_pos += 20

        if self.use_deepface and self.deepface_result and self.deepface_result[1]:
            cv2.putText(frame, "DeepFace:", (w-210, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 20

            for emotion, prob in sorted(self.deepface_result[1].items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
                bar_width = int(prob * 150)
                color = self.emotion_colors.get(emotion, (255, 255, 255))

                cv2.rectangle(frame, (w-210, y_pos-10), (w-210+bar_width, y_pos), color, -1)
                text = f"{emotion[:3]}: {prob:.2f}"
                cv2.putText(frame, text, (w-210, y_pos-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                y_pos += 18

        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w-210, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def save_screenshot(self, frame):
        """Guarda una captura de pantalla."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULTS_DIR, f"webcam_capture_{timestamp}.png")
        cv2.imwrite(filename, frame)
        print(f"Screenshot guardado: {filename}")

    def run(self):
        """Ejecuta el demo de webcam."""
        print("\n" + "="*50)
        print("DEMO DE RECONOCIMIENTO DE EMOCIONES - WEBCAM")
        print("="*50)
        print("Controles:")
        print("  q - Salir")
        print("  s - Guardar screenshot")
        print("="*50 + "\n")

        # Abrir webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: No se pudo abrir la webcam")
            print("Asegúrate de tener una webcam conectada")
            return

        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("Webcam iniciada. Presiona 'q' para salir.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error leyendo frame")
                break

            # Voltear horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)

            # Detectar rostros
            faces = self.detect_faces(frame)

            # Dibujar resultados
            frame = self.draw_results(frame, faces)
            frame = self.draw_sidebar(frame)

            # Calcular FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = time.time()
                self.fps = 30 / (current_time - self.last_time)
                self.last_time = current_time

            # Mostrar frame
            cv2.imshow('Emotion Recognition - Press Q to quit', frame)

            # Manejar teclas
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(frame)

        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        print("Demo finalizado.")


def main():
    parser = argparse.ArgumentParser(
        description='Demo de reconocimiento de emociones con webcam'
    )
    parser.add_argument(
        '--opencv', '-o',
        action='store_true',
        help='Usar solo modelo OpenCV'
    )
    parser.add_argument(
        '--deepface', '-d',
        action='store_true',
        help='Usar solo modelo DeepFace'
    )

    args = parser.parse_args()

    # Determinar qué modelos usar
    if args.opencv and not args.deepface:
        use_opencv, use_deepface = True, False
    elif args.deepface and not args.opencv:
        use_opencv, use_deepface = False, True
    else:
        use_opencv, use_deepface = True, True

    # Ejecutar demo
    demo = WebcamEmotionDemo(use_opencv=use_opencv, use_deepface=use_deepface)
    demo.run()


if __name__ == "__main__":
    main()
