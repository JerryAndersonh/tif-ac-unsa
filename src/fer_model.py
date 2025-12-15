"""
Modelo de reconocimiento facial de emociones usando FER (Facial Expression Recognition).
Utiliza el modelo preentrenado Mini-Xception, una CNN ligera y eficiente.

Este modelo viene preentrenado, similar a DeepFace, para una comparación justa.
"""
import os
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Intentar diferentes formas de importar FER
try:
    from fer import FER
except ImportError:
    try:
        from fer.fer import FER
    except ImportError:
        try:
            import fer
            FER = fer.FER
        except (ImportError, AttributeError):
            raise ImportError(
                "No se pudo importar FER. Por favor instala la librería:\n"
                "  pip install fer\n"
                "O si hay problemas de compatibilidad:\n"
                "  pip install fer==22.4.0\n"
                "  pip install tensorflow==2.13.0"
            )

from .config import EMOTIONS, TEST_DIR


class FEREmotionRecognizer:
    """
    Reconocedor de emociones faciales basado en FER (Mini-Xception).
    Modelo preentrenado - no requiere entrenamiento con FER2013.
    """

    def __init__(self, mtcnn=False):
        """
        Inicializa el reconocedor FER.

        Args:
            mtcnn: Si usar MTCNN para detección de rostros (más preciso pero más lento)
        """
        self.detector = FER(mtcnn=mtcnn)
        self.model_name = "FER (Mini-Xception)"

        # Mapeo de emociones FER a FER2013
        self.emotion_map = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprise',
            'neutral': 'neutral'
        }

        print(f"FER (Mini-Xception) inicializado - Modelo preentrenado listo")

    def predict(self, image_path):
        """
        Predice la emoción de una imagen facial.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Tupla (emoción_predicha, probabilidades)
        """
        try:
            import cv2

            # Leer imagen
            image = cv2.imread(image_path)
            if image is None:
                return None, None

            # Detectar emociones
            result = self.detector.detect_emotions(image)

            if not result or len(result) == 0:
                # Si no detecta rostro, intentar con la imagen completa
                # Redimensionar a un tamaño razonable
                image = cv2.resize(image, (48, 48))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                result = self.detector.detect_emotions(image)

                if not result or len(result) == 0:
                    return None, None

            # Obtener el primer rostro detectado
            emotions = result[0]['emotions']

            # Encontrar emoción dominante
            dominant_emotion = max(emotions, key=emotions.get)

            # Mapear emociones
            mapped_emotions = {}
            for emo, score in emotions.items():
                if emo in self.emotion_map:
                    mapped_emotions[self.emotion_map[emo]] = score

            return dominant_emotion, mapped_emotions

        except Exception as e:
            return None, None

    def predict_from_frame(self, frame):
        """
        Predice la emoción desde un frame de video (numpy array).

        Args:
            frame: Imagen BGR como numpy array

        Returns:
            Tupla (emoción_predicha, probabilidades, región_facial)
        """
        try:
            result = self.detector.detect_emotions(frame)

            if not result or len(result) == 0:
                return None, None, None

            # Primer rostro
            face_data = result[0]
            emotions = face_data['emotions']
            box = face_data['box']  # (x, y, w, h)

            dominant_emotion = max(emotions, key=emotions.get)

            # Convertir probabilidades a porcentaje
            probs = {k: v * 100 for k, v in emotions.items()}

            return dominant_emotion, probs, box

        except Exception as e:
            return None, None, None

    def predict_batch(self, image_paths, show_progress=True):
        """
        Predice emociones para múltiples imágenes.

        Args:
            image_paths: Lista de rutas a imágenes
            show_progress: Mostrar barra de progreso

        Returns:
            Lista de predicciones
        """
        predictions = []

        iterator = tqdm(image_paths, desc="FER prediciendo") if show_progress else image_paths

        for path in iterator:
            emotion, probs = self.predict(path)
            predictions.append({
                'path': path,
                'prediction': emotion,
                'probabilities': probs
            })

        return predictions


def test_fer_model():
    """Función de prueba del modelo FER."""
    model = FEREmotionRecognizer()

    print("\nPruebas de predicción FER:")

    for emotion in ['happy', 'sad', 'angry']:
        test_dir = os.path.join(TEST_DIR, emotion)
        if os.path.exists(test_dir):
            test_images = os.listdir(test_dir)[:3]

            print(f"\nProbando emoción: {emotion}")
            for img in test_images:
                img_path = os.path.join(test_dir, img)
                pred, probs = model.predict(img_path)
                print(f"  Imagen: {img}")
                print(f"  Real: {emotion}, Predicho: {pred}")
                if probs:
                    top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"  Top 3: {top_3}")
                print()


if __name__ == "__main__":
    test_fer_model()
