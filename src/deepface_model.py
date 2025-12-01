"""
Modelo de reconocimiento facial de emociones usando DeepFace.
Utiliza redes neuronales convolucionales preentrenadas para
detección y clasificación de emociones faciales.
"""
import os
import numpy as np
from tqdm import tqdm
import warnings

# Suprimir warnings de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from deepface import DeepFace
from .config import EMOTIONS, DEEPFACE_TO_FER2013, TEST_DIR


class DeepFaceEmotionRecognizer:
    """
    Reconocedor de emociones faciales basado en DeepFace.
    Utiliza modelos preentrenados de deep learning para
    análisis de emociones faciales.
    """

    def __init__(self, model_name='VGG-Face', detector_backend='opencv'):
        """
        Inicializa el reconocedor DeepFace.

        Args:
            model_name: Modelo de reconocimiento facial a usar
                       Opciones: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'ArcFace'
            detector_backend: Backend para detección de rostros
                             Opciones: 'opencv', 'ssd', 'mtcnn', 'retinaface', 'mediapipe'
        """
        self.model_name = model_name
        self.detector_backend = detector_backend

        # Mapeo de emociones de DeepFace a FER2013
        self.emotion_map = DEEPFACE_TO_FER2013

        print(f"DeepFace inicializado con detector: {detector_backend}")

    def predict(self, image_path, enforce_detection=False):
        """
        Predice la emoción de una imagen facial usando DeepFace.

        Args:
            image_path: Ruta a la imagen
            enforce_detection: Si es True, lanza error si no detecta rostro

        Returns:
            Tupla (emoción_predicha, probabilidades)
        """
        try:
            # Analizar imagen con DeepFace
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                enforce_detection=enforce_detection,
                detector_backend=self.detector_backend,
                silent=True
            )

            # DeepFace puede devolver lista o diccionario
            if isinstance(result, list):
                result = result[0]

            # Obtener emociones
            emotions = result.get('emotion', {})
            dominant_emotion = result.get('dominant_emotion', None)

            # Mapear a nombres de FER2013 (convertir a minúsculas)
            mapped_emotions = {}
            for emotion, score in emotions.items():
                emotion_lower = emotion.lower()
                if emotion_lower in self.emotion_map:
                    mapped_emotions[self.emotion_map[emotion_lower]] = score / 100.0

            # Normalizar probabilidades para que sumen 1
            total = sum(mapped_emotions.values())
            if total > 0:
                mapped_emotions = {k: v/total for k, v in mapped_emotions.items()}

            # Obtener predicción dominante
            if dominant_emotion:
                dominant_emotion = dominant_emotion.lower()
                if dominant_emotion in self.emotion_map:
                    dominant_emotion = self.emotion_map[dominant_emotion]

            return dominant_emotion, mapped_emotions

        except Exception as e:
            # Si falla la detección, intentar sin detección forzada
            if enforce_detection:
                return self.predict(image_path, enforce_detection=False)
            return None, None

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

        iterator = tqdm(image_paths, desc="DeepFace prediciendo") if show_progress else image_paths

        for path in iterator:
            emotion, probs = self.predict(path)
            predictions.append({
                'path': path,
                'prediction': emotion,
                'probabilities': probs
            })

        return predictions

    def analyze_image_detailed(self, image_path):
        """
        Realiza un análisis detallado de una imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Diccionario con análisis completo
        """
        try:
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion', 'age', 'gender', 'race'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True
            )

            if isinstance(result, list):
                result = result[0]

            return {
                'emotion': result.get('dominant_emotion', 'unknown'),
                'emotion_scores': result.get('emotion', {}),
                'age': result.get('age', None),
                'gender': result.get('dominant_gender', None),
                'gender_scores': result.get('gender', {}),
                'race': result.get('dominant_race', None),
                'race_scores': result.get('race', {}),
                'region': result.get('region', {})
            }

        except Exception as e:
            return {'error': str(e)}


def test_deepface_model():
    """Función de prueba del modelo DeepFace."""
    model = DeepFaceEmotionRecognizer()

    # Probar con algunas imágenes de test
    print("\nPruebas de predicción DeepFace:")

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
                    print(f"  Top 3 probabilidades: {top_3}")
                print()


if __name__ == "__main__":
    test_deepface_model()
