"""
Configuración del proyecto de reconocimiento facial de emociones.
"""
import os

# Rutas base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'fer2013')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Rutas de datos
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Emociones del dataset FER2013
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Mapeo de emociones DeepFace a FER2013
DEEPFACE_TO_FER2013 = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'surprise': 'surprise'
}

# Parámetros de imagen
IMG_SIZE = 48  # FER2013 usa imágenes de 48x48
IMG_SIZE_DEEPFACE = 224  # DeepFace requiere imágenes más grandes

# Parámetros de muestreo para pruebas
SAMPLE_SIZE_PER_EMOTION = 100  # Número de imágenes por emoción para pruebas
RANDOM_SEED = 42

# Crear directorios si no existen
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
