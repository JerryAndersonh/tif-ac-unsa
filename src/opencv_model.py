"""
Modelo de reconocimiento facial de emociones usando OpenCV.
Utiliza Haar Cascade para detección de rostros y un clasificador
entrenado con el dataset FER2013.
"""
import cv2
import numpy as np
import os
from tqdm import tqdm
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .config import EMOTIONS, TRAIN_DIR, TEST_DIR, MODELS_DIR, IMG_SIZE


class OpenCVEmotionRecognizer:
    """
    Reconocedor de emociones faciales basado en OpenCV.
    Utiliza Haar Cascade para detección de rostros y características HOG/LBP
    con un clasificador SVM para reconocimiento de emociones.
    """

    def __init__(self):
        # Cargar clasificador Haar Cascade para detección de rostros
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Clasificador y scaler
        self.classifier = None
        self.scaler = None
        self.is_trained = False

        # Rutas de modelos guardados
        self.model_path = os.path.join(MODELS_DIR, 'opencv_emotion_model.pkl')
        self.scaler_path = os.path.join(MODELS_DIR, 'opencv_scaler.pkl')

    def extract_features(self, image):
        """
        Extrae características de una imagen facial usando HOG y LBP.

        Args:
            image: Imagen en escala de grises

        Returns:
            Vector de características
        """
        # Redimensionar a tamaño estándar
        if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        features = []

        # 1. Histograma de Gradientes Orientados (HOG)
        win_size = (IMG_SIZE, IMG_SIZE)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        hog_features = hog.compute(image)
        if hog_features is not None:
            features.extend(hog_features.flatten())

        # 2. Local Binary Patterns (LBP) simplificado
        lbp_hist = self._compute_lbp(image)
        features.extend(lbp_hist)

        # 3. Estadísticas básicas de intensidad
        features.append(np.mean(image))
        features.append(np.std(image))
        features.append(np.median(image))

        # 4. División en regiones y estadísticas por región
        h, w = image.shape
        regions = [
            image[0:h//2, 0:w//2],      # Superior izquierda
            image[0:h//2, w//2:w],      # Superior derecha
            image[h//2:h, 0:w//2],      # Inferior izquierda
            image[h//2:h, w//2:w]       # Inferior derecha
        ]

        for region in regions:
            features.append(np.mean(region))
            features.append(np.std(region))

        return np.array(features, dtype=np.float32)

    def _compute_lbp(self, image, radius=1, n_points=8):
        """
        Calcula el histograma LBP de una imagen.

        Args:
            image: Imagen en escala de grises
            radius: Radio del patrón LBP
            n_points: Número de puntos vecinos

        Returns:
            Histograma LBP normalizado
        """
        h, w = image.shape
        lbp_image = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_string = 0

                # Calcular patrón LBP
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(round(i + radius * np.sin(angle)))
                    y = int(round(j + radius * np.cos(angle)))

                    if image[x, y] >= center:
                        binary_string |= (1 << k)

                lbp_image[i - radius, j - radius] = binary_string

        # Calcular histograma
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)  # Normalizar

        return hist

    def detect_face(self, image):
        """
        Detecta rostros en una imagen usando Haar Cascade.

        Args:
            image: Imagen BGR o escala de grises

        Returns:
            Lista de regiones de rostros detectados (x, y, w, h)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        return faces

    def preprocess_image(self, image_path):
        """
        Preprocesa una imagen para el reconocimiento.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Imagen preprocesada en escala de grises
        """
        # Leer imagen
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return None

        # Redimensionar si es necesario
        if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Ecualización de histograma para mejorar contraste
        image = cv2.equalizeHist(image)

        return image

    def train(self, max_samples_per_class=None):
        """
        Entrena el modelo con el dataset FER2013.

        Args:
            max_samples_per_class: Número máximo de muestras por clase (None = todas)
        """
        print("Entrenando modelo OpenCV...")
        X = []
        y = []

        for emotion_idx, emotion in enumerate(EMOTIONS):
            emotion_dir = os.path.join(TRAIN_DIR, emotion)
            if not os.path.exists(emotion_dir):
                print(f"  Advertencia: No se encontró directorio {emotion_dir}")
                continue

            image_files = os.listdir(emotion_dir)

            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]

            print(f"  Procesando {emotion}: {len(image_files)} imágenes")

            for img_file in tqdm(image_files, desc=f"  {emotion}", leave=False):
                img_path = os.path.join(emotion_dir, img_file)
                image = self.preprocess_image(img_path)

                if image is not None:
                    features = self.extract_features(image)
                    X.append(features)
                    y.append(emotion_idx)

        X = np.array(X)
        y = np.array(y)

        print(f"  Total de muestras: {len(X)}")
        print(f"  Dimensión de características: {X.shape[1]}")

        # Normalizar características
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar clasificador SVM
        print("  Entrenando clasificador SVM...")
        self.classifier = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42
        )
        self.classifier.fit(X_scaled, y)

        self.is_trained = True

        # Guardar modelo
        self.save_model()
        print("  Modelo entrenado y guardado.")

    def predict(self, image_path):
        """
        Predice la emoción de una imagen facial.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Tupla (emoción_predicha, probabilidades)
        """
        if not self.is_trained:
            self.load_model()
            if not self.is_trained:
                raise ValueError("El modelo no está entrenado. Ejecute train() primero.")

        image = self.preprocess_image(image_path)

        if image is None:
            return None, None

        features = self.extract_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]

        return EMOTIONS[prediction], dict(zip(EMOTIONS, probabilities))

    def predict_batch(self, image_paths):
        """
        Predice emociones para múltiples imágenes.

        Args:
            image_paths: Lista de rutas a imágenes

        Returns:
            Lista de predicciones
        """
        predictions = []

        for path in tqdm(image_paths, desc="OpenCV prediciendo"):
            emotion, probs = self.predict(path)
            predictions.append({
                'path': path,
                'prediction': emotion,
                'probabilities': probs
            })

        return predictions

    def save_model(self):
        """Guarda el modelo entrenado."""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)

        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"  Modelo guardado en {self.model_path}")

    def load_model(self):
        """Carga un modelo previamente entrenado."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)

            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            self.is_trained = True
            print("Modelo OpenCV cargado desde archivo.")
            return True

        return False


def test_opencv_model():
    """Función de prueba del modelo OpenCV."""
    model = OpenCVEmotionRecognizer()

    # Entrenar con muestra pequeña
    model.train(max_samples_per_class=100)

    # Probar con una imagen
    test_emotion = 'happy'
    test_dir = os.path.join(TEST_DIR, test_emotion)
    if os.path.exists(test_dir):
        test_images = os.listdir(test_dir)[:5]

        print("\nPruebas de predicción:")
        for img in test_images:
            img_path = os.path.join(test_dir, img)
            pred, probs = model.predict(img_path)
            print(f"  Imagen: {img}")
            print(f"  Real: {test_emotion}, Predicho: {pred}")
            print(f"  Probabilidades: {probs}")
            print()


if __name__ == "__main__":
    test_opencv_model()
