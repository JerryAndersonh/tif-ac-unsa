"""
Módulo de carga y preparación de datos del dataset FER2013.
"""
import os
import random
from .config import EMOTIONS, TRAIN_DIR, TEST_DIR, RANDOM_SEED


class DataLoader:
    """
    Cargador de datos para el dataset FER2013.
    """

    def __init__(self, random_seed=RANDOM_SEED):
        """
        Inicializa el cargador de datos.

        Args:
            random_seed: Semilla para reproducibilidad
        """
        self.random_seed = random_seed
        random.seed(random_seed)

    def get_dataset_info(self):
        """
        Obtiene información sobre el dataset.

        Returns:
            Diccionario con estadísticas del dataset
        """
        info = {
            'train': {},
            'test': {},
            'total_train': 0,
            'total_test': 0
        }

        for emotion in EMOTIONS:
            train_dir = os.path.join(TRAIN_DIR, emotion)
            test_dir = os.path.join(TEST_DIR, emotion)

            train_count = len(os.listdir(train_dir)) if os.path.exists(train_dir) else 0
            test_count = len(os.listdir(test_dir)) if os.path.exists(test_dir) else 0

            info['train'][emotion] = train_count
            info['test'][emotion] = test_count
            info['total_train'] += train_count
            info['total_test'] += test_count

        return info

    def load_test_images(self, sample_size_per_emotion=None, shuffle=True):
        """
        Carga las rutas de imágenes de test.

        Args:
            sample_size_per_emotion: Número de imágenes por emoción (None = todas)
            shuffle: Si se deben mezclar las imágenes

        Returns:
            Tupla (rutas_imagenes, etiquetas)
        """
        image_paths = []
        labels = []

        for emotion in EMOTIONS:
            test_dir = os.path.join(TEST_DIR, emotion)

            if not os.path.exists(test_dir):
                print(f"Advertencia: No se encontró {test_dir}")
                continue

            images = os.listdir(test_dir)

            if shuffle:
                random.shuffle(images)

            if sample_size_per_emotion:
                images = images[:sample_size_per_emotion]

            for img in images:
                image_paths.append(os.path.join(test_dir, img))
                labels.append(emotion)

        # Mezclar dataset completo
        if shuffle:
            combined = list(zip(image_paths, labels))
            random.shuffle(combined)
            image_paths, labels = zip(*combined) if combined else ([], [])

        return list(image_paths), list(labels)

    def load_train_images(self, sample_size_per_emotion=None, shuffle=True):
        """
        Carga las rutas de imágenes de entrenamiento.

        Args:
            sample_size_per_emotion: Número de imágenes por emoción (None = todas)
            shuffle: Si se deben mezclar las imágenes

        Returns:
            Tupla (rutas_imagenes, etiquetas)
        """
        image_paths = []
        labels = []

        for emotion in EMOTIONS:
            train_dir = os.path.join(TRAIN_DIR, emotion)

            if not os.path.exists(train_dir):
                print(f"Advertencia: No se encontró {train_dir}")
                continue

            images = os.listdir(train_dir)

            if shuffle:
                random.shuffle(images)

            if sample_size_per_emotion:
                images = images[:sample_size_per_emotion]

            for img in images:
                image_paths.append(os.path.join(train_dir, img))
                labels.append(emotion)

        if shuffle:
            combined = list(zip(image_paths, labels))
            random.shuffle(combined)
            image_paths, labels = zip(*combined) if combined else ([], [])

        return list(image_paths), list(labels)

    def print_dataset_info(self):
        """Imprime información del dataset."""
        info = self.get_dataset_info()

        print("=" * 50)
        print("INFORMACIÓN DEL DATASET FER2013")
        print("=" * 50)

        print("\nImágenes de ENTRENAMIENTO por emoción:")
        for emotion, count in info['train'].items():
            print(f"  {emotion.capitalize():10s}: {count:5d} imágenes")
        print(f"  {'TOTAL':10s}: {info['total_train']:5d} imágenes")

        print("\nImágenes de TEST por emoción:")
        for emotion, count in info['test'].items():
            print(f"  {emotion.capitalize():10s}: {count:5d} imágenes")
        print(f"  {'TOTAL':10s}: {info['total_test']:5d} imágenes")

        print("=" * 50)


if __name__ == "__main__":
    loader = DataLoader()
    loader.print_dataset_info()
