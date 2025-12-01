#!/usr/bin/env python3
"""
Script de prueba individual para los modelos.
Útil para verificar que cada modelo funciona correctamente.
"""
import os
import sys

# Agregar directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TEST_DIR, EMOTIONS
from src.data_loader import DataLoader


def test_data_loader():
    """Prueba el cargador de datos."""
    print("=" * 50)
    print("PRUEBA: DataLoader")
    print("=" * 50)

    loader = DataLoader()
    loader.print_dataset_info()

    # Cargar muestra pequeña
    images, labels = loader.load_test_images(sample_size_per_emotion=5)
    print(f"\nMuestra cargada: {len(images)} imágenes")
    print("Primeras 3 imágenes:")
    for img, label in zip(images[:3], labels[:3]):
        print(f"  {label}: {os.path.basename(img)}")

    return True


def test_opencv_model():
    """Prueba el modelo OpenCV."""
    print("\n" + "=" * 50)
    print("PRUEBA: OpenCV Model")
    print("=" * 50)

    try:
        from src.opencv_model import OpenCVEmotionRecognizer

        model = OpenCVEmotionRecognizer()

        # Entrenar con muestra pequeña
        print("Entrenando con 50 imágenes por clase...")
        model.train(max_samples_per_class=50)

        # Probar predicción
        print("\nProbando predicciones:")
        for emotion in ['happy', 'sad', 'angry']:
            test_dir = os.path.join(TEST_DIR, emotion)
            if os.path.exists(test_dir):
                images = os.listdir(test_dir)[:3]
                for img in images:
                    img_path = os.path.join(test_dir, img)
                    pred, probs = model.predict(img_path)
                    print(f"  Real: {emotion}, Predicho: {pred}")

        print("\nOpenCV: OK")
        return True

    except Exception as e:
        print(f"Error en OpenCV: {e}")
        return False


def test_deepface_model():
    """Prueba el modelo DeepFace."""
    print("\n" + "=" * 50)
    print("PRUEBA: DeepFace Model")
    print("=" * 50)

    try:
        from src.deepface_model import DeepFaceEmotionRecognizer

        model = DeepFaceEmotionRecognizer()

        # Probar predicción
        print("Probando predicciones:")
        for emotion in ['happy', 'sad', 'angry']:
            test_dir = os.path.join(TEST_DIR, emotion)
            if os.path.exists(test_dir):
                images = os.listdir(test_dir)[:2]
                for img in images:
                    img_path = os.path.join(test_dir, img)
                    pred, probs = model.predict(img_path)
                    print(f"  Real: {emotion}, Predicho: {pred}")

        print("\nDeepFace: OK")
        return True

    except Exception as e:
        print(f"Error en DeepFace: {e}")
        return False


def test_evaluation():
    """Prueba el módulo de evaluación."""
    print("\n" + "=" * 50)
    print("PRUEBA: Evaluation Module")
    print("=" * 50)

    try:
        from src.evaluation import ModelEvaluator

        evaluator = ModelEvaluator()

        # Datos de prueba sintéticos
        y_true = ['happy', 'sad', 'angry', 'happy', 'sad', 'angry'] * 5
        y_pred_model1 = ['happy', 'sad', 'angry', 'happy', 'happy', 'angry'] * 5
        y_pred_model2 = ['happy', 'happy', 'angry', 'happy', 'sad', 'sad'] * 5

        # Evaluar
        metrics1 = evaluator.evaluate_predictions(y_true, y_pred_model1, 'Modelo1')
        metrics2 = evaluator.evaluate_predictions(y_true, y_pred_model2, 'Modelo2')

        print(f"Modelo1 Accuracy: {metrics1['accuracy']:.4f}")
        print(f"Modelo2 Accuracy: {metrics2['accuracy']:.4f}")

        # Generar reporte
        print("\nGenerando reporte...")
        report = evaluator.generate_report()
        print("Reporte generado correctamente")

        print("\nEvaluation: OK")
        return True

    except Exception as e:
        print(f"Error en Evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Ejecuta todas las pruebas."""
    print("=" * 60)
    print("EJECUTANDO TODAS LAS PRUEBAS")
    print("=" * 60)

    results = {
        'DataLoader': test_data_loader(),
        'Evaluation': test_evaluation(),
        'OpenCV': test_opencv_model(),
        'DeepFace': test_deepface_model()
    }

    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} pruebas pasadas")

    return all(results.values())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pruebas de los modelos')
    parser.add_argument('--all', action='store_true', help='Ejecutar todas las pruebas')
    parser.add_argument('--opencv', action='store_true', help='Probar solo OpenCV')
    parser.add_argument('--deepface', action='store_true', help='Probar solo DeepFace')
    parser.add_argument('--data', action='store_true', help='Probar solo DataLoader')
    parser.add_argument('--eval', action='store_true', help='Probar solo Evaluation')

    args = parser.parse_args()

    if args.opencv:
        test_opencv_model()
    elif args.deepface:
        test_deepface_model()
    elif args.data:
        test_data_loader()
    elif args.eval:
        test_evaluation()
    else:
        run_all_tests()
