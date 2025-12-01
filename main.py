#!/usr/bin/env python3
"""
Proyecto: Evaluación Comparativa de Modelos de Reconocimiento Facial de Emociones
OpenCV vs DeepFace usando el dataset FER2013

Este script ejecuta la comparación completa entre ambos modelos y genera
reportes con métricas de evaluación y visualizaciones.

Autor: [Tu nombre]
Fecha: 2025
"""
import os
import sys
import time
import argparse
from datetime import datetime

# Agregar directorio src al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import RESULTS_DIR, SAMPLE_SIZE_PER_EMOTION
from src.data_loader import DataLoader
from src.opencv_model import OpenCVEmotionRecognizer
from src.deepface_model import DeepFaceEmotionRecognizer
from src.evaluation import ModelEvaluator


def print_header():
    """Imprime el encabezado del programa."""
    print("=" * 70)
    print("  EVALUACIÓN COMPARATIVA: RECONOCIMIENTO FACIAL DE EMOCIONES")
    print("  OpenCV vs DeepFace - Dataset FER2013")
    print("=" * 70)
    print(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def run_comparison(sample_size=100, train_opencv=True, skip_opencv=False, skip_deepface=False):
    """
    Ejecuta la comparación completa entre modelos.

    Args:
        sample_size: Número de imágenes por emoción para test
        train_opencv: Si se debe entrenar el modelo OpenCV
        skip_opencv: Saltar evaluación de OpenCV
        skip_deepface: Saltar evaluación de DeepFace

    Returns:
        Diccionario con resultados
    """
    print_header()

    # Inicializar componentes
    data_loader = DataLoader()
    evaluator = ModelEvaluator()
    processing_times = {}

    # Mostrar información del dataset
    data_loader.print_dataset_info()

    # Cargar imágenes de test
    print(f"\nCargando {sample_size} imágenes por emoción para evaluación...")
    test_images, test_labels = data_loader.load_test_images(
        sample_size_per_emotion=sample_size
    )
    print(f"Total de imágenes de test: {len(test_images)}")

    # =========================================================================
    # EVALUACIÓN OPENCV
    # =========================================================================
    opencv_predictions = None

    if not skip_opencv:
        print("\n" + "=" * 70)
        print("EVALUACIÓN DEL MODELO OPENCV")
        print("=" * 70)

        opencv_model = OpenCVEmotionRecognizer()

        # Entrenar o cargar modelo
        if train_opencv:
            print("\nEntrenando modelo OpenCV...")
            train_start = time.time()
            opencv_model.train(max_samples_per_class=500)  # Entrenar con 500 por clase
            train_time = time.time() - train_start
            print(f"Tiempo de entrenamiento: {train_time:.2f} segundos")
        else:
            if not opencv_model.load_model():
                print("No se encontró modelo guardado. Entrenando...")
                opencv_model.train(max_samples_per_class=500)

        # Evaluar
        print("\nEvaluando modelo OpenCV...")
        opencv_start = time.time()
        opencv_results = opencv_model.predict_batch(test_images)
        opencv_time = time.time() - opencv_start
        processing_times['OpenCV'] = opencv_time

        opencv_predictions = [r['prediction'] for r in opencv_results]
        evaluator.evaluate_predictions(test_labels, opencv_predictions, 'OpenCV')

        print(f"Tiempo de evaluación: {opencv_time:.2f} segundos")
        print(f"Velocidad: {len(test_images)/opencv_time:.2f} imágenes/segundo")

    # =========================================================================
    # EVALUACIÓN DEEPFACE
    # =========================================================================
    deepface_predictions = None

    if not skip_deepface:
        print("\n" + "=" * 70)
        print("EVALUACIÓN DEL MODELO DEEPFACE")
        print("=" * 70)

        deepface_model = DeepFaceEmotionRecognizer()

        print("\nEvaluando modelo DeepFace...")
        deepface_start = time.time()
        deepface_results = deepface_model.predict_batch(test_images)
        deepface_time = time.time() - deepface_start
        processing_times['DeepFace'] = deepface_time

        deepface_predictions = [r['prediction'] for r in deepface_results]
        evaluator.evaluate_predictions(test_labels, deepface_predictions, 'DeepFace')

        print(f"Tiempo de evaluación: {deepface_time:.2f} segundos")
        print(f"Velocidad: {len(test_images)/deepface_time:.2f} imágenes/segundo")

    # =========================================================================
    # GENERACIÓN DE REPORTES Y VISUALIZACIONES
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERANDO REPORTES Y VISUALIZACIONES")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generar visualizaciones
    if evaluator.results:
        # Matrices de confusión
        cm_path = os.path.join(RESULTS_DIR, f'confusion_matrices_{timestamp}.png')
        evaluator.plot_confusion_matrices(save_path=cm_path)

        # Comparación de métricas
        metrics_path = os.path.join(RESULTS_DIR, f'metrics_comparison_{timestamp}.png')
        evaluator.plot_metrics_comparison(save_path=metrics_path)

        # Comparación por emoción
        emotion_path = os.path.join(RESULTS_DIR, f'per_emotion_comparison_{timestamp}.png')
        evaluator.plot_per_emotion_comparison(save_path=emotion_path)

        # Tiempos de procesamiento
        if processing_times:
            time_path = os.path.join(RESULTS_DIR, f'processing_times_{timestamp}.png')
            evaluator.plot_processing_time(processing_times, save_path=time_path)

        # Guardar resultados
        evaluator.save_results(f'comparison_results_{timestamp}')

        # Imprimir reporte
        print("\n" + evaluator.generate_report(processing_times))

    print("\n" + "=" * 70)
    print("EVALUACIÓN COMPLETADA")
    print(f"Los resultados se han guardado en: {RESULTS_DIR}")
    print("=" * 70)

    return {
        'evaluator': evaluator,
        'processing_times': processing_times,
        'opencv_predictions': opencv_predictions,
        'deepface_predictions': deepface_predictions,
        'test_labels': test_labels
    }


def run_quick_test():
    """Ejecuta una prueba rápida con pocas imágenes."""
    print("Ejecutando prueba rápida (10 imágenes por emoción)...")
    return run_comparison(sample_size=10, train_opencv=True)


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Evaluación comparativa de modelos de reconocimiento de emociones'
    )
    parser.add_argument(
        '--sample-size', '-s',
        type=int,
        default=100,
        help='Número de imágenes por emoción para evaluar (default: 100)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Ejecutar prueba rápida (10 imágenes por emoción)'
    )
    parser.add_argument(
        '--skip-opencv',
        action='store_true',
        help='Saltar evaluación de OpenCV'
    )
    parser.add_argument(
        '--skip-deepface',
        action='store_true',
        help='Saltar evaluación de DeepFace'
    )
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='No entrenar OpenCV (cargar modelo guardado)'
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_comparison(
            sample_size=args.sample_size,
            train_opencv=not args.no_train,
            skip_opencv=args.skip_opencv,
            skip_deepface=args.skip_deepface
        )


if __name__ == "__main__":
    main()
