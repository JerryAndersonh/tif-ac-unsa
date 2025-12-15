#!/usr/bin/env python3
"""
Proyecto: Evaluación Comparativa de Modelos de Reconocimiento Facial de Emociones
FER (Mini-Xception) vs DeepFace usando el dataset FER2013

Este script ejecuta la comparación completa entre ambos modelos PREENTRENADOS
y genera reportes con métricas de evaluación y visualizaciones.

Ambos modelos son preentrenados para una comparación justa.

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
from src.fer_model import FEREmotionRecognizer
from src.deepface_model import DeepFaceEmotionRecognizer
from src.evaluation import ModelEvaluator


def print_header():
    """Imprime el encabezado del programa."""
    print("=" * 70)
    print("  EVALUACIÓN COMPARATIVA: RECONOCIMIENTO FACIAL DE EMOCIONES")
    print("  FER (Mini-Xception) vs DeepFace - Dataset FER2013")
    print("  Ambos modelos PREENTRENADOS")
    print("=" * 70)
    print(f"  Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def run_comparison(sample_size=100, skip_fer=False, skip_deepface=False):
    """
    Ejecuta la comparación completa entre modelos.

    Args:
        sample_size: Número de imágenes por emoción para test
        skip_fer: Saltar evaluación de FER
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
    # EVALUACIÓN FER (Mini-Xception) - PREENTRENADO
    # =========================================================================
    fer_predictions = None

    if not skip_fer:
        print("\n" + "=" * 70)
        print("EVALUACIÓN DEL MODELO FER (Mini-Xception)")
        print("Modelo PREENTRENADO - No requiere entrenamiento")
        print("=" * 70)

        fer_model = FEREmotionRecognizer()

        # Evaluar
        print("\nEvaluando modelo FER...")
        fer_start = time.time()
        fer_results = fer_model.predict_batch(test_images)
        fer_time = time.time() - fer_start
        processing_times['FER'] = fer_time

        fer_predictions = [r['prediction'] for r in fer_results]
        evaluator.evaluate_predictions(test_labels, fer_predictions, 'FER')

        print(f"Tiempo de evaluación: {fer_time:.2f} segundos")
        print(f"Velocidad: {len(test_images)/fer_time:.2f} imágenes/segundo")

    # =========================================================================
    # EVALUACIÓN DEEPFACE - PREENTRENADO
    # =========================================================================
    deepface_predictions = None

    if not skip_deepface:
        print("\n" + "=" * 70)
        print("EVALUACIÓN DEL MODELO DEEPFACE")
        print("Modelo PREENTRENADO - No requiere entrenamiento")
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
        'fer_predictions': fer_predictions,
        'deepface_predictions': deepface_predictions,
        'test_labels': test_labels
    }


def run_quick_test():
    """Ejecuta una prueba rápida con pocas imágenes."""
    print("Ejecutando prueba rápida (10 imágenes por emoción)...")
    return run_comparison(sample_size=10)


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
        '--skip-fer',
        action='store_true',
        help='Saltar evaluación de FER'
    )
    parser.add_argument(
        '--skip-deepface',
        action='store_true',
        help='Saltar evaluación de DeepFace'
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        run_comparison(
            sample_size=args.sample_size,
            skip_fer=args.skip_fer,
            skip_deepface=args.skip_deepface
        )


if __name__ == "__main__":
    main()
