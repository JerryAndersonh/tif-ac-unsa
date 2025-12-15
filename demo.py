#!/usr/bin/env python3
"""
Script de demostración para analizar una imagen específica
con ambos modelos y mostrar los resultados.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import TEST_DIR, EMOTIONS
from src.fer_model import FEREmotionRecognizer
from src.deepface_model import DeepFaceEmotionRecognizer


def analyze_image(image_path):
    """
    Analiza una imagen con ambos modelos.

    Args:
        image_path: Ruta a la imagen
    """
    print("=" * 60)
    print("ANÁLISIS DE IMAGEN")
    print(f"Archivo: {image_path}")
    print("=" * 60)

    if not os.path.exists(image_path):
        print(f"Error: No se encontró la imagen {image_path}")
        return

    # FER (Mini-Xception)
    print("\n--- MODELO FER (Mini-Xception) ---")
    try:
        fer_model = FEREmotionRecognizer()
        pred_fer, probs_fer = fer_model.predict(image_path)
        print(f"Emoción detectada: {pred_fer}")
        if probs_fer:
            print("Probabilidades:")
            for emotion, prob in sorted(probs_fer.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"  {emotion:10s}: {prob:.4f} {bar}")
    except Exception as e:
        print(f"Error con FER: {e}")

    # DeepFace
    print("\n--- MODELO DEEPFACE ---")
    try:
        deepface_model = DeepFaceEmotionRecognizer()
        pred_deepface, probs_deepface = deepface_model.predict(image_path)
        print(f"Emoción detectada: {pred_deepface}")
        if probs_deepface:
            print("Probabilidades:")
            for emotion, prob in sorted(probs_deepface.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"  {emotion:10s}: {prob:.4f} {bar}")
    except Exception as e:
        print(f"Error con DeepFace: {e}")

    print("\n" + "=" * 60)


def analyze_random_samples():
    """Analiza muestras aleatorias del dataset de test."""
    import random

    print("=" * 60)
    print("ANÁLISIS DE MUESTRAS ALEATORIAS DEL DATASET")
    print("=" * 60)

    # Cargar modelos
    print("\nCargando modelos...")

    fer_model = FEREmotionRecognizer()
    deepface_model = DeepFaceEmotionRecognizer()

    # Seleccionar muestras
    for emotion in ['happy', 'sad', 'angry', 'surprise']:
        test_dir = os.path.join(TEST_DIR, emotion)
        if not os.path.exists(test_dir):
            continue

        images = os.listdir(test_dir)
        if images:
            sample = random.choice(images)
            img_path = os.path.join(test_dir, sample)

            print(f"\n{'='*50}")
            print(f"Emoción real: {emotion.upper()}")
            print(f"Archivo: {sample}")
            print("-" * 50)

            # FER
            pred_fer, _ = fer_model.predict(img_path)
            match_fer = "✓" if pred_fer == emotion else "✗"
            print(f"FER:      {pred_fer:10s} {match_fer}")

            # DeepFace
            pred_df, _ = deepface_model.predict(img_path)
            match_df = "✓" if pred_df == emotion else "✗"
            print(f"DeepFace: {pred_df:10s} {match_df}")


def main():
    parser = argparse.ArgumentParser(
        description='Demo de análisis de emociones faciales'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Ruta a una imagen para analizar'
    )
    parser.add_argument(
        '--random', '-r',
        action='store_true',
        help='Analizar muestras aleatorias del dataset'
    )

    args = parser.parse_args()

    if args.image:
        analyze_image(args.image)
    elif args.random:
        analyze_random_samples()
    else:
        print("Uso: python demo.py --image <ruta_imagen>")
        print("  o: python demo.py --random")
        print("\nEjecutando demo con muestras aleatorias...")
        analyze_random_samples()


if __name__ == "__main__":
    main()
